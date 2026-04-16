"""Progressive controller: E2E-aware per-destination PoP assignment.

For each (cell, destination), picks the PoP that minimizes
``sat_cost + ground_cost`` while respecting PoP capacity constraints
(GS feeder aggregate).

Progressive Filling:
1. Rank all PoPs per (cell, dest) by E2E cost.
2. Compute per-(cell, dest) demand from current traffic.
3. For ALL (cell, dest) pairs, assign to rank-1 PoP; if capacity
   exceeded, try rank-2, rank-3, etc.
4. No flow is left at a default PoP if that PoP is over capacity —
   every flow is subject to capacity constraints.
"""

from __future__ import annotations

import logging
from types import MappingProxyType

from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_satellite_fibs,
    compute_cell_sat_cost,
    rank_pops_by_e2e,
)
from vantage.control.policy.common.sat_cost import precompute_per_sat_routing
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    NetworkSnapshot,
    RoutingPlane,
)
from vantage.world.ground import GroundKnowledge

_log = logging.getLogger(__name__)


class ProgressiveController:
    """E2E-aware PoP selection with Progressive Filling."""

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
        dest_names: tuple[str, ...] = (),
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()
        self._dest_names = dest_names
        # We deliberately do NOT warn at construction time: a freshly
        # built GK is expected to be empty until the feedback loop
        # populates it across epochs. The warning lives inside
        # `resolve_dest_names` and fires once per controller if a
        # `compute_routing_plane` call actually finds no destinations.
        self._warned_no_dests = False

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    def resolve_dest_names(self) -> tuple[str, ...]:
        """Pick the destination set to plan against right now.

        Explicit ``dest_names`` (constructor arg) wins. Otherwise we
        derive from the :class:`GroundKnowledge` cache so a controller
        seeded only with ``ground_knowledge`` still produces real
        per-dest overrides instead of silently degrading to the
        nearest-PoP baseline. Returns an empty tuple if neither source
        has any destinations — and emits a single warning the first
        time that happens, so degraded behaviour is visible without
        spamming the log every epoch.
        """
        if self._dest_names:
            return self._dest_names
        derived = tuple(sorted({dest for _, dest in self._gk.all_entries()}))
        if not derived and not self._warned_no_dests:
            _log.warning(
                "ProgressiveController.resolve_dest_names: no explicit "
                "dest_names and ground_knowledge has no entries; "
                "degrading to nearest-PoP baseline this epoch."
            )
            self._warned_no_dests = True
        return derived

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        demand_per_pair: dict[tuple[str, str], float] | None = None,
        pop_capacity_gbps: dict[str, float] | None = None,
        version: int = 0,
    ) -> RoutingPlane:
        pops = snapshot.infra.pops

        # 1. Baseline: nearest PoP per cell (fallback for flows without ranking)
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid, pops=pops,
            built_at=snapshot.time_s, version=version,
        )

        # 2. Rank PoPs per (cell, dest) by E2E cost
        #    Use get_or_estimate: cache hit → cached value,
        #    cache miss → estimator (active probing).
        #    This ensures ALL PoPs are considered, not just cached ones.
        def _ground_cost(pop_code: str, dest: str) -> float | None:
            try:
                return self._gk.get_or_estimate(pop_code, dest)
            except KeyError:
                return None

        cell_sat_cost = compute_cell_sat_cost(snapshot, cell_grid)
        rankings = rank_pops_by_e2e(
            cell_grid=cell_grid,
            pops=pops,
            baseline=baseline,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=_ground_cost,
            dest_names=self.resolve_dest_names(),
        )

        # 3. Progressive Filling — ALL flows subject to capacity
        assignments = _progressive_filling(
            rankings=rankings,
            baseline=baseline,
            cell_grid=cell_grid,
            demand_per_pair=demand_per_pair or {},
            pop_capacity_gbps=pop_capacity_gbps or {},
        )

        # 4. Assemble RoutingPlane
        # assignments contains ALL (cell, dest) → pop mappings.
        # Only emit overrides for those that differ from baseline.
        per_dest_overrides: dict[tuple[int, str], str] = {}
        for (cell_id, dest), pop_code in assignments.items():
            default_pop = baseline.mapping.get(cell_id)
            if pop_code != default_pop:
                per_dest_overrides[(cell_id, dest)] = pop_code

        cell_to_pop = CellToPopTable(
            mapping=baseline.mapping,
            version=version,
            built_at=snapshot.time_s,
            per_dest=MappingProxyType(per_dest_overrides),
        )
        per_sat = precompute_per_sat_routing(snapshot)
        sat_fibs = build_satellite_fibs(snapshot, per_sat, version=version)
        return RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_fibs=MappingProxyType(sat_fibs),
            version=version,
            built_at=snapshot.time_s,
        )


def _progressive_filling(
    rankings: dict[tuple[int, str], list[tuple[str, float]]],
    baseline: CellToPopTable,
    cell_grid: CellGrid,
    demand_per_pair: dict[tuple[str, str], float],
    pop_capacity_gbps: dict[str, float],
) -> dict[tuple[int, str], str]:
    """Assign ALL (cell, dest) to PoPs with capacity constraints.

    Every (cell, dest) with a ranking gets assigned to the best PoP
    that has remaining capacity. No flow gets a free pass at an
    overloaded PoP.

    Returns ``{(cell_id, dest) → pop_code}`` for ALL assigned flows.
    """
    if not rankings:
        return {}

    # ── Estimate per-(cell, dest) demand ──
    cell_to_eps: dict[int, list[str]] = {}
    for ep_name, ep_cell in cell_grid.endpoint_to_cell.items():
        cell_to_eps.setdefault(ep_cell, []).append(ep_name)

    cell_dest_demand: dict[tuple[int, str], float] = {}
    for (cell_id, dest) in rankings:
        # Aggregate demand across every endpoint hosted by this cell.
        # Earlier code took only the first non-zero entry, which silently
        # dropped traffic from co-located endpoints and let `pop_load`
        # exceed cap without the algorithm noticing.
        demand = 0.0
        for ep_name in cell_to_eps.get(cell_id, ()):
            demand += demand_per_pair.get((ep_name, dest), 0.0)
        cell_dest_demand[(cell_id, dest)] = demand

    # ── Sort ALL (cell, dest) by best E2E cost (lowest first) ──
    # This ensures flows with the best opportunities get assigned first.
    all_items: list[tuple[float, int, str]] = []
    for (cell_id, dest), ranked in rankings.items():
        if ranked:
            all_items.append((ranked[0][1], cell_id, dest))
    all_items.sort()  # lowest E2E cost first

    # ── Assign greedily: every flow tries PoPs in rank order ──
    pop_load: dict[str, float] = {}
    assignments: dict[tuple[int, str], str] = {}

    for _, cell_id, dest in all_items:
        demand = cell_dest_demand.get((cell_id, dest), 0.0)
        ranked = rankings[(cell_id, dest)]

        assigned = False
        for pop_code, cost in ranked:
            cap = pop_capacity_gbps.get(pop_code, float("inf"))
            current = pop_load.get(pop_code, 0.0)
            if current + demand <= cap:
                assignments[(cell_id, dest)] = pop_code
                pop_load[pop_code] = current + demand
                assigned = True
                break

        if not assigned:
            # All PoPs full — fall back to rank-1 (best E2E, accept overload)
            best_pop = ranked[0][0]
            assignments[(cell_id, dest)] = best_pop
            pop_load[best_pop] = pop_load.get(best_pop, 0.0) + demand

    return assignments
