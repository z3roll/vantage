"""Progressive controller: improvement-first per-destination PoP assignment.

For each (cell, destination), picks the PoP that minimises
``sat_cost + ground_cost`` while respecting per-PoP capacity. Items
are processed in *expected RTT-saving* order so limited capacity
goes to the cells that benefit most — not to those that already
have low absolute latency.

Algorithm:

1. Rank all reachable PoPs per (cell, dest) by E2E cost
   (``sat_cost + ground_cost``).
2. For each (cell, dest), aggregate the current demand across every
   endpoint in the cell going to ``dest``, and compute the
   *improvement* delta:

       improvement = baseline_E2E_cost - best_alt_E2E_cost

   where ``baseline_pop`` is the cell's geographic nearest PoP. Skip
   pairs with non-positive improvement or zero demand — the data
   plane will route them via the baseline default at lookup time.
3. Sort the surviving pairs by ``improvement × demand`` descending
   — biggest aggregate RTT saving first.
4. Greedy first-fit through each pair's ranking, respecting per-PoP
   capacity. On overflow (no candidate PoP has remaining capacity),
   fall back to the cell's nearest PoP. Each cell's nearest differs,
   so overflow scatters geographically rather than concentrating on
   the most popular E2E-best PoP.

This is value-density greedy for a multi-knapsack / generalised
assignment problem (NP-hard). Empirically within ~10–20% of the
LP-relaxation optimum at our scale (~10⁴ cells × ~10¹ services × ~10²
PoPs); fast enough for the 15 s control-plane refresh budget.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable

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

    def _ground_cost(self, pop_code: str, dest: str) -> float:
        """Resolve ground RTT for ``(pop_code, dest)`` — fail loud if
        unknown.

        Pre-2026-04-17 this swallowed ``KeyError`` and returned
        ``None``, causing ``rank_pops_by_e2e`` to silently drop the
        PoP from the ranking. The new contract: ``ground_knowledge``
        must serve every (PoP, dest) the controller plans against;
        a missing pair surfaces as a ``KeyError`` so the operator
        notices instead of the algorithm quietly degrading.
        """
        return self._gk.get_or_estimate(pop_code, dest)

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

        # 1. Baseline: nearest PoP per cell. Used as both the data-plane
        #    fallback for cells without overrides AND the reference
        #    cost against which "improvement" is measured.
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid, pops=pops,
            built_at=snapshot.time_s, version=version,
        )

        # 2. Rank all reachable PoPs per (cell, dest) by E2E cost.
        cell_sat_cost = compute_cell_sat_cost(snapshot, cell_grid)
        rankings = rank_pops_by_e2e(
            cell_grid=cell_grid,
            pops=pops,
            baseline=baseline,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=self._ground_cost,
            dest_names=self.resolve_dest_names(),
        )

        # 3. Improvement-first greedy assignment.
        assignments = _progressive_filling(
            rankings=rankings,
            baseline=baseline,
            cell_grid=cell_grid,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=self._ground_cost,
            demand_per_pair=demand_per_pair or {},
            pop_capacity_gbps=pop_capacity_gbps or {},
        )

        # 4. Assemble RoutingPlane. Only emit overrides where the
        # assigned PoP differs from baseline; cells absent from
        # `assignments` (skipped or no improvement) get baseline
        # automatically via `CellToPopTable.pop_of(cell)`.
        per_dest_overrides: dict[tuple[int, str], str] = {}
        for (cell_id, dest), pop_code in assignments.items():
            if pop_code != baseline.mapping.get(cell_id):
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
    cell_sat_cost: dict[tuple[int, str], float],
    ground_cost_fn: Callable[[str, str], float],
    demand_per_pair: dict[tuple[str, str], float],
    pop_capacity_gbps: dict[str, float],
) -> dict[tuple[int, str], str]:
    """Greedy first-fit assignment ordered by ``improvement × demand``.

    For each (cell, dest) in ``rankings``:

    * Aggregate ``demand`` across every endpoint hosted by this cell
      that targets ``dest`` (a single cell can host many endpoints).
    * ``baseline_cost`` = sat-segment + ground-segment RTT through
      the cell's geographic-nearest PoP (``baseline.mapping[cell]``).
    * ``improvement`` = ``baseline_cost - best_alt_cost`` where
      ``best_alt`` is the top of the ranking.

    Skip (no override emitted; data plane falls back to baseline at
    lookup time):

    * ``demand <= 0``: no traffic to allocate.
    * ``improvement <= 0``: baseline already optimal — no win
      available to claim.
    * Baseline PoP not sat-reachable from this cell (rare): can't
      quantify improvement, leave the cell on the baseline default.

    Surviving items are processed in descending order of
    ``improvement × demand``. Each picks the first PoP in its ranking
    with remaining capacity. **Overflow** (no candidate has room): the
    cell falls back to its geographic nearest PoP. Each cell's
    nearest differs, so overflow scatters by geography rather than
    piling onto the most popular E2E-best PoP. The fallback still
    increments ``pop_load[nearest]`` so downstream cells considering
    the same nearest see the realised overflow.

    Returns ``{(cell, dest) → pop}`` for explicitly assigned cells.
    Cells absent from the result get baseline via
    :meth:`CellToPopTable.pop_of` at data-plane time.
    """
    if not rankings:
        return {}

    # ── per-cell endpoint index (for demand aggregation) ──
    cell_to_eps: dict[int, list[str]] = {}
    for ep_name, ep_cell in cell_grid.endpoint_to_cell.items():
        cell_to_eps.setdefault(ep_cell, []).append(ep_name)

    # ── Build the work queue of (priority, cell, dest, demand) ──
    # ``priority`` = ``improvement × demand``; we negate so a plain
    # ascending sort orders by priority descending.
    queue: list[tuple[float, int, str, float]] = []
    for (cell_id, dest), ranked in rankings.items():
        if not ranked:
            continue
        demand = sum(
            demand_per_pair.get((ep_name, dest), 0.0)
            for ep_name in cell_to_eps.get(cell_id, ())
        )
        if demand <= 0.0:
            continue
        baseline_pop = baseline.mapping[cell_id]
        baseline_sat = cell_sat_cost.get((cell_id, baseline_pop))
        if baseline_sat is None:
            # Baseline PoP not sat-reachable from this cell: extremely
            # rare (would imply the geographic nearest is in a
            # satellite-blind region). Skip and let data plane keep
            # the baseline default — we have no quantitative basis
            # for spending capacity on this pair.
            continue
        baseline_cost = baseline_sat + ground_cost_fn(baseline_pop, dest)
        best_alt_cost = ranked[0][1]
        improvement = baseline_cost - best_alt_cost
        if improvement <= 0.0:
            continue
        queue.append((-(improvement * demand), cell_id, dest, demand))

    # Tuple sort: primary key is the negated priority. Subsequent
    # tuple elements (cell_id int, dest str) provide a stable,
    # deterministic tie-break across runs with identical inputs.
    queue.sort()

    # ── Greedy first-fit by improvement × demand ──
    pop_load: dict[str, float] = {}
    assignments: dict[tuple[int, str], str] = {}
    for _priority, cell_id, dest, demand in queue:
        ranked = rankings[(cell_id, dest)]
        assigned = False
        for pop_code, _cost in ranked:
            cap = pop_capacity_gbps.get(pop_code, float("inf"))
            current = pop_load.get(pop_code, 0.0)
            if current + demand <= cap:
                assignments[(cell_id, dest)] = pop_code
                pop_load[pop_code] = current + demand
                assigned = True
                break

        if not assigned:
            # Overflow: degrade gracefully to the cell's geographic
            # nearest PoP (= baseline). Spreads overflow naturally
            # because each cell's nearest differs. Recording the
            # assignment in `pop_load` keeps later cells' capacity
            # view consistent if they consider the same PoP — but
            # the assignment will not be written as an override
            # because it equals baseline.
            nearest = baseline.mapping[cell_id]
            assignments[(cell_id, dest)] = nearest
            pop_load[nearest] = pop_load.get(nearest, 0.0) + demand

    return assignments
