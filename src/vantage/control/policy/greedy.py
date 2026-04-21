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
    compute_cell_ingress,
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
from vantage.forward import compute_egress_options
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

    def _ground_cost(self, pop_code: str, dest: str) -> float | None:
        """Resolve ground RTT for ``(pop_code, dest)`` from GK cache."""
        return self._gk.get(pop_code, dest)

    def _prime_ground_knowledge(
        self, snapshot: NetworkSnapshot, dest_names: Iterable[str],
    ) -> None:
        """Populate GK with an estimated RTT for every (pop, dest)
        pair not already cached.

        Without this pre-population the cache-only ``_ground_cost``
        starves the planner during cold-start: feedback only records
        PoPs that flows actually routed through (= the geographic
        nearest tier), so remote PoPs never get cache entries and PG
        can't rank them. A PG plan with only 2–3 rankable PoPs is
        worse than baseline — it emits a short cascade that strictly
        shrinks the data plane's fallback chain.

        The bootstrap fills every reachable (pop, dest) pair with an
        estimator-derived RTT (``GeographicGroundDelay`` = haversine
        + fiber detour). ``GroundDelayFeedback.observe`` then refines
        those bootstrap values into real measurements via EWMA as
        flows route through each pair over time. Already-cached
        pairs are left untouched — never overwrite a real
        measurement with an estimate.
        """
        estimator = self._gk.estimator
        if estimator is None:
            return
        for pop in snapshot.infra.pops:
            for dest in dest_names:
                if self._gk.get(pop.code, dest) is not None:
                    continue
                try:
                    rtt = estimator.estimate(pop.code, dest) * 2
                except KeyError:
                    continue
                self._gk.put(pop.code, dest, rtt)

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        demand_per_pair: dict[tuple[str, str], float] | None = None,
        sat_feeder_cap_gbps: float = 20.0,
        version: int = 0,
    ) -> RoutingPlane:
        pops = snapshot.infra.pops
        dest_names = self.resolve_dest_names()

        # 0. Bootstrap GK: ensure every (pop, dest) pair has at least
        # an estimator-derived RTT so the planner can rank ALL PoPs
        # from epoch 0, not just the handful feedback has observed.
        # Real measurements overwrite the estimates via EWMA as
        # GroundDelayFeedback ticks.
        self._prime_ground_knowledge(snapshot, dest_names)

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
            dest_names=dest_names,
        )

        # 3. Improvement-first greedy assignment with per-sat-feeder
        # capacity tracking. The egress-sat callback tells
        # `_progressive_filling` which sat a given (cell, pop) pair
        # will contend for so it can sum load against the 20 Gbps
        # per-Ka-feeder cap rather than the 160 Gbps per-PoP
        # aggregate. Data plane (RoutingPlaneForward) still has the
        # last word via its multi-egress reroute + baseline fallback
        # if the controller's plan proves wrong at realize time.
        cell_ingress = compute_cell_ingress(snapshot, cell_grid)
        cell_pop_egress = _build_egress_resolver(snapshot, cell_ingress)
        assignments = _progressive_filling(
            rankings=rankings,
            baseline=baseline,
            cell_grid=cell_grid,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=self._ground_cost,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair or {},
            sat_feeder_cap_gbps=sat_feeder_cap_gbps,
        )

        # 4. Assemble RoutingPlane. Emit a per_dest cascade for
        # *every* (cell, dest) the controller has rankings for —
        # not just those PG actively moved off baseline. This is
        # the negative-improvement fallback the user asked for:
        # when the chosen PoP saturates at realize time, the data
        # plane walks the rest of the E2E-sorted ranking (which
        # includes PoPs with negative improvement) least-bad-first,
        # rather than dumping into the geographic top-N tail.
        #
        # The cascade head is PG's chosen PoP if it planned an
        # alternate; otherwise it's the cell's geographic nearest
        # (= the same as baseline's head). Either way the tail is
        # all *other* reachable PoPs in E2E ASC order.
        per_dest_overrides: dict[tuple[int, str], tuple[str, ...]] = {}
        baseline_mapping = baseline.mapping
        for (cell_id, dest), ranked in rankings.items():
            if not ranked:
                continue
            chosen_pop = assignments.get((cell_id, dest))
            if chosen_pop is None:
                base_ranked = baseline_mapping.get(cell_id)
                if not base_ranked:
                    continue
                chosen_pop = base_ranked[0]
            tail = tuple(p for p, _ in ranked if p != chosen_pop)
            ranked_tuple = (chosen_pop,) + tail
            if ranked_tuple != baseline_mapping.get(cell_id, ()):
                per_dest_overrides[(cell_id, dest)] = ranked_tuple

        cell_to_pop = CellToPopTable(
            mapping=baseline_mapping,
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


def _build_egress_resolver(
    snapshot: NetworkSnapshot,
    cell_ingress: dict[int, int],
    *,
    k: int = 8,
) -> Callable[[int, str], tuple[int, ...]]:
    """Build a cached ``(cell, pop) → top-K egress sats`` callback.

    Wraps :func:`compute_egress_options` (the same helper the data
    plane uses) so the controller's capacity check considers every
    sat the data plane could route flows through. Cache key is
    ``(ingress, pop)`` — multiple cells sharing the same ingress
    reuse the same result.

    Returns a tuple of egress sat IDs ordered ascending by sat-segment
    RTT (the same order the data plane walks). Empty tuple means no
    reachable egress for this (cell, pop) — :func:`_progressive_filling`
    treats that pair as unassignable from this cell.
    """
    cache: dict[tuple[int, str], tuple[int, ...]] = {}

    def resolver(cell_id: int, pop_code: str) -> tuple[int, ...]:
        ingress = cell_ingress.get(cell_id)
        if ingress is None:
            return ()
        key = (ingress, pop_code)
        if key not in cache:
            opts = compute_egress_options(snapshot, ingress, pop_code, k=k)
            cache[key] = tuple(o.egress_sat for o in opts)
        return cache[key]

    return resolver


def _progressive_filling(
    rankings: dict[tuple[int, str], list[tuple[str, float]]],
    baseline: CellToPopTable,
    cell_grid: CellGrid,
    cell_sat_cost: dict[tuple[int, str], float],
    ground_cost_fn: Callable[[str, str], float | None],
    cell_pop_egress: Callable[[int, str], tuple[int, ...]],
    demand_per_pair: dict[tuple[str, str], float],
    sat_feeder_cap_gbps: float,
) -> dict[tuple[int, str], str]:
    """Capacity-aware greedy assignment with full-cascade walk.

    For each (cell, dest) with positive demand:

    * Aggregate ``demand`` across every endpoint hosted by this cell
      that targets ``dest`` (a single cell can host many endpoints).
    * Compute ``improvement`` = ``baseline_cost - best_alt_cost``
      using the cell's geographic-nearest PoP as baseline.
    * Items are processed in descending order of
      ``improvement × demand`` (negative-improvement items go last
      so positive-improvement cells claim scarce capacity first).

    For each item, walk the *full* E2E-sorted ranking and within
    each PoP walk *all* top-K egress sats: the first
    ``(pop, sat)`` whose ``sat_load`` plus this cell's aggregate
    demand stays under ``sat_feeder_cap_gbps`` wins. This matches
    the data plane's per-sat first-fit so plan-time and realize-time
    decisions stay aligned. Walking all K sats per PoP (instead of
    only the primary) lets the controller fill every Ka antenna at
    a GS before moving to the next-ranked PoP, eliminating the
    pathology where one PoP's primary sat saturates while its 7
    siblings sit empty.

    Last-resort fallback (every (pop, sat) in the ranking is full):
    pick the (pop, sat) with the *lowest current load ratio* and
    accept the overflow — distributes the unfittable demand evenly
    rather than always landing on the same hot spot. Mirrors the
    data plane's defensive ``charge`` tail.

    Returns ``{(cell, dest) → pop}`` for every (cell, dest) with
    positive demand. ``compute_routing_plane`` consumes this to
    assemble the per-dest cascade tuple.
    """
    if not rankings:
        return {}

    # ── per-cell endpoint index (for demand aggregation) ──
    cell_to_eps: dict[int, list[str]] = {}
    for ep_name, ep_cell in cell_grid.endpoint_to_cell.items():
        cell_to_eps.setdefault(ep_cell, []).append(ep_name)

    # ── Build the work queue of (priority, cell, dest, demand) ──
    # ``priority`` = ``improvement × demand``; we negate so a plain
    # ascending sort orders by priority descending. Negative
    # improvement items still enter the queue (they go last) so
    # the controller emits a capacity-aware plan for every (cell,
    # dest) with traffic — including the ones where baseline is
    # already E2E-optimal but might still need fallback sat
    # diversity at realize time.
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
        baseline_ranked = baseline.mapping.get(cell_id)
        if not baseline_ranked:
            continue
        baseline_pop = baseline_ranked[0]
        baseline_sat = cell_sat_cost.get((cell_id, baseline_pop))
        if baseline_sat is None:
            continue
        baseline_ground = ground_cost_fn(baseline_pop, dest)
        if baseline_ground is None:
            # Baseline PoP's ground RTT not cached yet. We can't
            # quantify "improvement" without it, so defer this pair
            # to the baseline mapping via no-override. Once feedback
            # populates GK for this PoP/dest the next plan refresh
            # will pick it up.
            continue
        baseline_cost = baseline_sat + baseline_ground
        best_alt_cost = ranked[0][1]
        improvement = baseline_cost - best_alt_cost
        queue.append((-(improvement * demand), cell_id, dest, demand))

    # Tuple sort: primary key is the negated priority. Subsequent
    # tuple elements (cell_id int, dest str) provide a stable,
    # deterministic tie-break across runs with identical inputs.
    queue.sort()

    # ── Greedy first-fit across the full PoP × per-PoP-sats grid ──
    sat_load: dict[int, float] = {}
    assignments: dict[tuple[int, str], str] = {}
    for _priority, cell_id, dest, demand in queue:
        ranked = rankings[(cell_id, dest)]
        chosen_pop: str | None = None
        chosen_sat: int | None = None

        # Track the global best fallback (lowest load ratio across
        # the entire cascade) in case nothing fits — used as the
        # least-bad overflow target.
        best_fallback_pop: str | None = None
        best_fallback_sat: int | None = None
        best_fallback_ratio = float("inf")

        for pop_code, _cost in ranked:
            egress_sats = cell_pop_egress(cell_id, pop_code)
            if not egress_sats:
                continue
            for egress_sat in egress_sats:
                current = sat_load.get(egress_sat, 0.0)
                if current + demand <= sat_feeder_cap_gbps:
                    chosen_pop = pop_code
                    chosen_sat = egress_sat
                    break
                ratio = current / max(sat_feeder_cap_gbps, 1e-9)
                if ratio < best_fallback_ratio:
                    best_fallback_ratio = ratio
                    best_fallback_pop = pop_code
                    best_fallback_sat = egress_sat
            if chosen_pop is not None:
                break

        if chosen_pop is None:
            # Genuine cascade exhaustion: every (pop, sat) we could
            # find is over capacity. Accept the overflow on the
            # least-loaded option so the contention spreads instead
            # of stacking on one sat. ``best_fallback_*`` is None
            # only when literally no PoP in the cascade had any
            # reachable egress sat — pathological enough to log.
            if best_fallback_pop is None or best_fallback_sat is None:
                _log.warning(
                    "_progressive_filling: cell %d / dest %r has no "
                    "reachable egress in the cascade; routing this "
                    "cell will fail at realize time.", cell_id, dest,
                )
                continue
            chosen_pop = best_fallback_pop
            chosen_sat = best_fallback_sat

        assignments[(cell_id, dest)] = chosen_pop
        sat_load[chosen_sat] = sat_load.get(chosen_sat, 0.0) + demand

    return assignments
