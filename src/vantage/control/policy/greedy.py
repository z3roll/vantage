"""Progressive controller: improvement-first per-destination PoP assignment.

Coarse-grained planner. For each (cell, destination), picks the PoP
that minimises ``sat_cost + ground_cost`` while respecting
**PoP-aggregate** capacity (sum of attached GS.max_capacity).
Fine-grained egress-satellite / GS selection — and any future
ISL-capacity handling — stays in the data plane
(:class:`~vantage.forward.RoutingPlaneForward`), which still has
the last word via its multi-egress reroute + baseline fallback at
realize time.

Algorithm:

1. Rank all reachable PoPs per (cell, dest) by E2E cost
   (``sat_cost + ground_cost``).
2. For each (cell, dest), aggregate the current demand across every
   endpoint in the cell going to ``dest``, and compute the
   *improvement* delta:

       improvement = baseline_E2E_cost - best_alt_E2E_cost

   where ``baseline_pop`` is the cell's geographic nearest PoP.
3. Sort the surviving pairs by ``improvement × demand`` descending
   — biggest aggregate RTT saving first.
4. Greedy first-fit through each pair's ranking, respecting per-PoP
   aggregate capacity. On cascade exhaustion, overflow goes to the
   least-loaded-ratio PoP so contention spreads instead of piling
   on the single most-popular E2E-best PoP.

This is value-density greedy for a multi-knapsack / generalised
assignment problem (NP-hard). Empirically within ~10–20% of the
LP-relaxation optimum at our scale (~10⁴ cells × ~10¹ services × ~10²
PoPs); fast enough for the 15 s control-plane refresh budget.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_pop_egress_table,
    build_sat_path_table,
    compute_cell_sat_cost,
    compute_pop_capacity,
    rank_pops_by_e2e,
)
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    NetworkSnapshot,
    RoutingPlane,
)
from vantage.world.ground import GroundKnowledge

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from vantage.domain import PoP

_log = logging.getLogger(__name__)


class ProgressiveController:
    """E2E-aware PoP selection with Progressive Filling."""

    # Defaults for the GK-score composition used at rank time
    # (``mu + lambda_dev · dev + stale_per_epoch_ms · staleness``).
    #
    #   * ``lambda_dev = 1.0`` adds one full within-epoch stddev to
    #     the cost of a pair. Two PoPs with the same expected mean
    #     RTT but one with +4 ms stddev shows up as +4 ms worse in
    #     ranking — enough to push flows toward the steadier route
    #     when the alternative is equally fast on average.
    #   * ``stale_per_epoch_ms = 0.05`` adds 0.05 ms per epoch since
    #     the pair was last observed. At a 15-epoch plan cadence a
    #     just-stale pair costs +0.75 ms; a pair that's gone 600
    #     epochs (10 min at 1-epoch-per-second) without any realised
    #     flow costs +30 ms, which is large enough to deprioritise
    #     it relative to PoPs the planner keeps validating.
    #   * Prior-only entries (``last_epoch = -1``) are exempt — the
    #     planner treats priors as "not stale in the same sense as a
    #     rotted measurement"; see :meth:`GroundKnowledge.score`.
    _DEFAULT_LAMBDA_DEV: float = 1.0
    _DEFAULT_STALE_PER_EPOCH_MS: float = 0.05

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
        dest_names: tuple[str, ...] = (),
        *,
        score_lambda_dev: float = _DEFAULT_LAMBDA_DEV,
        score_stale_per_epoch_ms: float = _DEFAULT_STALE_PER_EPOCH_MS,
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()
        self._dest_names = dest_names
        # We deliberately do NOT warn at construction time: a freshly
        # built GK is expected to be empty until the feedback loop
        # populates it across epochs. The warning lives inside
        # `resolve_dest_names` and fires once per controller if a
        # `compute_routing_plane` call actually finds no destinations.
        self._warned_no_dests = False
        self._score_lambda_dev = float(score_lambda_dev)
        self._score_stale_per_epoch_ms = float(score_stale_per_epoch_ms)
        # Per-step wall-clock timings (ms) for the most recent
        # ``compute_routing_plane`` invocation. ``run.py`` reads this
        # after a refresh to export per-step breakdowns to the
        # dashboard. Empty until the first call.
        self._last_timing: Mapping[str, float] = MappingProxyType({})

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> Mapping[str, float]:
        """Step timings (ms) from the most recent plan build."""
        return self._last_timing

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

    def _make_ground_cost(
        self,
        *,
        current_epoch: int,
        pops: "Iterable[PoP] | None" = None,
        dest_names: "Iterable[str] | None" = None,
    ) -> "Callable[[str, str], float | None]":
        """Build the ``(pop, dest) → cost_ms`` function used for this plan.

        Bound to ``current_epoch`` so the staleness penalty reflects
        how long ago each pair was last observed, measured against
        the epoch we're planning *for* (typically the ``version``
        argument of :meth:`compute_routing_plane`, which is wired to
        the current epoch in ``run.py``).

        Behaviour per pair:

        1. If :meth:`GroundKnowledge.score` returns a value — i.e. a
           learned stat exists for the pair — that score is used.
           It composes ``mu_ms + λ·dev_ms + stale_per_epoch·Δepoch``,
           so noisy/stale pairs cost more than calm/fresh pairs with
           the same mean.
        2. Otherwise fall back to the deterministic estimator
           (``estimator.estimate(pop, dest) * 2`` — one-way → RTT).
           This keeps PoPs that have never been observed rankable
           without inflating their score artificially.
        3. If neither is available return ``None`` so
           :func:`rank_pops_by_e2e` drops the pair from the ranking.

        When ``pops`` **and** ``dest_names`` are both supplied the
        table is **precomputed once per refresh** and the returned
        lookup is a plain ``dict.get`` — this is the hot path
        invoked from :meth:`compute_routing_plane`. Otherwise
        (external callers / tests that probe ad-hoc pairs) the
        returned callable evaluates on demand so unknown pairs
        still resolve correctly. Semantics are identical in both
        modes; only the per-call cost differs.
        """
        gk = self._gk
        estimator = gk.estimator
        lambda_dev = self._score_lambda_dev
        stale_per_epoch_ms = self._score_stale_per_epoch_ms

        def compute(pop_code: str, dest: str) -> float | None:
            scored = gk.score(
                pop_code, dest,
                current_epoch=current_epoch,
                lambda_dev=lambda_dev,
                stale_per_epoch_ms=stale_per_epoch_ms,
            )
            if scored is not None:
                return scored
            if estimator is None:
                return None
            try:
                return estimator.estimate(pop_code, dest) * 2
            except KeyError:
                return None

        if pops is None or dest_names is None:
            return compute

        # Precompute hot path: rank_pops_by_e2e + _progressive_filling
        # together evaluate ground_cost ~ O(n_cells × n_dests × n_pops)
        # times per refresh (~10⁶ calls at production scale). The
        # score is cell-independent, so one (pop, dest) entry covers
        # every cell — drop the call count from 10⁶ to n_pops ×
        # n_dests (~48 × 14 ≈ 700) and keep the per-lookup work at a
        # single ``dict.get``.
        table: dict[tuple[str, str], float | None] = {}
        dest_tuple = tuple(dest_names)
        for pop in pops:
            pop_code = pop.code
            for dest in dest_tuple:
                table[(pop_code, dest)] = compute(pop_code, dest)

        def lookup(pop_code: str, dest: str) -> float | None:
            return table.get((pop_code, dest))

        return lookup

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        demand_per_pair: dict[tuple[str, str], float] | None = None,
        version: int = 0,
    ) -> RoutingPlane:
        pops = snapshot.infra.pops
        dest_names = self.resolve_dest_names()
        perf = time.perf_counter

        # Pre-refactor there was a 0-th ``_prime_ground_knowledge``
        # step here that filled GK with estimator-derived RTTs so the
        # cache-only ground cost could rank every PoP from epoch 0.
        # The current ground-cost closure falls back to the estimator
        # directly on a miss, so the bootstrap is redundant.
        t0 = perf()
        t_prime = t0  # retained so the timing report below keeps a
                       # "prime_gk_ms" field valued at 0.

        # Build a per-refresh scoring table bound to ``version`` (=
        # current epoch at plan build time) so :func:`rank_pops_by_e2e`
        # and :func:`_progressive_filling` share the same cost
        # accounting — same mean + noise + staleness penalty on both
        # the baseline side and the alternates side of the
        # improvement delta. Passing ``pops`` + ``dest_names`` forces
        # the precomputed-table path: ~10⁶ per-call evaluations
        # collapse to ~700 one-time ones.
        ground_cost = self._make_ground_cost(
            current_epoch=int(version),
            pops=pops,
            dest_names=dest_names,
        )

        # 1. Baseline: nearest PoP per cell. Used as both the data-plane
        #    fallback for cells without overrides AND the reference
        #    cost against which "improvement" is measured.
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid, pops=pops,
            built_at=snapshot.time_s, version=version,
        )
        t_baseline = perf()

        # 2. Rank all reachable PoPs per (cell, dest) by E2E cost.
        cell_sat_cost = compute_cell_sat_cost(snapshot, cell_grid)
        t_cell_sat = perf()
        rankings = rank_pops_by_e2e(
            cell_grid=cell_grid,
            pops=pops,
            baseline=baseline,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=ground_cost,
            dest_names=dest_names,
        )
        t_rankings = perf()

        # 3. Improvement-first greedy assignment against PoP-aggregate
        # capacity. PG is coarse-grained: it contends at the PoP
        # level only. Per-sat-feeder and per-GS-feeder contention is
        # enforced at realize time by RoutingPlaneForward, which
        # additionally has multi-egress reroute + baseline fallback
        # if the planner's choice is suboptimal under the actual
        # fine-grained load.
        pop_cap = compute_pop_capacity(snapshot)
        t_pop_cap = perf()
        assignments = _progressive_filling(
            rankings=rankings,
            baseline=baseline,
            cell_grid=cell_grid,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=ground_cost,
            pop_cap=pop_cap,
            demand_per_pair=demand_per_pair or {},
        )
        t_progressive = perf()

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
        t_assemble = perf()
        sat_paths = build_sat_path_table(snapshot, version=version)
        t_sat_paths = perf()
        pop_egress = build_pop_egress_table(snapshot, version=version)
        t_pop_egress = perf()

        # Cascade-assembly cost (between progressive-fill and
        # sat_paths) is folded into ``progressive_fill_ms`` since it
        # is part of finalising PG's PoP assignment, not part of the
        # sat-path table build.
        self._last_timing = MappingProxyType({
            "prime_gk_ms": (t_prime - t0) * 1000.0,
            "baseline_ms": (t_baseline - t_prime) * 1000.0,
            "cell_sat_cost_ms": (t_cell_sat - t_baseline) * 1000.0,
            "rankings_ms": (t_rankings - t_cell_sat) * 1000.0,
            "pop_cap_ms": (t_pop_cap - t_rankings) * 1000.0,
            "progressive_fill_ms": (t_assemble - t_pop_cap) * 1000.0,
            "sat_paths_ms": (t_sat_paths - t_assemble) * 1000.0,
            "pop_egress_ms": (t_pop_egress - t_sat_paths) * 1000.0,
        })

        return RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_paths=sat_paths,
            pop_egress=pop_egress,
            version=version,
            built_at=snapshot.time_s,
        )


def _progressive_filling(
    rankings: dict[tuple[int, str], list[tuple[str, float]]],
    baseline: CellToPopTable,
    cell_grid: CellGrid,
    cell_sat_cost: dict[tuple[int, str], float],
    ground_cost_fn: Callable[[str, str], float | None],
    pop_cap: Mapping[str, float],
    demand_per_pair: dict[tuple[str, str], float],
) -> dict[tuple[int, str], str]:
    """Capacity-aware greedy assignment at PoP granularity.

    PG plans coarsely: it picks a PoP per (cell, dest) against a
    **PoP-aggregate** capacity budget (sum of GS.max_capacity on
    GSs attached to the PoP). Fine-grained egress-sat / GS
    selection, and any future ISL-capacity handling, remain the
    data plane's responsibility.

    For each (cell, dest) with positive demand:

    * Aggregate ``demand`` across every endpoint hosted by this cell
      that targets ``dest``.
    * Compute ``improvement`` = ``baseline_cost - best_alt_cost``
      using the cell's geographic-nearest PoP as baseline.
    * Items are processed in descending order of
      ``improvement × demand`` — so positive-improvement cells
      claim scarce PoP capacity first; negative-improvement items
      still enter the queue so the controller emits a cascade for
      every (cell, dest) with traffic.

    For each item, walk the full E2E-sorted ranking and pick the
    first PoP whose ``pop_load[pop] + demand ≤ pop_cap[pop]``. On
    cascade exhaustion, overflow goes to the PoP with the lowest
    current load ratio so contention spreads rather than always
    stacking on the single most-popular PoP. The data plane's
    defensive fallback handles any residual overrun at realize
    time.

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

    queue.sort()

    # ── Greedy first-fit at PoP granularity ──
    pop_load: dict[str, float] = {}
    assignments: dict[tuple[int, str], str] = {}
    for _priority, cell_id, dest, demand in queue:
        ranked = rankings[(cell_id, dest)]
        chosen_pop: str | None = None
        best_fallback_pop: str | None = None
        best_fallback_ratio = float("inf")

        for pop_code, _cost in ranked:
            cap = pop_cap.get(pop_code, 0.0)
            if cap <= 0.0:
                continue
            current = pop_load.get(pop_code, 0.0)
            if current + demand <= cap:
                chosen_pop = pop_code
                break
            ratio = current / cap
            if ratio < best_fallback_ratio:
                best_fallback_ratio = ratio
                best_fallback_pop = pop_code

        if chosen_pop is None:
            if best_fallback_pop is None:
                _log.warning(
                    "_progressive_filling: cell %d / dest %r has no "
                    "PoP with positive capacity in the cascade; "
                    "routing this cell will fall back to baseline.",
                    cell_id, dest,
                )
                continue
            chosen_pop = best_fallback_pop

        assignments[(cell_id, dest)] = chosen_pop
        pop_load[chosen_pop] = pop_load.get(chosen_pop, 0.0) + demand

    return assignments
