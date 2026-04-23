"""Dual-price / Lagrangian-relaxation controller for (cell, dest) → PoP.

Same **coarse-grained** planning surface as
:class:`~vantage.control.policy.greedy.ProgressiveController`: the
planner picks a PoP per ``(cell, dest)`` subject to **PoP-aggregate**
capacity (sum of GS ``max_capacity`` for the GSs attached to the PoP);
fine-grained egress-sat / GS selection stays in the data plane.

What changes is the *assignment* step. Instead of greedy first-fit on
the E2E-sorted ranking, we relax the PoP-capacity constraints into a
Lagrangian and iterate a subgradient update on the dual prices
``λ_p``:

    L(assign, λ) = Σ_i d_i · c(i, p_i) + Σ_p λ_p · (Σ_{i:p_i=p} d_i − cap_p)

For fixed ``λ`` the Lagrangian separates across items: each
``(cell, dest)`` picks the PoP that minimises ``c(i, p) + λ_p``. After
all items pick, we look at the per-PoP aggregate load — PoPs with
load > cap raise their ``λ``, PoPs with slack lower theirs (floored at
0). Repeat.

Two safety nets on top of the vanilla subgradient:

1. **Repair pass.** Subgradient methods do not guarantee a feasible
   primal iterate at termination. After the last iteration we scan for
   overloaded PoPs and greedily move the "cheapest to move" items
   (smallest cost-delta to the next feasible PoP in the item's E2E
   ranking) until either the PoP is within cap or no further moves
   are possible.
2. **Incumbent guard.** We also run
   :func:`~vantage.control.policy.greedy._progressive_filling` on the
   same inputs and keep whichever assignment has the lower weighted
   objective ``Σ d_i · c(i, p_i) + penalty · overflow``. This
   guarantees we never ship something strictly worse than the
   production greedy, even if the Lagrangian iteration gets stuck in
   a poor region (e.g. when the LP relaxation has a large integrality
   gap).

Output:
  * :class:`CellToPopTable` with the nearest-PoP baseline on ``mapping``
    and per-``(cell, dest)`` overrides on ``per_dest``.
  * Override cascade head = chosen PoP. Tail = the remaining PoPs in
    the item's E2E ranking, least-E2E-cost first. Same contract as
    :class:`~vantage.control.policy.greedy.ProgressiveController` so
    the data plane consumes the output identically.
"""

from __future__ import annotations

import logging
import time
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
from vantage.control.policy.greedy import _progressive_filling
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    NetworkSnapshot,
    RoutingPlane,
)
from vantage.world.ground import GroundKnowledge

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from vantage.domain import PoP

_log = logging.getLogger(__name__)


# Item carried through the solver: (cell_id, dest, demand_gbps,
# ranked_pops). ``ranked_pops`` is the list[(pop_code, e2e_cost_ms)]
# produced by :func:`rank_pops_by_e2e`, sorted ascending by cost.
_Item = tuple[int, str, float, list[tuple[str, float]]]


class DualPriceController:
    """Dual-price / Lagrangian-relaxation planner.

    Same constructor surface as
    :class:`~vantage.control.policy.greedy.ProgressiveController`. Extra
    hyper-parameters (``dp_iterations`` / ``dp_step_size`` /
    ``dp_incumbent_penalty``) control the Lagrangian iteration and the
    incumbent-comparison penalty term.
    """

    _DEFAULT_LAMBDA_DEV: float = 1.0
    _DEFAULT_STALE_PER_EPOCH_MS: float = 0.05
    # Subgradient defaults. 30 iterations is a pragmatic sweet spot: at
    # production scale one iteration is O(|items| × |pops_in_cascade|)
    # ~5–10 ms; 30 iterations stays under the 15 s refresh budget with
    # plenty of slack for the baseline/rankings/assemble steps. A
    # decreasing step size is classical for subgradient convergence,
    # but a constant step works well here because we have the repair
    # pass + incumbent guard as safety nets.
    _DEFAULT_ITERATIONS: int = 30
    _DEFAULT_STEP_SIZE: float = 2.0
    # Penalty (ms per Gbps of overflow) applied when scoring
    # infeasible candidate assignments against the incumbent greedy
    # result. A single Gbps of overflow costs the same as a 10 000 ms
    # RTT inflation — huge by design: we strongly prefer the feasible
    # incumbent over any infeasible dual-price iterate.
    _DEFAULT_INCUMBENT_PENALTY: float = 1.0e4

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
        dest_names: tuple[str, ...] = (),
        *,
        score_lambda_dev: float = _DEFAULT_LAMBDA_DEV,
        score_stale_per_epoch_ms: float = _DEFAULT_STALE_PER_EPOCH_MS,
        dp_iterations: int = _DEFAULT_ITERATIONS,
        dp_step_size: float = _DEFAULT_STEP_SIZE,
        dp_incumbent_penalty: float = _DEFAULT_INCUMBENT_PENALTY,
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()
        self._dest_names = dest_names
        self._warned_no_dests = False
        self._score_lambda_dev = float(score_lambda_dev)
        self._score_stale_per_epoch_ms = float(score_stale_per_epoch_ms)
        self._dp_iterations = int(dp_iterations)
        self._dp_step_size = float(dp_step_size)
        self._dp_incumbent_penalty = float(dp_incumbent_penalty)
        self._last_timing: Mapping[str, float] = MappingProxyType({})
        # Whether the most recent plan accepted the dual-price
        # assignment (True) or fell back to the incumbent greedy
        # result (False). Exposed for observability / tests.
        self._last_accepted_dualprice: bool = False

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> Mapping[str, float]:
        return self._last_timing

    @property
    def last_accepted_dualprice(self) -> bool:
        """``True`` iff the most recent plan shipped the dual-price
        assignment (vs. fell back to the greedy incumbent)."""
        return self._last_accepted_dualprice

    def resolve_dest_names(self) -> tuple[str, ...]:
        if self._dest_names:
            return self._dest_names
        derived = tuple(sorted({dest for _, dest in self._gk.all_entries()}))
        if not derived and not self._warned_no_dests:
            _log.warning(
                "DualPriceController.resolve_dest_names: no explicit "
                "dest_names and ground_knowledge has no entries; "
                "degrading to nearest-PoP baseline this epoch."
            )
            self._warned_no_dests = True
        return derived

    def _make_ground_cost(
        self,
        *,
        current_epoch: int,
        pops: Iterable[PoP] | None = None,
        dest_names: Iterable[str] | None = None,
    ) -> Callable[[str, str], float | None]:
        """Same GK-scored ground cost used by the greedy controller.

        Kept in-class (not imported from greedy) to avoid a circular
        import and to keep each controller's scoring knobs
        independent — a future experiment may want to vary λ_dev /
        staleness on just one of them without touching the other.
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

        t0 = perf()

        ground_cost = self._make_ground_cost(
            current_epoch=int(version),
            pops=pops,
            dest_names=dest_names,
        )

        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid, pops=pops,
            built_at=snapshot.time_s, version=version,
        )
        t_baseline = perf()

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

        pop_cap = compute_pop_capacity(snapshot)
        t_pop_cap = perf()

        demand_per_pair = demand_per_pair or {}

        # Materialise items once; both the dual-price solver and the
        # incumbent greedy consume the same shape.
        items = _build_items(rankings, cell_grid, demand_per_pair)

        # Dual-price primal+dual loop → repair → candidate assignment.
        dp_assignments = _dualprice_solve(
            items=items,
            pop_cap=pop_cap,
            iterations=self._dp_iterations,
            step_size=self._dp_step_size,
        )
        t_dualprice = perf()

        # Incumbent: greedy with the same inputs (re-runs demand
        # aggregation internally — the duplicated cost is negligible
        # vs. the refresh budget and keeps the comparison fair).
        greedy_assignments = _progressive_filling(
            rankings=rankings,
            baseline=baseline,
            cell_grid=cell_grid,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=ground_cost,
            pop_cap=pop_cap,
            demand_per_pair=demand_per_pair,
        )
        t_greedy = perf()

        # Incumbent guard: pick whichever scores lower on the common
        # objective (weighted E2E cost + penalty × overflow). Ties go
        # to the dual-price iterate so the solver is observable.
        dp_cost = _weighted_cost(
            dp_assignments, items, pop_cap,
            overflow_penalty=self._dp_incumbent_penalty,
        )
        greedy_cost = _weighted_cost(
            greedy_assignments, items, pop_cap,
            overflow_penalty=self._dp_incumbent_penalty,
        )
        if dp_cost <= greedy_cost:
            assignments = dp_assignments
            self._last_accepted_dualprice = True
        else:
            assignments = greedy_assignments
            self._last_accepted_dualprice = False
        t_compare = perf()

        # Assemble per-dest overrides. Head = chosen_pop; tail = the
        # remaining PoPs in the E2E ranking (ascending cost). Same
        # contract as ProgressiveController.
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

        self._last_timing = MappingProxyType({
            "baseline_ms": (t_baseline - t0) * 1000.0,
            "cell_sat_cost_ms": (t_cell_sat - t_baseline) * 1000.0,
            "rankings_ms": (t_rankings - t_cell_sat) * 1000.0,
            "pop_cap_ms": (t_pop_cap - t_rankings) * 1000.0,
            "dualprice_ms": (t_dualprice - t_pop_cap) * 1000.0,
            "greedy_incumbent_ms": (t_greedy - t_dualprice) * 1000.0,
            "incumbent_guard_ms": (t_compare - t_greedy) * 1000.0,
            "assemble_ms": (t_assemble - t_compare) * 1000.0,
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


# ---------------------------------------------------------------------------
# Solver helpers (module-level so tests can exercise them directly)
# ---------------------------------------------------------------------------


def _build_items(
    rankings: dict[tuple[int, str], list[tuple[str, float]]],
    cell_grid: CellGrid,
    demand_per_pair: dict[tuple[str, str], float],
) -> list[_Item]:
    """Aggregate per-endpoint demand into per-(cell, dest) items.

    Items with zero (or missing) demand are dropped: the dual-price
    solver only has leverage on items with non-zero load, and the
    remaining dests still get their baseline cascade from the plane
    assembly step. Items with empty rankings are also dropped — the
    controller cannot plan them at all.
    """
    cell_to_eps: dict[int, list[str]] = {}
    for ep_name, ep_cell in cell_grid.endpoint_to_cell.items():
        cell_to_eps.setdefault(ep_cell, []).append(ep_name)

    items: list[_Item] = []
    for (cell_id, dest), ranked in rankings.items():
        if not ranked:
            continue
        demand = sum(
            demand_per_pair.get((ep_name, dest), 0.0)
            for ep_name in cell_to_eps.get(cell_id, ())
        )
        if demand <= 0.0:
            continue
        items.append((cell_id, dest, demand, ranked))
    return items


def _dualprice_solve(
    items: list[_Item],
    pop_cap: Mapping[str, float],
    *,
    iterations: int,
    step_size: float,
) -> dict[tuple[int, str], str]:
    """Subgradient loop on ``λ_p`` followed by a feasibility repair.

    Returns ``{(cell_id, dest) → pop_code}`` for every item with at
    least one reachable, positive-capacity PoP in its ranking.
    """
    if not items:
        return {}

    # Only PoPs with positive capacity are biddable. Zero-cap PoPs
    # (no attached GS) are skipped outright — both in the argmin and
    # in the repair search.
    biddable_pops = {p for p, c in pop_cap.items() if c > 0.0}

    lambdas: dict[str, float] = {p: 0.0 for p in biddable_pops}

    last_assignments: dict[tuple[int, str], str] = {}
    last_pop_load: dict[str, float] = {p: 0.0 for p in biddable_pops}

    for _ in range(max(1, iterations)):
        assignments: dict[tuple[int, str], str] = {}
        pop_load: dict[str, float] = {p: 0.0 for p in biddable_pops}

        # Primal: each item minimises (cost + λ_p) over biddable PoPs
        # in its cascade. Items whose cascade is disjoint from the
        # biddable set are unassignable — they carry through as
        # missing keys and fall back to the baseline head at
        # assembly time.
        for cell_id, dest, demand, ranked in items:
            best_pop: str | None = None
            best_score = float("inf")
            for pop_code, e2e_cost in ranked:
                if pop_code not in biddable_pops:
                    continue
                score = e2e_cost + lambdas[pop_code]
                if score < best_score:
                    best_score = score
                    best_pop = pop_code
            if best_pop is None:
                continue
            assignments[(cell_id, dest)] = best_pop
            pop_load[best_pop] += demand

        # Dual: λ_p += α × (load_p − cap_p) / cap_p, floored at 0.
        # The cap-normalisation keeps the per-PoP update comparable
        # across shells of very different sizes (small PoPs don't
        # need tiny steps; big PoPs don't need huge steps).
        for p in biddable_pops:
            cap = pop_cap[p]
            if cap <= 0.0:
                continue
            delta = (pop_load[p] - cap) / cap
            lambdas[p] = max(0.0, lambdas[p] + step_size * delta)

        last_assignments = assignments
        last_pop_load = pop_load

    return _repair_feasibility(last_assignments, last_pop_load, items, pop_cap)


def _repair_feasibility(
    assignments: dict[tuple[int, str], str],
    pop_load: dict[str, float],
    items: list[_Item],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str]:
    """Move items off overloaded PoPs in cost-delta-ascending order.

    Greedy, bounded by O(|overloaded items| × |pops|) per overload
    resolution pass. Items that can't find any alternative with slack
    stay where they are — the controller accepts the residual
    overflow and lets the incumbent-guard decide whether to ship
    this assignment or fall back to greedy.
    """
    # Working copies so the caller's originals stay untouched if they
    # later want to compare pre- vs. post-repair.
    assignments = dict(assignments)
    pop_load = dict(pop_load)
    item_index: dict[tuple[int, str], _Item] = {
        (c, d): (c, d, demand, ranked) for c, d, demand, ranked in items
    }

    def _overloaded() -> list[tuple[str, float]]:
        out = []
        for p, load in pop_load.items():
            cap = pop_cap.get(p, 0.0)
            if cap > 0.0 and load > cap:
                out.append((p, load - cap))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # Cap the outer loop: each iteration either moves ≥1 item or
    # exits; bound by total item count to avoid any pathological
    # spin.
    for _ in range(len(items) + 1):
        overloaded = _overloaded()
        if not overloaded:
            break
        hot_pop, _ = overloaded[0]

        # Candidates on hot_pop, each with the smallest cost-delta
        # move into a PoP that currently has enough slack.
        best_move: tuple[float, tuple[int, str], str] | None = None
        for key, pop in list(assignments.items()):
            if pop != hot_pop:
                continue
            _, _, demand, ranked = item_index[key]
            # Find the current-PoP cost.
            cur_cost = None
            for p, c in ranked:
                if p == hot_pop:
                    cur_cost = c
                    break
            if cur_cost is None:
                continue
            for p, c in ranked:
                if p == hot_pop:
                    continue
                cap = pop_cap.get(p, 0.0)
                if cap <= 0.0:
                    continue
                slack = cap - pop_load.get(p, 0.0)
                if slack < demand:
                    continue
                delta = c - cur_cost
                if best_move is None or delta < best_move[0]:
                    best_move = (delta, key, p)
                break  # ranked ASC by cost — first feasible is best

        if best_move is None:
            # No feasible destination for any item on hot_pop. Accept
            # the residual overflow.
            break

        _, key, new_pop = best_move
        _, _, demand, _ = item_index[key]
        assignments[key] = new_pop
        pop_load[hot_pop] -= demand
        pop_load[new_pop] = pop_load.get(new_pop, 0.0) + demand

    return assignments


def _weighted_cost(
    assignments: dict[tuple[int, str], str],
    items: list[_Item],
    pop_cap: Mapping[str, float],
    *,
    overflow_penalty: float,
) -> float:
    """Objective used by the incumbent guard.

    ``Σ d_i × c(i, p_i)  +  overflow_penalty × Σ_p max(0, load_p − cap_p)``

    Items without an assignment contribute 0 on the cost side — they
    fall back to baseline at assembly time, and the baseline is the
    same for both candidates, so they cancel out of the comparison.
    The overflow term is what lets the guard prefer the (slightly
    more expensive) feasible greedy over an infeasible dual-price
    iterate in capacity-stressed regimes.
    """
    if not items:
        return 0.0
    item_index: dict[tuple[int, str], _Item] = {
        (c, d): (c, d, demand, ranked) for c, d, demand, ranked in items
    }
    cost = 0.0
    pop_load: dict[str, float] = {}
    for key, pop in assignments.items():
        item = item_index.get(key)
        if item is None:
            continue
        _, _, demand, ranked = item
        for p, c in ranked:
            if p == pop:
                cost += demand * c
                break
        pop_load[pop] = pop_load.get(pop, 0.0) + demand
    for p, load in pop_load.items():
        cap = pop_cap.get(p, 0.0)
        if cap > 0.0 and load > cap:
            cost += overflow_penalty * (load - cap)
    return cost
