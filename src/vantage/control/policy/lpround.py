"""LP-relaxation controller for (cell, dest) → PoP assignment.

Formulation::

    min  Σ_{i,p} d_i · c(i, p) · x_{i,p}
    s.t. Σ_p x_{i,p} = 1                      ∀ i      (each item goes somewhere)
         Σ_i d_i · x_{i,p} ≤ cap_p            ∀ p      (PoP aggregate capacity)
         x_{i,p} ∈ [0, 1]                              (LP relaxation of GAP)

``OPT_LP`` (the optimum of this LP) is a **provable lower bound** on
the optimum of the integer problem (minimisation + variables
relaxed to a superset → objective can only go down), so the solver
output doubles as a global-optimality reference.

For deployment we need an integer assignment. After the LP solve,
each item picks the PoP that carries the highest fractional weight
(``argmax_p x_{i,p}``). At our scale the LP is typically *almost*
integer already — with continuous demand and ``d_i ≪ cap_p`` for
most items, the integrality gap is small — so the rounded
assignment is very close to ``OPT_LP``. A defensive
overflow-repair step catches the rare case where rounding pushes a
tight PoP slightly over its cap.

Trade-off vs greedy / dual-price:

* Strictly optimal on the LP relaxation (no step-size tuning, no
  subgradient convergence risk).
* One LP solve per refresh (seconds-to-tens-of-seconds with
  HiGHS at production scale).
* Upper-bounded suboptimality on the integer problem: at most the
  integrality gap of the LP.

No candidate pruning, no early termination. Every (item, pop)
pair from :func:`rank_pops_by_e2e` becomes a variable, and the
solver runs to its full LP optimum. This is the full-precision
comparison mode the research dashboard wants.
"""

from __future__ import annotations

import logging
import time
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

from vantage.control.costing import build_ground_cost_lookup
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_pop_egress_table,
    build_sat_path_table,
    compute_cell_sat_cost,
    compute_pop_capacity,
    rank_pops_by_e2e,
)
from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import CellToPopTable, RoutingPlane
from vantage.model import CellGrid, NetworkSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from vantage.model import PoP


_log = logging.getLogger(__name__)


# Item carried into the solver: same shape as the other controllers'
# private `_Item` — (cell_id, dest, demand_gbps, ranked_pops).
_Item = tuple[int, str, float, list[tuple[str, float]]]


class LPRoundingController:
    """LP-relaxation + argmax-rounding planner.

    Public interface matches
    :class:`~vantage.control.policy.greedy.GreedyController`:
    :meth:`compute_routing_plane` returns a :class:`RoutingPlane`
    whose ``cell_to_pop`` mapping carries nearest-PoP as the default
    and the LP-rounded assignment as per-destination overrides.

    The most recent LP's optimum objective is exposed via
    :attr:`last_lp_opt` so run.py can emit it as a global-optimality
    lower bound on the dashboard.
    """

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
        self._warned_no_dests = False
        self._score_lambda_dev = float(score_lambda_dev)
        self._score_stale_per_epoch_ms = float(score_stale_per_epoch_ms)
        self._last_timing: Mapping[str, float] = MappingProxyType({})
        self._last_lp_opt: float | None = None

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> Mapping[str, float]:
        return self._last_timing

    @property
    def last_lp_opt(self) -> float | None:
        """LP objective value of the most recent plan, or ``None`` if
        the LP solver failed. Useful as a global-optimality lower
        bound: no integer controller can do strictly better than
        ``last_lp_opt`` on this epoch's demand."""
        return self._last_lp_opt

    def resolve_dest_names(self) -> tuple[str, ...]:
        if self._dest_names:
            return self._dest_names
        derived = tuple(sorted({dest for _, dest in self._gk.all_entries()}))
        if not derived and not self._warned_no_dests:
            _log.warning(
                "LPRoundingController.resolve_dest_names: no explicit "
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
        return build_ground_cost_lookup(
            self._gk,
            current_epoch=current_epoch,
            lambda_dev=self._score_lambda_dev,
            stale_per_epoch_ms=self._score_stale_per_epoch_ms,
            pops=pops,
            dest_names=dest_names,
        )

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

        ground_cost = build_ground_cost_lookup(
            self._gk,
            current_epoch=int(version),
            lambda_dev=self._score_lambda_dev,
            stale_per_epoch_ms=self._score_stale_per_epoch_ms,
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
        from vantage.control.evaluation import build_ranked_demand_items

        items = build_ranked_demand_items(rankings, cell_grid, demand_per_pair)

        # Solve LP + round. On LP failure the controller falls back
        # to an empty assignment (every item defaults to baseline at
        # assembly time). We log the failure but don't raise; refresh
        # loops shouldn't die because the solver hit a numerical edge.
        assignments, lp_opt = _solve_lp_and_round(
            items=items,
            pop_cap=pop_cap,
        )
        self._last_lp_opt = lp_opt
        t_lp = perf()

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
            "lp_solve_ms": (t_lp - t_pop_cap) * 1000.0,
            "assemble_ms": (t_assemble - t_lp) * 1000.0,
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
# Module-level helpers (shared with MILPController)
# ---------------------------------------------------------------------------


def _build_items(
    rankings: dict[tuple[int, str], list[tuple[str, float]]],
    cell_grid: CellGrid,
    demand_per_pair: dict[tuple[str, str], float],
) -> list[_Item]:
    """Compatibility wrapper around the public evaluation item builder."""
    from vantage.control.evaluation import build_ranked_demand_items

    return build_ranked_demand_items(rankings, cell_grid, demand_per_pair)


def _build_lp_arrays(
    items: list[_Item],
    pop_cap: Mapping[str, float],
) -> tuple[
    np.ndarray,            # c:           objective coefficients, shape (n_vars,)
    tuple[np.ndarray, np.ndarray, np.ndarray, int, int],  # a_ub sparse (rows, cols, vals) + shape
    np.ndarray,            # b_ub:        upper bounds per pop, shape (n_pops,)
    tuple[np.ndarray, np.ndarray, np.ndarray, int, int],  # a_eq sparse
    np.ndarray,            # b_eq:        1s per item
    list[tuple[int, str]], # var_index:   var_id → (item_index, pop_code)
    list[tuple[int, str, float]],  # item meta
]:
    """Materialise the LP in (sparse) array form.

    Every ``(item, pop)`` pair where the PoP has positive capacity
    and appears in the item's E2E ranking becomes a variable — no
    top-K truncation, full precision.

    **Items with no biddable candidate** (every ranked PoP has
    zero capacity) are *excluded from the model entirely*: they
    contribute no variables and no row in the assignment-equality
    matrix. Caller code handles them at assembly time by falling
    back to the baseline ``CellToPopTable``. Earlier versions of
    this helper kept a dummy equality row for such items, which
    made the whole LP infeasible (``Σ (0 · x) = 1``).

    We emit row/col/val triples for both the inequality (per-PoP
    capacity) and equality (per-item assignment) matrices so callers
    can pass them to :func:`scipy.sparse.csr_matrix` without building
    a dense n-vars × n-cons array — that array is ~50 M entries at
    production scale, prohibitively large.
    """
    biddable_pops = {p for p, cap in pop_cap.items() if cap > 0.0}
    pop_idx: dict[str, int] = {
        p: i for i, p in enumerate(sorted(biddable_pops))
    }
    n_pops = len(pop_idx)

    # var_index[k] = (item_i, pop_code). ``item_i`` stays as the
    # ORIGINAL index into ``items`` so decoders downstream can look
    # up ``items[item_i]`` directly. The assignment-equality rows,
    # however, use a *compact* solvable-item index — infeasibilising
    # the model with dummy rows for un-solvable items is precisely
    # what Bug 1 was about.
    var_index: list[tuple[int, str]] = []
    obj: list[float] = []
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []

    # Meta covers only solvable items. Un-solvable items are
    # dropped entirely — the caller discovers them by diffing
    # ``items`` against the returned assignment keys.
    item_meta: list[tuple[int, str, float]] = []
    n_solvable = 0

    for i, (cell_id, dest, demand, ranked) in enumerate(items):
        cand: list[tuple[str, float]] = [
            (pop_code, c) for pop_code, c in ranked
            if pop_code in biddable_pops
        ]
        if not cand:
            # No biddable candidate — omit from the model.
            continue
        item_meta.append((cell_id, dest, demand))
        eq_row = n_solvable
        n_solvable += 1
        for pop_code, c in cand:
            var_id = len(var_index)
            var_index.append((i, pop_code))
            obj.append(demand * c)
            # Assignment: Σ_p x_{i,p} = 1 for each solvable item.
            eq_rows.append(eq_row)
            eq_cols.append(var_id)
            eq_vals.append(1.0)
            # Capacity: Σ_i d_i · x_{i,p} ≤ cap_p for each pop.
            ub_rows.append(pop_idx[pop_code])
            ub_cols.append(var_id)
            ub_vals.append(demand)

    n_vars = len(var_index)

    c = np.asarray(obj, dtype=np.float64)
    b_ub = np.fromiter(
        (pop_cap[p] for p in sorted(biddable_pops)),
        dtype=np.float64, count=n_pops,
    )
    # NB: sized by the *solvable* count, not the raw item count.
    b_eq = np.ones(n_solvable, dtype=np.float64)

    a_ub = (
        np.asarray(ub_rows, dtype=np.int64),
        np.asarray(ub_cols, dtype=np.int64),
        np.asarray(ub_vals, dtype=np.float64),
        n_pops, n_vars,
    )
    a_eq = (
        np.asarray(eq_rows, dtype=np.int64),
        np.asarray(eq_cols, dtype=np.int64),
        np.asarray(eq_vals, dtype=np.float64),
        n_solvable, n_vars,
    )

    return c, a_ub, b_ub, a_eq, b_eq, var_index, item_meta


def _as_sparse(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
               n_rows: int, n_cols: int):
    """Wrap (rows, cols, vals) into a scipy CSR matrix without
    importing ``scipy.sparse`` at module import time — tests that
    mock the solver don't need the transitive dependency."""
    from scipy.sparse import csr_matrix  # local import
    return csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


def _solve_lp_and_round(
    items: list[_Item],
    pop_cap: Mapping[str, float],
) -> tuple[dict[tuple[int, str], str], float | None]:
    """Solve the LP, round to an integer assignment, repair overflow.

    Returns ``({(cell, dest): pop}, lp_opt_value)``. ``lp_opt_value``
    is ``None`` when the LP solver fails (infeasibility,
    ill-conditioning); in that case the returned assignment dict is
    empty and callers should fall back to their baseline mapping.
    """
    if not items:
        return {}, 0.0

    c_arr, a_ub, b_ub, a_eq, b_eq, var_index, _item_meta = _build_lp_arrays(
        items, pop_cap,
    )
    if c_arr.size == 0:
        return {}, 0.0

    a_ub_sparse = _as_sparse(*a_ub)
    a_eq_sparse = _as_sparse(*a_eq)

    res = linprog(
        c_arr,
        A_ub=a_ub_sparse, b_ub=b_ub,   # noqa: N803  (scipy API kwarg name)
        A_eq=a_eq_sparse, b_eq=b_eq,   # noqa: N803
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not res.success:
        _log.warning("LP solver failed: %s", res.message)
        return {}, None

    x = res.x  # fractional optimum, shape (n_vars,)

    # Argmax-rounding: for each item, pick the candidate PoP that
    # the LP placed the most weight on. Ties are broken by the
    # cascade order (lower E2E cost first, since var_index preserves
    # the ranked cascade's order per item).
    best_weight: dict[int, float] = {}
    chosen: dict[int, str] = {}
    for var_id, (item_i, pop_code) in enumerate(var_index):
        w = x[var_id]
        if w > best_weight.get(item_i, -1.0):
            best_weight[item_i] = w
            chosen[item_i] = pop_code

    assignments: dict[tuple[int, str], str] = {}
    for item_i, pop_code in chosen.items():
        cell_id, dest, _demand, _ranked = items[item_i]
        assignments[(cell_id, dest)] = pop_code

    # Defensive overflow repair: rounding sometimes pushes a tight
    # PoP a sliver over cap even when the LP was feasible. Move the
    # cheapest-delta item to its next-best candidate with slack.
    assignments = _repair_overflow(assignments, items, pop_cap)

    return assignments, float(res.fun)


_FEASIBILITY_TOL: float = 1e-6


def _repair_overflow(
    assignments: dict[tuple[int, str], str],
    items: list[_Item],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str]:
    """Return an assignment guaranteed to respect ``pop_cap``.

    Strategy:

    1. If the input assignment is already feasible, return it as-is.
    2. **Narrow sub-MILP.** Freeze every item currently on a
       non-overloaded PoP; re-optimise the remaining ("movable")
       items against the residual capacity with a full integer
       program. This handles chain-move / swap cases the previous
       single-step greedy heuristic used to miss.
    3. **Full-problem MILP fallback.** If the narrow sub-problem is
       infeasible (e.g. the movable items cannot fit into the
       residual cap without the frozen items also moving), re-solve
       the full integer program over all ``items``.
    4. **Raise** if both strategies fail — the instance has no
       feasible integer assignment given ``pop_cap``. This is never a
       silent return of an overloaded solution.
    """
    assignments = dict(assignments)
    item_by_key: dict[tuple[int, str], tuple[float, list[tuple[str, float]]]] = {
        (c, d): (demand, ranked) for c, d, demand, ranked in items
    }

    def _compute_load(assign: dict[tuple[int, str], str]) -> dict[str, float]:
        ld: dict[str, float] = {}
        for key, pop in assign.items():
            demand, _ = item_by_key[key]
            ld[pop] = ld.get(pop, 0.0) + demand
        return ld

    def _is_feasible(ld: dict[str, float]) -> bool:
        return all(
            used <= pop_cap.get(p, 0.0) + _FEASIBILITY_TOL
            for p, used in ld.items()
        )

    load = _compute_load(assignments)
    if _is_feasible(load):
        return assignments

    # ── Strategy 1: narrow sub-MILP on items currently on overloaded PoPs.
    overloaded = {
        p for p, used in load.items()
        if used > pop_cap.get(p, 0.0) + _FEASIBILITY_TOL
    }
    movable_keys = [k for k, pop in assignments.items() if pop in overloaded]
    if movable_keys:
        movable_set = set(movable_keys)
        frozen_load: dict[str, float] = {}
        for key, pop in assignments.items():
            if key in movable_set:
                continue
            demand, _ = item_by_key[key]
            frozen_load[pop] = frozen_load.get(pop, 0.0) + demand
        remaining_cap = {
            p: max(0.0, pop_cap.get(p, 0.0) - frozen_load.get(p, 0.0))
            for p in pop_cap
        }
        sub_items: list[_Item] = []
        for (cell_id, dest) in movable_keys:
            demand, ranked = item_by_key[(cell_id, dest)]
            sub_items.append((cell_id, dest, demand, ranked))
        sub_result = _solve_sub_milp(sub_items, remaining_cap)
        if sub_result is not None:
            candidate = dict(assignments)
            candidate.update(sub_result)
            if _is_feasible(_compute_load(candidate)):
                return candidate

    # ── Strategy 2: full-problem MILP over every item.
    full_result = _solve_sub_milp(list(items), pop_cap)
    if full_result is not None and _is_feasible(_compute_load(full_result)):
        return full_result

    raise RuntimeError(
        "_repair_overflow: could not produce a capacity-feasible integer "
        "assignment (narrow and full sub-MILP both failed or stayed "
        "overloaded). Given pop_cap, no feasible integer solution exists."
    )


def _weighted_cost(
    assignments: dict[tuple[int, str], str],
    items: list[_Item],
    pop_cap: Mapping[str, float],
    *,
    overflow_penalty: float,
) -> float:
    """Compatibility wrapper around the public assignment objective."""
    from vantage.control.evaluation import compute_assignment_objective

    return compute_assignment_objective(
        assignments,
        items,
        pop_cap,
        overflow_penalty=overflow_penalty,
    )


def _solve_sub_milp(
    items: list[_Item],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str] | None:
    """Solve a (possibly small) GAP instance to integer optimality.

    Shared helper used by :func:`_repair_overflow`. Returns the
    integer assignment or ``None`` if the solver reports
    infeasibility / failure. Items with no biddable candidate under
    ``pop_cap`` are omitted from the model and therefore won't
    appear as keys in the returned dict — the caller must handle
    them (e.g. by leaving the original assignment in place or
    falling back to baseline at assembly time).
    """
    # Local import to avoid the scipy.sparse / scipy.optimize import
    # cost on modules that only use the LP path.
    from scipy.optimize import Bounds, LinearConstraint, milp

    if not items:
        return {}

    c_arr, a_ub, b_ub, a_eq, b_eq, var_index, _item_meta = _build_lp_arrays(
        items, pop_cap,
    )
    if c_arr.size == 0:
        # No biddable candidates anywhere — genuinely infeasible for
        # the caller's purposes.
        return None

    a_ub_sparse = _as_sparse(*a_ub)
    a_eq_sparse = _as_sparse(*a_eq)
    integrality = np.ones(c_arr.size, dtype=np.int32)
    bounds = Bounds(lb=0.0, ub=1.0)
    constraints = [
        LinearConstraint(a_ub_sparse, ub=b_ub),
        LinearConstraint(a_eq_sparse, lb=b_eq, ub=b_eq),
    ]
    res = milp(
        c_arr, constraints=constraints,
        integrality=integrality, bounds=bounds,
        options={"disp": False},
    )
    if not res.success or res.x is None:
        return None

    best: dict[int, tuple[float, str]] = {}
    for var_id, (item_i, pop_code) in enumerate(var_index):
        w = float(res.x[var_id])
        prev = best.get(item_i)
        if prev is None or w > prev[0]:
            best[item_i] = (w, pop_code)

    out: dict[tuple[int, str], str] = {}
    for item_i, (_w, pop_code) in best.items():
        cell_id, dest, _d, _r = items[item_i]
        out[(cell_id, dest)] = pop_code
    return out
