"""Generalized assignment solvers shared by control policies."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

import numpy as np
from scipy.optimize import linprog

from vantage.control.policy.common.planning import RankedDemandItem

__all__ = [
    "as_sparse",
    "build_gap_arrays",
    "repair_overflow",
    "solve_lp_and_round",
    "solve_milp",
    "solve_sub_milp",
]

_log = logging.getLogger(__name__)
_FEASIBILITY_TOL: float = 1e-6

SparseTriples = tuple[np.ndarray, np.ndarray, np.ndarray, int, int]


def build_gap_arrays(
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
) -> tuple[
    np.ndarray,
    SparseTriples,
    np.ndarray,
    SparseTriples,
    np.ndarray,
    list[tuple[int, str]],
    list[tuple[int, str, float]],
]:
    """Materialise the GAP LP/MILP matrices in sparse-triplet form.

    Every ``(item, pop)`` pair where the PoP has positive capacity
    and appears in the item's E2E ranking becomes a variable. Items
    with no biddable candidate are omitted from the model entirely;
    caller code handles those by falling back to the baseline routing
    plane at assembly time.
    """
    biddable_pops = {pop for pop, cap in pop_cap.items() if cap > 0.0}
    pop_idx: dict[str, int] = {
        pop: idx for idx, pop in enumerate(sorted(biddable_pops))
    }
    n_pops = len(pop_idx)

    var_index: list[tuple[int, str]] = []
    obj: list[float] = []
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    item_meta: list[tuple[int, str, float]] = []
    n_solvable = 0

    for item_idx, (cell_id, dest, demand, ranked) in enumerate(items):
        candidates = [
            (pop_code, cost)
            for pop_code, cost in ranked
            if pop_code in biddable_pops
        ]
        if not candidates:
            continue
        item_meta.append((cell_id, dest, demand))
        eq_row = n_solvable
        n_solvable += 1
        for pop_code, cost in candidates:
            var_id = len(var_index)
            var_index.append((item_idx, pop_code))
            obj.append(demand * cost)
            eq_rows.append(eq_row)
            eq_cols.append(var_id)
            eq_vals.append(1.0)
            ub_rows.append(pop_idx[pop_code])
            ub_cols.append(var_id)
            ub_vals.append(demand)

    n_vars = len(var_index)
    c = np.asarray(obj, dtype=np.float64)
    b_ub = np.fromiter(
        (pop_cap[pop] for pop in sorted(biddable_pops)),
        dtype=np.float64,
        count=n_pops,
    )
    b_eq = np.ones(n_solvable, dtype=np.float64)
    a_ub = (
        np.asarray(ub_rows, dtype=np.int64),
        np.asarray(ub_cols, dtype=np.int64),
        np.asarray(ub_vals, dtype=np.float64),
        n_pops,
        n_vars,
    )
    a_eq = (
        np.asarray(eq_rows, dtype=np.int64),
        np.asarray(eq_cols, dtype=np.int64),
        np.asarray(eq_vals, dtype=np.float64),
        n_solvable,
        n_vars,
    )
    return c, a_ub, b_ub, a_eq, b_eq, var_index, item_meta


def as_sparse(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_rows: int, n_cols: int):
    """Wrap sparse triplets into a SciPy CSR matrix."""
    from scipy.sparse import csr_matrix

    return csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


def _decode_assignment(
    x: np.ndarray,
    var_index: list[tuple[int, str]],
    items: list[RankedDemandItem],
) -> dict[tuple[int, str], str]:
    best: dict[int, tuple[float, str]] = {}
    for var_id, (item_idx, pop_code) in enumerate(var_index):
        weight = float(x[var_id])
        prev = best.get(item_idx)
        if prev is None or weight > prev[0]:
            best[item_idx] = (weight, pop_code)

    out: dict[tuple[int, str], str] = {}
    for item_idx, (_weight, pop_code) in best.items():
        cell_id, dest, _demand, _ranked = items[item_idx]
        out[(cell_id, dest)] = pop_code
    return out


def solve_lp_and_round(
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
) -> tuple[dict[tuple[int, str], str], float | None]:
    """Solve the LP relaxation, round to an integer assignment, and repair."""
    if not items:
        return {}, 0.0

    c_arr, a_ub, b_ub, a_eq, b_eq, var_index, _item_meta = build_gap_arrays(
        items,
        pop_cap,
    )
    if c_arr.size == 0:
        return {}, 0.0

    res = linprog(
        c_arr,
        A_ub=as_sparse(*a_ub),
        b_ub=b_ub,
        A_eq=as_sparse(*a_eq),
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not res.success:
        _log.warning("LP solver failed: %s", res.message)
        return {}, None

    assignments = _decode_assignment(res.x, var_index, items)
    assignments = repair_overflow(assignments, items, pop_cap)
    return assignments, float(res.fun)


def _assignment_load(
    assignments: Mapping[tuple[int, str], str],
    items: list[RankedDemandItem],
) -> dict[str, float]:
    item_by_key = {
        (cell_id, dest): demand
        for cell_id, dest, demand, _ranked in items
    }
    load: dict[str, float] = {}
    for key, pop_code in assignments.items():
        demand = item_by_key[key]
        load[pop_code] = load.get(pop_code, 0.0) + demand
    return load


def _is_feasible(load: Mapping[str, float], pop_cap: Mapping[str, float]) -> bool:
    return all(
        used <= pop_cap.get(pop_code, 0.0) + _FEASIBILITY_TOL
        for pop_code, used in load.items()
    )


def repair_overflow(
    assignments: Mapping[tuple[int, str], str],
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str]:
    """Return a capacity-feasible assignment using greedy local repair.

    LP argmax rounding usually misses capacity by a tiny amount, but solving a
    sub-MILP just to repair that overflow can become larger than the original
    LP's practical budget. This repair keeps the LP-rounded solution as the
    starting point, moves items out of overloaded PoPs to the cheapest
    currently-capable alternate PoP, and falls back to a full first-fit
    decreasing rebuild if local moves get stuck.
    """
    assignments = dict(assignments)
    load = _assignment_load(assignments, items)
    if _is_feasible(load, pop_cap):
        return assignments

    repaired = _repair_overflow_by_local_moves(assignments, items, pop_cap, load)
    if repaired is not None:
        return repaired

    rebuilt = _repair_overflow_by_first_fit(items, pop_cap)
    if rebuilt is not None:
        return rebuilt

    raise RuntimeError(
        "repair_overflow: could not produce a capacity-feasible integer "
        "assignment with greedy local or full first-fit repair. Given "
        "pop_cap, no feasible integer solution may exist."
    )


def _ranked_cost_lookup(
    ranked: list[tuple[str, float]],
) -> dict[str, float]:
    return {pop_code: cost for pop_code, cost in ranked}


def _repair_overflow_by_local_moves(
    assignments: dict[tuple[int, str], str],
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
    load: dict[str, float],
) -> dict[tuple[int, str], str] | None:
    item_by_key: dict[tuple[int, str], tuple[float, list[tuple[str, float]]]] = {
        (cell_id, dest): (demand, ranked)
        for cell_id, dest, demand, ranked in items
    }
    keys_by_pop: dict[str, set[tuple[int, str]]] = {}
    for key, pop_code in assignments.items():
        keys_by_pop.setdefault(pop_code, set()).add(key)

    while True:
        overloaded = sorted(
            (
                (pop_code, used - pop_cap.get(pop_code, 0.0))
                for pop_code, used in load.items()
                if used > pop_cap.get(pop_code, 0.0) + _FEASIBILITY_TOL
            ),
            key=lambda item: -item[1],
        )
        if not overloaded:
            return assignments

        moved = False
        for pop_code, over in overloaded:
            best_move: tuple[float, float, float, tuple[int, str], str] | None = None
            for key in keys_by_pop.get(pop_code, ()):
                demand, ranked = item_by_key[key]
                cost_by_pop = _ranked_cost_lookup(ranked)
                current_cost = cost_by_pop.get(pop_code, ranked[-1][1])
                for target_pop, target_cost in ranked:
                    if target_pop == pop_code:
                        continue
                    cap = pop_cap.get(target_pop, 0.0)
                    if cap <= 0.0 or load.get(target_pop, 0.0) + demand > cap + _FEASIBILITY_TOL:
                        continue
                    cost_increase = max(0.0, target_cost - current_cost)
                    relieves_overflow = 0 if demand >= over else 1
                    candidate = (
                        relieves_overflow,
                        cost_increase / max(demand, 1e-9),
                        cost_increase,
                        key,
                        target_pop,
                    )
                    if best_move is None or candidate < best_move:
                        best_move = candidate
                    break

            if best_move is None:
                continue

            _relieves, _unit_penalty, _penalty, key, target_pop = best_move
            demand, _ranked = item_by_key[key]
            assignments[key] = target_pop
            keys_by_pop[pop_code].remove(key)
            keys_by_pop.setdefault(target_pop, set()).add(key)
            load[pop_code] = load.get(pop_code, 0.0) - demand
            load[target_pop] = load.get(target_pop, 0.0) + demand
            moved = True
            break

        if not moved:
            return None


def _repair_overflow_by_first_fit(
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str] | None:
    load: dict[str, float] = {}
    out: dict[tuple[int, str], str] = {}
    sorted_items = sorted(
        items,
        key=lambda item: (-item[2], item[1], item[0]),
    )
    for cell_id, dest, demand, ranked in sorted_items:
        chosen_pop = None
        for pop_code, _cost in ranked:
            cap = pop_cap.get(pop_code, 0.0)
            if cap > 0.0 and load.get(pop_code, 0.0) + demand <= cap + _FEASIBILITY_TOL:
                chosen_pop = pop_code
                break
        if chosen_pop is None:
            return None
        out[(cell_id, dest)] = chosen_pop
        load[chosen_pop] = load.get(chosen_pop, 0.0) + demand
    return out


def solve_sub_milp(
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str] | None:
    """Solve a small GAP instance to integer optimality."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    if not items:
        return {}

    c_arr, a_ub, b_ub, a_eq, b_eq, var_index, _item_meta = build_gap_arrays(
        items,
        pop_cap,
    )
    if c_arr.size == 0:
        return None

    integrality = np.ones(c_arr.size, dtype=np.int32)
    bounds = Bounds(lb=0.0, ub=1.0)
    constraints = [
        LinearConstraint(as_sparse(*a_ub), ub=b_ub),
        LinearConstraint(as_sparse(*a_eq), lb=b_eq, ub=b_eq),
    ]
    res = milp(
        c_arr,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False},
    )
    if not res.success or res.x is None:
        return None
    return _decode_assignment(res.x, var_index, items)


def solve_milp(
    items: list[RankedDemandItem],
    pop_cap: Mapping[str, float],
) -> tuple[dict[tuple[int, str], str], float | None, dict[str, object]]:
    """Solve the GAP MILP to proven optimality, with LP-rounded fallback."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    if not items:
        return {}, 0.0, {"status": "optimal", "solver_message": "", "solver_time_s": 0.0}

    c_arr, a_ub, b_ub, a_eq, b_eq, var_index, _item_meta = build_gap_arrays(
        items,
        pop_cap,
    )
    if c_arr.size == 0:
        return {}, 0.0, {"status": "optimal", "solver_message": "empty", "solver_time_s": 0.0}

    integrality = np.ones(c_arr.size, dtype=np.int32)
    bounds = Bounds(lb=0.0, ub=1.0)
    constraints = [
        LinearConstraint(as_sparse(*a_ub), ub=b_ub),
        LinearConstraint(as_sparse(*a_eq), lb=b_eq, ub=b_eq),
    ]

    t_start = time.perf_counter()
    res = milp(
        c_arr,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False},
    )
    solver_time_s = time.perf_counter() - t_start
    message = str(getattr(res, "message", "")) or ""

    if res.success and res.x is not None:
        assignments = _decode_assignment(res.x, var_index, items)
        assignments = repair_overflow(assignments, items, pop_cap)
        return assignments, float(res.fun), {
            "status": "optimal",
            "solver_message": message,
            "solver_time_s": solver_time_s,
        }

    _log.warning(
        "MILP solve failed to return a feasible integer solution "
        "(message=%r). Falling back to LP-rounded.",
        message,
    )
    lp_assignments, lp_obj = solve_lp_and_round(items, pop_cap)
    return lp_assignments, lp_obj, {
        "status": "lp_fallback",
        "solver_message": message,
        "solver_time_s": solver_time_s,
    }
