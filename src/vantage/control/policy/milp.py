"""MILP controller for (cell, dest) → PoP assignment (provable optimum).

Same problem as :mod:`vantage.control.policy.lpround`, but the
variables ``x_{i,p}`` are declared **integer** ({0, 1}) so the
solver returns a feasible integer-optimal assignment — the
theoretical global optimum of the Generalised Assignment Problem
formulation (up to the candidate-pruning horizon described below).

    min  Σ_{i,p} d_i · c(i, p) · x_{i,p}
    s.t. Σ_p x_{i,p} = 1                    ∀ i
         Σ_i d_i · x_{i,p} ≤ cap_p          ∀ p
         x_{i,p} ∈ {0, 1}

GAP is NP-hard, so there is **no polynomial-time algorithm** for
this in general. This controller runs in **full-precision mode**:
no candidate pruning, no time budget. HiGHS runs until it proves
integer optimality. At production scale (tens of thousands of
items × ~48 PoPs = 10⁶-level binary variables) that can take
anywhere from seconds to arbitrarily long — there is no
theoretical upper bound on runtime. A refresh-loop timeout or
``Ctrl-C`` is the only abort mechanism.

The controller exposes :attr:`last_solve_meta` so run.py / the
dashboard can show the solver outcome per refresh: ``"optimal"``
when HiGHS proved optimality, ``"failed"`` if the solver returned
an error (in which case the fallback is the LP-relaxation's
rounded assignment).
"""

from __future__ import annotations

import logging
import time
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from vantage.control.costing import build_ground_cost_lookup
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_pop_egress_table,
    build_sat_path_table,
    compute_cell_sat_cost,
    compute_pop_capacity,
    rank_pops_by_e2e,
)
from vantage.control.policy.lpround import (
    _as_sparse,
    _build_lp_arrays,
    _repair_overflow,
    _solve_lp_and_round,
)
from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import CellToPopTable, RoutingPlane
from vantage.model import CellGrid, NetworkSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from vantage.model import PoP


_log = logging.getLogger(__name__)


class MILPController:
    """HiGHS MILP planner with a wall-clock budget.

    Constructor surface mirrors
    :class:`~vantage.control.policy.greedy.GreedyController`.
    No solver-precision knobs — this controller is full-precision
    mode by design.
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
        # Solve outcome metadata: "optimal" / "lp_fallback" / "failed".
        # Exposed for observability.
        self._last_solve_meta: Mapping[str, object] = MappingProxyType({})
        self._last_milp_opt: float | None = None

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> Mapping[str, float]:
        return self._last_timing

    @property
    def last_solve_meta(self) -> Mapping[str, object]:
        return self._last_solve_meta

    @property
    def last_milp_opt(self) -> float | None:
        """Objective value of the most recent MILP solution (integer
        feasible). ``None`` if no feasible integer solution was
        found within the time budget."""
        return self._last_milp_opt

    def resolve_dest_names(self) -> tuple[str, ...]:
        if self._dest_names:
            return self._dest_names
        derived = tuple(sorted({dest for _, dest in self._gk.all_entries()}))
        if not derived and not self._warned_no_dests:
            _log.warning(
                "MILPController.resolve_dest_names: no explicit "
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

        assignments, obj, meta = _solve_milp(
            items=items,
            pop_cap=pop_cap,
        )
        self._last_milp_opt = obj
        self._last_solve_meta = MappingProxyType(meta)
        t_solve = perf()

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
            "milp_solve_ms": (t_solve - t_pop_cap) * 1000.0,
            "assemble_ms": (t_assemble - t_solve) * 1000.0,
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


def _solve_milp(
    items: list[tuple[int, str, float, list[tuple[str, float]]]],
    pop_cap: Mapping[str, float],
) -> tuple[dict[tuple[int, str], str], float | None, dict[str, object]]:
    """Solve the MILP to proven optimality.

    No candidate pruning, no time limit — HiGHS runs until it
    proves integer optimality. On solver error (rare: infeasibility,
    numerical issues) we fall back to the LP-rounded assignment so
    the refresh loop still produces *some* plan.

    Returns ``(assignments, objective, meta_dict)``.

    ``meta_dict`` keys:
      * ``status`` — ``"optimal"`` / ``"lp_fallback"`` / ``"failed"``.
      * ``solver_message`` — raw solver text.
      * ``solver_time_s`` — wall-clock consumed by the MILP call.
    """
    if not items:
        return {}, 0.0, {"status": "optimal", "solver_message": "", "solver_time_s": 0.0}

    c_arr, a_ub, b_ub, a_eq, b_eq, var_index, _item_meta = _build_lp_arrays(
        items, pop_cap,
    )
    if c_arr.size == 0:
        return {}, 0.0, {"status": "optimal", "solver_message": "empty", "solver_time_s": 0.0}

    a_ub_sparse = _as_sparse(*a_ub)
    a_eq_sparse = _as_sparse(*a_eq)

    # Binary-integer constraint: integrality=1 per variable.
    n_vars = c_arr.size
    integrality = np.ones(n_vars, dtype=np.int32)
    bounds = Bounds(lb=0.0, ub=1.0)

    constraints = [
        LinearConstraint(a_ub_sparse, ub=b_ub),
        LinearConstraint(a_eq_sparse, lb=b_eq, ub=b_eq),
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

    def _decode(x: np.ndarray) -> dict[tuple[int, str], str]:
        # x is 0/1 (with tiny numerical noise). Pick the active
        # variable per item. If no variable is active for some
        # item (should not happen on a feasible MILP), that item
        # falls through to baseline at assembly time.
        best: dict[int, tuple[float, str]] = {}
        for var_id, (item_i, pop_code) in enumerate(var_index):
            w = float(x[var_id])
            prev = best.get(item_i)
            if prev is None or w > prev[0]:
                best[item_i] = (w, pop_code)
        assigned: dict[tuple[int, str], str] = {}
        for item_i, (_w, pop_code) in best.items():
            cell_id, dest, _demand, _ranked = items[item_i]
            assigned[(cell_id, dest)] = pop_code
        return assigned

    message = str(getattr(res, "message", "")) or ""
    if res.success and res.x is not None:
        assignments = _decode(res.x)
        # HiGHS may return a solution with microscopic fractionality
        # from presolve; run the same repair pass we use after LP.
        assignments = _repair_overflow(assignments, items, pop_cap)
        return assignments, float(res.fun), {
            "status": "optimal",
            "solver_message": message,
            "solver_time_s": solver_time_s,
        }

    # MILP solver error (no time limit any more, so this path is
    # for genuine failures: infeasibility or numerical issues).
    _log.warning(
        "MILP solve failed to return a feasible integer solution "
        "(message=%r). Falling back to LP-rounded.", message,
    )
    lp_assignments, lp_obj = _solve_lp_and_round(items, pop_cap)
    return lp_assignments, lp_obj, {
        "status": "lp_fallback",
        "solver_message": message,
        "solver_time_s": solver_time_s,
    }
