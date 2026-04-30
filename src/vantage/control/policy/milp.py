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

from vantage.control.policy.common.assembly import assemble_assignment_routing_plane
from vantage.control.policy.common.gap import solve_milp
from vantage.control.policy.common.planning import (
    build_e2e_planning_context,
    build_policy_ground_cost,
    resolve_policy_dest_names,
)
from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import RoutingPlane
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
        dest_names, self._warned_no_dests = resolve_policy_dest_names(
            controller_name="MILPController",
            explicit_dest_names=self._dest_names,
            ground_knowledge=self._gk,
            warned_no_dests=self._warned_no_dests,
            logger=_log,
        )
        return dest_names

    def _make_ground_cost(
        self,
        *,
        current_epoch: int,
        pops: Iterable[PoP] | None = None,
        dest_names: Iterable[str] | None = None,
    ) -> Callable[[str, str], float | None]:
        return build_policy_ground_cost(
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
        dest_names = self.resolve_dest_names()
        perf = time.perf_counter

        ctx = build_e2e_planning_context(
            snapshot=snapshot,
            cell_grid=cell_grid,
            ground_knowledge=self._gk,
            dest_names=dest_names,
            demand_per_pair=demand_per_pair or {},
            version=version,
            score_lambda_dev=self._score_lambda_dev,
            score_stale_per_epoch_ms=self._score_stale_per_epoch_ms,
            include_items=True,
        )

        assignments, obj, meta = solve_milp(
            items=ctx.items,
            pop_cap=ctx.pop_cap,
        )
        self._last_milp_opt = obj
        self._last_solve_meta = MappingProxyType(meta)
        t_solve = perf()

        assembly = assemble_assignment_routing_plane(
            snapshot=snapshot,
            baseline=ctx.baseline,
            rankings=ctx.rankings,
            assignments=assignments,
            version=version,
        )

        self._last_timing = MappingProxyType({
            "baseline_ms": (
                ctx.timing.baseline_done - ctx.timing.start
            ) * 1000.0,
            "cell_sat_cost_ms": (
                ctx.timing.cell_sat_cost_done - ctx.timing.baseline_done
            ) * 1000.0,
            "rankings_ms": (
                ctx.timing.rankings_done - ctx.timing.cell_sat_cost_done
            ) * 1000.0,
            "pop_cap_ms": (
                ctx.timing.pop_cap_done - ctx.timing.rankings_done
            ) * 1000.0,
            "milp_solve_ms": (t_solve - ctx.timing.pop_cap_done) * 1000.0,
            "assemble_ms": assembly.timing_ms["cell_to_pop_ms"],
            "sat_paths_ms": assembly.timing_ms["sat_paths_ms"],
            "pop_egress_ms": assembly.timing_ms["pop_egress_ms"],
        })

        return assembly.plane
