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

from vantage.control.policy.common.assembly import assemble_assignment_routing_plane
from vantage.control.policy.common.gap import solve_lp_and_round
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
        dest_names, self._warned_no_dests = resolve_policy_dest_names(
            controller_name="LPRoundingController",
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

        # Solve LP + round. On LP failure the controller falls back
        # to an empty assignment (every item defaults to baseline at
        # assembly time). We log the failure but don't raise; refresh
        # loops shouldn't die because the solver hit a numerical edge.
        assignments, lp_opt = solve_lp_and_round(
            items=ctx.items,
            pop_cap=ctx.pop_cap,
        )
        self._last_lp_opt = lp_opt
        t_lp = perf()

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
            "lp_solve_ms": (t_lp - ctx.timing.pop_cap_done) * 1000.0,
            "assemble_ms": assembly.timing_ms["cell_to_pop_ms"],
            "sat_paths_ms": assembly.timing_ms["sat_paths_ms"],
            "pop_egress_ms": assembly.timing_ms["pop_egress_ms"],
        })

        return assembly.plane
