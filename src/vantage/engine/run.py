"""TE engine: epoch loop orchestrator.

RoutingPlane path:

    snapshot → controller.compute_routing_plane → RoutingPlaneForward
    → realize  (+ UsageBook capacity tracking)
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

from vantage.control.controller import SupportsGroundFeedback
from vantage.domain import (
    CapacityView,
    CellGrid,
    EpochResult,
    RoutingPlane,
    ShellConfig,
    UsageBook,
)
from vantage.engine.context import RunContext
from vantage.engine.feedback import FeedbackObserver, GroundDelayFeedback
from vantage.forward import (
    RoutingPlaneForward,
    realize,
)
from vantage.traffic import TrafficGenerator

# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for a TE engine run."""

    num_epochs: int = 10
    epoch_interval_s: float = 300.0  # 5 minutes


@dataclass(frozen=True, slots=True)
class RoutingEpochStats:
    """Per-epoch capacity stats harvested from the :class:`UsageBook`."""

    epoch: int
    max_isl_utilization: float
    max_sat_feeder_utilization: float
    max_gs_feeder_utilization: float
    saturated_isl_count: int
    saturated_sat_feeder_count: int
    saturated_gs_feeder_count: int
    # Top overloaded links: list of (id, utilization, load_gbps, capacity_gbps)
    top_sat_feeders: tuple[tuple[int, float, float, float], ...] = ()
    top_gs_feeders: tuple[tuple[str, float, float, float], ...] = ()
    top_isls: tuple[tuple[tuple[int, int], float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class RunResult:
    """Aggregated result of a full run.

    ``capacity`` is populated by :func:`run_routing`.
    """

    config: RunConfig
    controller_name: str
    epochs: tuple[EpochResult, ...]
    wall_time_s: float
    capacity: tuple[RoutingEpochStats, ...] | None = field(default=None)

    @property
    def num_epochs(self) -> int:
        return len(self.epochs)

    @property
    def avg_total_rtt(self) -> float:
        """Average total RTT across all flows and epochs, in ms."""
        all_rtts = [
            f.total_rtt
            for epoch in self.epochs
            for f in epoch.flow_outcomes
        ]
        return sum(all_rtts) / len(all_rtts) if all_rtts else 0.0


# ---------------------------------------------------------------------------
# RoutingPlane execution path
# ---------------------------------------------------------------------------


def run_routing(
    context: RunContext,
    cell_grid: CellGrid,
    traffic,  # TrafficGenerator
    controller,  # any controller with compute_routing_plane()
    *,
    config: RunConfig | None = None,
    controller_name: str = "routing",
    shell: ShellConfig | None = None,
) -> RunResult:
    """Run the RoutingPlane-based epoch loop.

    Supports any controller with ``compute_routing_plane()``.
    If the controller has ``ground_knowledge``, a feedback observer is
    set up to populate it from observed traffic (passive learning).
    """
    if config is None:
        config = RunConfig()

    epoch_results: list[EpochResult] = []
    capacity_stats: list[RoutingEpochStats] = []
    plane: RoutingPlane | None = None
    t_start = time.perf_counter()

    if shell is None:
        shell = context.world.shell

    gs_by_id = {gs.gs_id: gs for gs in context.world.ground_stations}

    # Set up feedback observer if controller uses ground knowledge
    feedback: FeedbackObserver | None = None
    ctrl_gk = getattr(controller, "ground_knowledge", None)
    if ctrl_gk is not None and ctrl_gk is context.ground_knowledge:
        feedback = GroundDelayFeedback(ctrl_gk)

    # Per-PoP capacity = sum of max_capacity across all connected GSs.
    # Each GS has 8 Ka-band antennas × 20 Gbps = 160 Gbps.
    snap0 = context.world.snapshot_at(0, 0.0)
    gs_map = {gs.gs_id: gs for gs in context.world.ground_stations}
    pop_capacity: dict[str, float] = {}
    for pop in snap0.infra.pops:
        total_cap = 0.0
        for gs_id, _ in snap0.infra.pop_gs_edges(pop.code):
            gs = gs_map.get(gs_id)
            if gs:
                total_cap += gs.max_capacity
        pop_capacity[pop.code] = total_cap

    # Cache the controller signature once instead of re-introspecting per epoch.
    controller_sig = inspect.signature(controller.compute_routing_plane)
    controller_wants_demand = "demand_per_pair" in controller_sig.parameters

    for epoch in range(config.num_epochs):
        t = epoch * config.epoch_interval_s

        # Advance clocks for time-aware components
        context.ground_knowledge.set_clock(t)
        estimator = context.ground_knowledge.estimator
        if hasattr(estimator, "set_time"):
            estimator.set_time(t)

        snapshot = context.world.snapshot_at(epoch, t)

        # Generate this epoch's demand BEFORE invoking the controller so
        # capacity-aware policies see ground-truth demand from epoch 0
        # onwards. Earlier code passed `prev_demand_per_pair` derived
        # from the previous epoch's `flow_outcomes`, which (a) was empty
        # at epoch 0 — silently disabling capacity awareness on the
        # first plane — and (b) dropped any flow `realize` couldn't
        # route (no visible sat / no FIB / no ground RTT), causing the
        # controller to under-estimate demand for chronically dropped
        # pairs.
        demand = traffic.generate(epoch)

        if plane is None or plane.is_stale(t):
            kwargs: dict = {"version": epoch}
            if controller_wants_demand:
                kwargs["demand_per_pair"] = {
                    (fk.src, fk.dst): d for fk, d in demand.flows.items()
                }
                kwargs["pop_capacity_gbps"] = pop_capacity
            plane = controller.compute_routing_plane(
                snapshot, cell_grid, **kwargs
            )

        view = CapacityView.from_snapshot(
            sat_state=snapshot.satellite,
            shell=shell,
            ground_stations=gs_by_id,
        )
        book = UsageBook(view=view)

        # path_table is no longer threaded through — RoutingPlaneForward
        # builds its top-K options lazily per (ingress, pop) on first
        # touch within this realize call.
        strategy = RoutingPlaneForward(plane, cell_grid, book)
        result = realize(strategy, snapshot, demand, context)

        # Feedback: update ground knowledge from observed flows
        if feedback is not None:
            feedback.observe(result)

        epoch_results.append(result)
        capacity_stats.append(_collect_capacity_stats(epoch, book))

    wall_time = time.perf_counter() - t_start
    return RunResult(
        config=config,
        controller_name=controller_name,
        epochs=tuple(epoch_results),
        capacity=tuple(capacity_stats),
        wall_time_s=wall_time,
    )


def _collect_capacity_stats(
    epoch: int, book: UsageBook, top_n: int = 5,
) -> RoutingEpochStats:
    """Fold the usage book into a compact per-epoch stat record."""
    max_isl = max(
        (book.isl_utilization(a, b) for (a, b) in book.isl_used),
        default=0.0,
    )
    max_sat = max(
        (book.sat_feeder_utilization(s) for s in book.sat_feeder_used),
        default=0.0,
    )
    max_gs = max(
        (book.gs_feeder_utilization(g) for g in book.gs_feeder_used),
        default=0.0,
    )
    sat_isl = sum(1 for (a, b) in book.isl_used if book.is_isl_saturated(a, b))
    sat_sat = sum(1 for s in book.sat_feeder_used if book.is_sat_feeder_saturated(s))
    sat_gs = sum(1 for g in book.gs_feeder_used if book.is_gs_feeder_saturated(g))

    # Top-N overloaded links: (id, utilization, load_gbps, capacity_gbps)
    top_sf = sorted(
        (
            (s, book.sat_feeder_utilization(s),
             book.sat_feeder_used[s], book.view.sat_feeder_cap(s))
            for s in book.sat_feeder_used
        ),
        key=lambda x: -x[1],
    )[:top_n]

    top_gf = sorted(
        (
            (g, book.gs_feeder_utilization(g),
             book.gs_feeder_used[g], book.view.gs_feeder_cap(g))
            for g in book.gs_feeder_used
        ),
        key=lambda x: -x[1],
    )[:top_n]

    top_il = sorted(
        (
            ((a, b), book.isl_utilization(a, b),
             book.isl_used[(a, b)], book.view.isl_cap(a, b))
            for (a, b) in book.isl_used
        ),
        key=lambda x: -x[1],
    )[:top_n]

    return RoutingEpochStats(
        epoch=epoch,
        max_isl_utilization=max_isl,
        max_sat_feeder_utilization=max_sat,
        max_gs_feeder_utilization=max_gs,
        saturated_isl_count=sat_isl,
        saturated_sat_feeder_count=sat_sat,
        saturated_gs_feeder_count=sat_gs,
        top_sat_feeders=tuple(top_sf),
        top_gs_feeders=tuple(top_gf),
        top_isls=tuple(top_il),
    )
