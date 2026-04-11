"""TE engine epoch loop for the routing-plane execution path.

Mirror of :mod:`vantage.engine.run` but wired to the new
``RoutingPlane`` + ``UsageBook`` stack:

    world.snapshot_at(t)                    → NetworkSnapshot
    traffic.generate(epoch)                 → TrafficDemand
    controller.compute_routing_plane(...)   → RoutingPlane         [cached, 15 s cadence]
    CapacityView.from_snapshot(...)         → per-epoch view
    UsageBook(view)                         → per-epoch book
    forward_routing.realize_via_routing_plane(...) → EpochResult
    aggregate capacity stats                → RoutingEpochStats

Two orchestration details worth pinning:

    * **Plane caching**: the plane is re-computed only when
      :meth:`RoutingPlane.is_stale` says so (default cadence 15 s).
      In an experiment with a 300 s epoch interval that still means
      "refresh every epoch", but the logic scales cleanly down to
      sub-15 s epochs and up to tens of seconds without changing the
      control loop.
    * **Per-epoch book**: a fresh :class:`UsageBook` is built every
      epoch. Books are *never* reused — that would accumulate demand
      across epochs and produce nonsense utilization numbers.

This module does **not** touch :mod:`vantage.engine.run`; the legacy
``CostTables`` path remains available in parallel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import (
    CapacityView,
    CellGrid,
    EpochResult,
    RoutingPlane,
    ShellConfig,
    UsageBook,
)
from vantage.engine.context import RunContext
from vantage.engine.run import RunConfig
from vantage.forward_routing import realize_via_routing_plane

__all__ = [
    "RoutingEpochStats",
    "RoutingRunResult",
    "run_routing",
]


@dataclass(frozen=True, slots=True)
class RoutingEpochStats:
    """Per-epoch capacity stats harvested from the :class:`UsageBook`.

    Exposed separately from :class:`EpochResult` so the legacy flow
    bookkeeping (averages, ``FlowOutcome`` tuples, ...) stays
    unchanged and capacity observability is purely additive.
    """

    epoch: int
    max_isl_utilization: float
    max_sat_feeder_utilization: float
    max_gs_feeder_utilization: float
    saturated_isl_count: int
    saturated_sat_feeder_count: int
    saturated_gs_feeder_count: int


@dataclass(frozen=True, slots=True)
class RoutingRunResult:
    """Full-run aggregate with both latency outcomes and capacity stats."""

    config: RunConfig
    epochs: tuple[EpochResult, ...]
    capacity: tuple[RoutingEpochStats, ...]
    wall_time_s: float

    @property
    def num_epochs(self) -> int:
        return len(self.epochs)

    @property
    def avg_total_rtt(self) -> float:
        """Average total RTT across all flows and epochs (ms)."""
        rtts = [f.total_rtt for e in self.epochs for f in e.flow_outcomes]
        return sum(rtts) / len(rtts) if rtts else 0.0


def run_routing(
    context: RunContext,
    cell_grid: CellGrid,
    traffic,  # TrafficGenerator  (import-free to avoid a hard dep here)
    controller: NearestPoPController,
    *,
    config: RunConfig | None = None,
    shell: ShellConfig | None = None,
) -> RoutingRunResult:
    """Drive the routing-plane epoch loop.

    Args:
        context: Shared :class:`RunContext` (world, endpoints,
            ground_knowledge). Reused unchanged.
        cell_grid: Static :class:`CellGrid` built from the endpoint
            population at setup time — each flow's source is looked up
            here to find its cell.
        traffic: Any object with a ``generate(epoch) → TrafficDemand``
            method (the existing ``TrafficGenerator`` protocol).
        controller: A :class:`NearestPoPController` (or any future
            controller exposing ``compute_routing_plane``).
        config: :class:`RunConfig`; defaults to the engine default.
        shell: Optional :class:`ShellConfig` used to build the
            :class:`CapacityView` each epoch. If omitted, it is pulled
            from :attr:`vantage.world.world.WorldModel.shell`.

    Returns:
        A :class:`RoutingRunResult` with per-epoch flow outcomes and
        capacity stats.
    """
    if config is None:
        config = RunConfig()

    epoch_results: list[EpochResult] = []
    capacity_stats: list[RoutingEpochStats] = []
    plane: RoutingPlane | None = None
    t_start = time.perf_counter()

    # Resolve the shell once. Single-shell constellations can rely on
    # the default; multi-shell callers pass it explicitly.
    if shell is None:
        shell = context.world.shell

    # ``gs_by_id`` lookup table for the capacity view (built once —
    # ground infrastructure is static). No snapshot computation needed.
    gs_by_id = {gs.gs_id: gs for gs in context.world.ground_stations}

    for epoch in range(config.num_epochs):
        t = epoch * config.epoch_interval_s
        snapshot = context.world.snapshot_at(epoch, t)

        # Refresh the plane only if it's stale (15 s cadence by default).
        if plane is None or plane.is_stale(t):
            plane = controller.compute_routing_plane(
                snapshot, cell_grid, version=epoch
            )

        # Fresh capacity view + usage book for this epoch.
        view = CapacityView.from_snapshot(
            sat_state=snapshot.satellite,
            shell=shell,
            ground_stations=gs_by_id,
        )
        book = UsageBook(view=view)

        demand = traffic.generate(epoch)
        result = realize_via_routing_plane(
            routing_plane=plane,
            cell_grid=cell_grid,
            usage_book=book,
            snapshot=snapshot,
            demand=demand,
            context=context,
        )
        epoch_results.append(result)
        capacity_stats.append(_collect_capacity_stats(epoch, book))

    wall_time = time.perf_counter() - t_start
    return RoutingRunResult(
        config=config,
        epochs=tuple(epoch_results),
        capacity=tuple(capacity_stats),
        wall_time_s=wall_time,
    )


def _collect_capacity_stats(epoch: int, book: UsageBook) -> RoutingEpochStats:
    """Fold the usage book into a compact per-epoch stat record."""
    # Utilization 0.0 when nothing was charged; keeps max() from raising.
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
    return RoutingEpochStats(
        epoch=epoch,
        max_isl_utilization=max_isl,
        max_sat_feeder_utilization=max_sat,
        max_gs_feeder_utilization=max_gs,
        saturated_isl_count=sat_isl,
        saturated_sat_feeder_count=sat_sat,
        saturated_gs_feeder_count=sat_gs,
    )
