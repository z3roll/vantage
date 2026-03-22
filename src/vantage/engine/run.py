"""TE engine: epoch loop orchestrator.

Execution pipeline per epoch:

    world.snapshot_at(t)    → NetworkSnapshot        [truth layer]
    traffic.generate(epoch) → TrafficDemand          [demand layer]
    controller.optimize()   → RoutingIntent          [control layer]
    forward.realize()       → EpochResult            [execution layer]
    feedback.observe()      → ground_knowledge.put() [feedback layer]

The engine delegates feedback to a FeedbackObserver (default:
GroundDelayFeedback), making the feedback mechanism explicit and
replaceable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from vantage.engine.context import RunContext
from vantage.control.controller import SupportsGroundFeedback, TEController
from vantage.engine_feedback import FeedbackObserver, GroundDelayFeedback
from vantage.forward import realize
from vantage.domain import EpochResult
from vantage.traffic import TrafficGenerator


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for a TE engine run."""

    num_epochs: int = 10
    epoch_interval_s: float = 300.0  # 5 minutes


@dataclass(frozen=True, slots=True)
class RunResult:
    """Aggregated result of a full run."""

    config: RunConfig
    controller_name: str
    epochs: tuple[EpochResult, ...]
    wall_time_s: float

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


def run(
    context: RunContext,
    traffic: TrafficGenerator,
    controller: TEController,
    config: RunConfig | None = None,
    controller_name: str = "unknown",
) -> RunResult:
    """Run the TE engine epoch loop.

    For each epoch:
    1. world.snapshot_at(t) → NetworkSnapshot
    2. traffic.generate(epoch) → TrafficDemand
    3. controller.optimize(snapshot, demand) → RoutingIntent
    4. forward.realize(intent, snapshot, demand, context) → EpochResult
    5. feedback.observe(result) → update ground knowledge
    """
    if config is None:
        config = RunConfig()

    epoch_results: list[EpochResult] = []
    t_start = time.perf_counter()

    # Fill in baseline propagation estimates for calibration
    calibration = context.world.calibration
    if calibration is not None:
        from vantage.world.satellite.visibility import SphericalAccessModel
        snapshot_0 = context.world.snapshot_at(0, 0.0)
        calibration.fill_estimates(
            snapshot_0, context.endpoints, SphericalAccessModel()
        )

    # Set up feedback observer
    feedback: FeedbackObserver | None = None
    if isinstance(controller, SupportsGroundFeedback):
        _fb_ctrl: SupportsGroundFeedback = controller  # type: ignore[assignment]
        assert _fb_ctrl.ground_knowledge is context.ground_knowledge, (
            "Controller's ground_knowledge must be the same object as "
            "context.ground_knowledge to ensure feedback consistency"
        )
        feedback = GroundDelayFeedback(context.ground_knowledge)

    for epoch in range(config.num_epochs):
        t = epoch * config.epoch_interval_s

        snapshot = context.world.snapshot_at(epoch, t)
        demand = traffic.generate(epoch)
        intent = controller.optimize(snapshot, demand)
        result = realize(intent, snapshot, demand, context)
        epoch_results.append(result)

        # Feedback: delegate to observer
        if feedback is not None:
            feedback.observe(result)

    wall_time = time.perf_counter() - t_start

    return RunResult(
        config=config,
        controller_name=controller_name,
        epochs=tuple(epoch_results),
        wall_time_s=wall_time,
    )
