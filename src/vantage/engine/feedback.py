"""Feedback observer: aggregate realized truth into per-epoch GK updates.

Contract after the flow-scoped truth refactor:

  * ``GroundTruth`` produces a per-flow RTT inside ``forward.measure``,
    so different flows on the same ``(pop, dest)`` within one epoch
    carry *different* realised RTTs.
  * :class:`GroundDelayFeedback` does NOT forward those per-flow
    samples into :class:`GroundKnowledge` one at a time. Instead, it
    groups all realised ground-RTTs by ``(pop_code, flow_key.dst)``,
    computes the epoch mean + within-epoch population stddev, and
    performs **exactly one** ``knowledge.update`` per
    ``(pop, dest, epoch)``. This keeps GK's ``epoch_count`` an
    epoch-level counter (not a flow counter) and prevents noisy
    high-demand PoPs from flooding the EWMA with their flow count.

Only entries where ``ground_rtt > 0`` are absorbed (``0`` is
reserved for "no measurement"). If every flow on a given pair has
``ground_rtt == 0`` the pair gets no update this epoch.
"""

from __future__ import annotations

import math

from vantage.domain.result import EpochResult
from vantage.world.ground import GroundKnowledge


def _population_stddev(samples: list[float], mean: float) -> float:
    """Return the population stddev of ``samples`` around ``mean``.

    Population (not sample) stddev because we're treating the flows
    that actually landed on this ``(pop, dest)`` as the full
    realisation for the epoch; there's no further "sample of the
    sample" inference to make. With ``len(samples) == 1`` the
    stddev is exactly ``0``.
    """
    n = len(samples)
    if n <= 1:
        return 0.0
    variance = 0.0
    for x in samples:
        diff = x - mean
        variance += diff * diff
    variance /= n
    return math.sqrt(variance)


class GroundDelayFeedback:
    """Absorbs epoch aggregates of realized ground-truth RTT into GK."""

    def __init__(self, knowledge: GroundKnowledge) -> None:
        self._knowledge = knowledge

    def observe(self, result: EpochResult) -> None:
        epoch = int(result.epoch)
        # Bucket per-flow realised RTTs by (pop, dest). Skipping
        # zeros matches the pre-refactor "don't pollute GK with a
        # placeholder" behaviour.
        buckets: dict[tuple[str, str], list[float]] = {}
        for flow_out in result.flow_outcomes:
            rtt = float(flow_out.ground_rtt)
            if rtt <= 0.0:
                continue
            buckets.setdefault(
                (flow_out.pop_code, flow_out.flow_key.dst),
                [],
            ).append(rtt)

        # Exactly one GK update per (pop, dest, epoch), carrying the
        # epoch aggregate — not the raw flow samples.
        for (pop_code, dest), samples in buckets.items():
            n = len(samples)
            mean = sum(samples) / n
            stddev = _population_stddev(samples, mean)
            self._knowledge.update(
                pop_code=pop_code,
                dest=dest,
                epoch_mean_rtt=mean,
                epoch_dev_rtt=stddev,
                epoch=epoch,
            )
