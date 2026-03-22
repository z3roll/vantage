"""Feedback observer: updates ground knowledge from realized results.

Extracted from the engine epoch loop so the feedback mechanism is
explicit, testable, and replaceable. The engine calls
``observer.observe(result)`` after each epoch.
"""

from __future__ import annotations

from typing import Protocol

from vantage.domain.result import EpochResult
from vantage.world.ground import GroundKnowledge


class FeedbackObserver(Protocol):
    """Protocol for post-epoch feedback processing."""

    def observe(self, result: EpochResult) -> None:
        """Process realized results and update knowledge stores."""
        ...


class GroundDelayFeedback:
    """Writes realized ground delays back into GroundKnowledge.

    Only writes entries where ground_rtt > 0 (i.e., a delay was
    actually computed via cache or estimation). Zero values indicate
    missing data and should not pollute the knowledge store.
    """

    def __init__(self, knowledge: GroundKnowledge) -> None:
        self._knowledge = knowledge

    def observe(self, result: EpochResult) -> None:
        for flow_out in result.flow_outcomes:
            if flow_out.ground_rtt > 0:
                self._knowledge.put(
                    flow_out.pop_code,
                    flow_out.flow_key.dst,
                    flow_out.ground_rtt,
                )
