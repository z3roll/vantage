"""Tests for TEController Protocol and cost-table controllers."""

from __future__ import annotations

import pytest

from vantage.control.controller import (
    SupportsGroundFeedback,
    create_controller,
)
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.control.policy.ground_only import GroundOnlyController
from vantage.control.policy.greedy import VantageGreedyController
from vantage.world.ground import GroundKnowledge


@pytest.mark.unit
class TestControllerFactory:

    def test_create_nearest_pop(self) -> None:
        ctrl = create_controller("nearest_pop")
        assert isinstance(ctrl, NearestPoPController)

    def test_create_ground_only(self) -> None:
        ctrl = create_controller("ground_only")
        assert isinstance(ctrl, GroundOnlyController)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown controller"):
            create_controller("nonexistent")

    def test_greedy_supports_feedback_protocol(self) -> None:
        ctrl = VantageGreedyController()
        assert isinstance(ctrl, SupportsGroundFeedback)

    def test_nearest_pop_not_feedback(self) -> None:
        ctrl = NearestPoPController()
        assert not isinstance(ctrl, SupportsGroundFeedback)

    def test_ground_only_not_feedback(self) -> None:
        ctrl = GroundOnlyController()
        assert not isinstance(ctrl, SupportsGroundFeedback)

    def test_greedy_shares_knowledge(self) -> None:
        gk = GroundKnowledge()
        ctrl = VantageGreedyController(ground_knowledge=gk)
        assert ctrl.ground_knowledge is gk
