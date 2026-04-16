"""Tests for TEController Protocol and cost-table controllers."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from vantage.control.controller import (
    SupportsGroundFeedback,
    create_controller,
)
from vantage.control.policy.greedy import (
    ProgressiveController,
    _progressive_filling,
)
from vantage.control.policy.ground_only import GroundOnlyController
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import Cell, CellGrid, CellToPopTable
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
        ctrl = ProgressiveController()
        assert isinstance(ctrl, SupportsGroundFeedback)

    def test_nearest_pop_not_feedback(self) -> None:
        ctrl = NearestPoPController()
        assert not isinstance(ctrl, SupportsGroundFeedback)

    def test_ground_only_not_feedback(self) -> None:
        ctrl = GroundOnlyController()
        assert not isinstance(ctrl, SupportsGroundFeedback)

    def test_greedy_shares_knowledge(self) -> None:
        gk = GroundKnowledge()
        ctrl = ProgressiveController(ground_knowledge=gk)
        assert ctrl.ground_knowledge is gk


@pytest.mark.unit
class TestProgressiveFilling:
    """Bugs found 2026-04-17 in :func:`_progressive_filling` /
    :class:`ProgressiveController` — see top-level chat summary."""

    @staticmethod
    def _grid(endpoint_to_cell: dict[str, int]) -> CellGrid:
        cells = {
            cid: Cell(cell_id=cid, lat_deg=0.0, lon_deg=0.0)
            for cid in set(endpoint_to_cell.values())
        }
        return CellGrid(
            cells=MappingProxyType(cells),
            endpoint_to_cell=MappingProxyType(endpoint_to_cell),
        )

    def test_demand_aggregates_across_endpoints_in_same_cell(self) -> None:
        """Two endpoints in one cell, both sending to the same dest, must
        contribute their *summed* demand to the capacity check.

        Repro: cell={epA, epB}, both → destX. demand epA=5, epB=7. POP1
        cap=8 (can hold 5 OR 7 alone but not the 12 sum). POP2 cap=100.
        Rankings put POP1 first. With the bug, the algorithm only sees
        the first endpoint's demand (5) and picks POP1, blowing past
        cap. With the fix, it sees 12, spills to POP2.
        """
        grid = self._grid({"epA": 1, "epB": 1})
        rankings = {(1, "destX"): [("POP1", 10.0), ("POP2", 20.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP1"}),
            version=0, built_at=0.0,
        )
        demand_per_pair = {("epA", "destX"): 5.0, ("epB", "destX"): 7.0}
        pop_capacity = {"POP1": 8.0, "POP2": 100.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {(1, "destX"): "POP2"}, (
            "expected POP2 (aggregate 12 Gbps exceeds POP1 cap 8); "
            f"got {result} — Bug 1 not fixed"
        )

    def test_dest_names_derived_from_ground_knowledge_when_unset(self) -> None:
        """If a caller passes ``ground_knowledge`` but omits ``dest_names``,
        ProgressiveController should derive the destination set from the
        GK cache instead of silently degenerating to nearest-PoP.

        Without the fix, ``ProgressiveController(ground_knowledge=gk)``
        leaves ``dest_names = ()``, ``rank_pops_by_e2e`` returns ``{}``,
        and the resulting plane has no per-dest overrides at all.
        """
        gk = GroundKnowledge()
        gk.put("POP1", "destA", 10.0)
        gk.put("POP2", "destB", 20.0)
        gk.put("POP1", "destB", 15.0)

        ctrl = ProgressiveController(ground_knowledge=gk)
        assert ctrl.resolve_dest_names() == ("destA", "destB"), (
            "expected dest_names derived from GK entries; "
            "Bug 4 not fixed"
        )

    def test_dest_names_explicit_takes_precedence_over_gk(self) -> None:
        """Explicit ``dest_names`` must override GK-derived defaults so
        callers can scope the planning surface (e.g., per-service)."""
        gk = GroundKnowledge()
        gk.put("POP1", "destA", 10.0)
        gk.put("POP2", "destB", 20.0)

        ctrl = ProgressiveController(
            ground_knowledge=gk, dest_names=("only_this",),
        )
        assert ctrl.resolve_dest_names() == ("only_this",)
