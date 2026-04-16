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

    @staticmethod
    def _const_gc(value: float = 5.0):
        """Build a ground_cost_fn that returns a constant for any (pop, dest)."""
        def _fn(pop: str, dest: str) -> float:
            del pop, dest
            return value
        return _fn

    def test_demand_aggregates_across_endpoints_in_same_cell(self) -> None:
        """Two endpoints in one cell, both sending to the same dest, must
        contribute their *summed* demand to the capacity check.

        Repro: cell={epA, epB}, both → destX. demand epA=5, epB=7
        (sum=12). POP1 cap=8 (can hold 5 OR 7 alone but not the 12
        sum). POP2 cap=100. Rankings put POP1 first (best E2E).
        Baseline is POP3 with high E2E cost so this (cell, dest) has
        positive improvement and actually enters the work queue.
        With the demand-aggregation fix, the algorithm sees the full
        12 Gbps, can't fit on POP1 (cap=8), spills to POP2.
        """
        grid = self._grid({"epA": 1, "epB": 1})
        rankings = {(1, "destX"): [("POP1", 10.0), ("POP2", 20.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP3"}),  # baseline ≠ any rank
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "POP3"): 50.0}
        # baseline_cost(1, destX) = 50 + 50 = 100; improvement = 100 - 10 = 90.
        ground_cost_fn = self._const_gc(50.0)
        demand_per_pair = {("epA", "destX"): 5.0, ("epB", "destX"): 7.0}
        pop_capacity = {"POP1": 8.0, "POP2": 100.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {(1, "destX"): "POP2"}, (
            "expected POP2 (aggregate 12 Gbps exceeds POP1 cap 8); "
            f"got {result}"
        )

    def test_sorts_by_improvement_times_demand(self) -> None:
        """Two cells compete for limited capacity on PoP 'BEST'.

        - Cell 1: demand 0.5 Gbps, baseline 20, best 10 → impr 10,
          value = 10 × 0.5 = 5.
        - Cell 2: demand 10 Gbps, baseline 15, best 10 → impr 5,
          value = 5 × 10 = 50.

        Pure improvement sort would prefer cell 1 (impr 10 > 5).
        ``improvement × demand`` sort prefers cell 2 (50 > 5),
        because giving it the BEST PoP saves more total RTT-Gbps.
        BEST cap=10 fits exactly one of the two; verify cell 2 wins.
        """
        grid = self._grid({"ep1": 1, "ep2": 2})
        rankings = {
            (1, "destX"): [("BEST", 10.0), ("ALT", 30.0)],
            (2, "destX"): [("BEST", 10.0), ("ALT", 30.0)],
        }
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "BASE1", 2: "BASE2"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {
            (1, "BASE1"): 10.0,  # baseline_cost(1) = 10 + 10 = 20
            (2, "BASE2"): 5.0,   # baseline_cost(2) = 5 + 10 = 15
        }
        ground_cost_fn = self._const_gc(10.0)
        demand_per_pair = {("ep1", "destX"): 0.5, ("ep2", "destX"): 10.0}
        pop_capacity = {"BEST": 10.0, "ALT": 100.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {
            (2, "destX"): "BEST",
            (1, "destX"): "ALT",
        }, (
            f"cell 2 (improvement × demand = 50) should win BEST over "
            f"cell 1 (= 5); got {result}"
        )

    def test_overflow_falls_back_to_nearest_pop(self) -> None:
        """When every PoP in a cell's ranking is over capacity, the
        cell falls back to its geographic nearest PoP (baseline), not
        to ranked[0]. Pre-fix piled all overflow onto the most popular
        E2E-best PoP; new behaviour scatters overflow across nearests.
        """
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("HOT", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "NEAR"}),  # nearest ≠ HOT
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "NEAR"): 50.0}
        ground_cost_fn = self._const_gc(50.0)  # baseline_cost = 100
        demand_per_pair = {("epA", "destX"): 5.0}
        pop_capacity = {"HOT": 0.0}  # no room → forced overflow

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {(1, "destX"): "NEAR"}, (
            f"expected fallback to baseline NEAR, not rank-1 HOT; got {result}"
        )

    def test_zero_improvement_is_skipped(self) -> None:
        """If a cell's nearest PoP is already its best E2E option
        (improvement ≤ 0), the cell is not assigned an override —
        data plane will fall back to baseline at lookup time."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("POP1", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP1"}),  # nearest == rank-1
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "POP1"): 5.0}
        ground_cost_fn = self._const_gc(5.0)  # baseline = 5+5 = 10 = best
        demand_per_pair = {("epA", "destX"): 1.0}
        pop_capacity = {"POP1": 100.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {}, (
            f"zero-improvement cells must not be assigned; got {result}"
        )

    def test_zero_demand_is_skipped(self) -> None:
        """A (cell, dest) with positive improvement but zero current
        demand is skipped — there's no point spending capacity on a
        non-existent flow."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("BEST", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "WORSE"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "WORSE"): 50.0}
        ground_cost_fn = self._const_gc(50.0)  # impr = 100 - 10 = 90
        demand_per_pair: dict[tuple[str, str], float] = {}  # zero demand
        pop_capacity = {"BEST": 100.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {}

    def test_overflow_load_displaces_subsequent_cell(self) -> None:
        """The overflow path's ``pop_load[nearest] += demand`` is only
        load-bearing if a *later* cell would otherwise wrongly fit on
        that same PoP. Without the bookkeeping, cell B would see
        NEAR as empty and assign to it; with it, B is correctly
        displaced.

        Setup: two cells, both processed by improvement-first sort.
        - Cell 1 has only HOT in its ranking (cap=0) → forced to
          overflow to its nearest = NEAR. demand=8 → NEAR sees 8.
        - Cell 2 has only NEAR in its ranking (cap=10). Without
          load tracking, cell 2 would fit (5 ≤ 10). With it,
          pop_load[NEAR]=8 already, so 8+5 > 10 and cell 2 is
          forced to overflow to its nearest = OTHER.
        """
        grid = self._grid({"epA": 1, "epB": 2})
        rankings = {
            (1, "destX"): [("HOT", 10.0)],
            (2, "destX"): [("NEAR", 10.0)],
        }
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "NEAR", 2: "OTHER"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "NEAR"): 50.0, (2, "OTHER"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        # Both cells have the same improvement (= 90); priority is
        # impr × demand. Cell 1: 90 × 8 = 720. Cell 2: 90 × 5 = 450.
        # Cell 1 processed first → overflows to NEAR → NEAR sees 8.
        demand_per_pair = {("epA", "destX"): 8.0, ("epB", "destX"): 5.0}
        pop_capacity = {"HOT": 0.0, "NEAR": 10.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            demand_per_pair=demand_per_pair, pop_capacity_gbps=pop_capacity,
        )

        assert result == {
            (1, "destX"): "NEAR",
            (2, "destX"): "OTHER",
        }, (
            f"cell 1's overflow onto NEAR (load=8) must displace cell 2 "
            f"to OTHER; got {result}"
        )

    def test_ground_cost_propagates_keyerror_when_data_missing(self) -> None:
        """ProgressiveController._ground_cost no longer swallows
        KeyError. Missing (pop, dest) data surfaces as a hard error
        so the operator notices instead of the algorithm silently
        dropping the PoP from the candidate set."""
        gk = GroundKnowledge()  # no estimator, no cached entries
        ctrl = ProgressiveController(ground_knowledge=gk)
        with pytest.raises(KeyError):
            ctrl._ground_cost("POP1", "destX")

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
