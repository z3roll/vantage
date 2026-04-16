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

    @staticmethod
    def _stub_egress(mapping: dict[tuple[int, str], int]):
        """Build a ``cell_pop_egress`` callback from a pre-built dict.

        Returns ``None`` for any (cell, pop) not in ``mapping`` —
        same convention as the production resolver in greedy.py.
        """
        def _fn(cell_id: int, pop_code: str) -> int | None:
            return mapping.get((cell_id, pop_code))
        return _fn

    def test_demand_aggregates_across_endpoints_in_same_cell(self) -> None:
        """Two endpoints in one cell sending to the same dest must
        contribute their *summed* demand to the capacity check.

        Repro under the per-sat-feeder cap model: cell={epA, epB},
        both → destX. demand epA=5, epB=7 (sum=12). The chosen PoP's
        primary egress sat has cap 10 — it fits 5 OR 7 alone but not
        the 12 sum. With aggregate-demand tracking, the cell falls
        through to the baseline (no other ranking entry in this
        fixture). Without aggregation (the pre-2026-04-17 bug), the
        algorithm would see only 5 Gbps and incorrectly assign POP1.
        """
        grid = self._grid({"epA": 1, "epB": 1})
        rankings = {(1, "destX"): [("POP1", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP_NEAR"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "POP_NEAR"): 50.0}
        ground_cost_fn = self._const_gc(50.0)   # baseline = 100, impr = 90
        cell_pop_egress = self._stub_egress({
            (1, "POP1"): 5,        # primary egress for POP1 = sat 5
            (1, "POP_NEAR"): 6,    # baseline egress = sat 6
        })
        demand_per_pair = {("epA", "destX"): 5.0, ("epB", "destX"): 7.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=10.0,
        )

        # Aggregate 12 > sat cap 10 → POP1 unavailable → fallback to
        # baseline POP_NEAR. With the bug (demand=5), POP1 would fit
        # and be chosen instead.
        assert result == {(1, "destX"): "POP_NEAR"}, (
            f"expected fallback to POP_NEAR (aggregate 12 > 10 sat cap); "
            f"got {result}"
        )

    def test_progressive_uses_sat_feeder_cap_to_pick_alternate_pop(self) -> None:
        """When a cell's primary-PoP-via-primary-sat is saturated by an
        earlier high-priority cell, the next cell should pick a PoP
        whose primary egress is a *different* sat (rather than queueing
        onto the saturated one)."""
        grid = self._grid({"epA": 1, "epB": 2})
        rankings = {
            (1, "destX"): [("POP_BEST", 10.0), ("POP_ALT", 20.0)],
            (2, "destX"): [("POP_BEST", 10.0), ("POP_ALT", 20.0)],
        }
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP_BASE", 2: "POP_BASE"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {
            (1, "POP_BASE"): 50.0, (2, "POP_BASE"): 50.0,
        }
        ground_cost_fn = self._const_gc(50.0)   # baseline = 100, impr = 90
        # Both cells share the same primary egress sat for POP_BEST
        # (sat 5) and the same alternate sat for POP_ALT (sat 6).
        cell_pop_egress = self._stub_egress({
            (1, "POP_BEST"): 5, (1, "POP_ALT"): 6, (1, "POP_BASE"): 7,
            (2, "POP_BEST"): 5, (2, "POP_ALT"): 6, (2, "POP_BASE"): 7,
        })
        # Cell 1 demand 12 fills sat 5 (cap 15) almost full. Cell 2
        # demand 6 wouldn't fit on sat 5 (12+6 > 15) → must pick
        # POP_ALT via sat 6.
        demand_per_pair = {("epA", "destX"): 12.0, ("epB", "destX"): 6.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=15.0,
        )

        # Cell 1 priority = 90 × 12 = 1080 (higher), processed first
        # → POP_BEST (sat 5 charged 12). Cell 2 priority = 90 × 6 =
        # 540, sat 5 is at 12 + 6 = 18 > 15 → falls through to
        # POP_ALT (sat 6 empty).
        assert result == {
            (1, "destX"): "POP_BEST",
            (2, "destX"): "POP_ALT",
        }, (
            f"cell 2 should be displaced to POP_ALT by cell 1's load "
            f"on the shared sat 5; got {result}"
        )

    def test_overflow_falls_back_to_nearest_pop(self) -> None:
        """When every ranked PoP's primary egress sat is saturated,
        the cell falls back to its geographic nearest PoP (baseline),
        not to ranked[0]."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("HOT", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "NEAR"}),  # nearest ≠ HOT
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "NEAR"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        cell_pop_egress = self._stub_egress({
            (1, "HOT"): 5,
            (1, "NEAR"): 6,
        })
        demand_per_pair = {("epA", "destX"): 5.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=0.0,    # nothing fits → forced overflow
        )

        assert result == {(1, "destX"): "NEAR"}, (
            f"expected fallback to baseline NEAR, not rank-1 HOT; got {result}"
        )

    def test_zero_improvement_is_skipped(self) -> None:
        """If a cell's nearest PoP is already its best E2E option
        (improvement ≤ 0), the cell is not assigned an override."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("POP1", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP1"}),  # nearest == rank-1
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "POP1"): 5.0}
        ground_cost_fn = self._const_gc(5.0)   # baseline = 10 = best
        cell_pop_egress = self._stub_egress({(1, "POP1"): 5})
        demand_per_pair = {("epA", "destX"): 1.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=20.0,
        )

        assert result == {}

    def test_zero_demand_is_skipped(self) -> None:
        """A (cell, dest) with positive improvement but zero current
        demand is skipped — no point spending capacity on a
        non-existent flow."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("BEST", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "WORSE"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "WORSE"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        cell_pop_egress = self._stub_egress({(1, "BEST"): 5, (1, "WORSE"): 6})
        demand_per_pair: dict[tuple[str, str], float] = {}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=20.0,
        )

        assert result == {}

    def test_overflow_load_displaces_subsequent_cell(self) -> None:
        """When cell A overflows to its baseline (sat S), the recorded
        ``sat_load[S]`` must displace cell B if B's only candidate
        also lands on sat S. Without bookkeeping, B would see S as
        empty and incorrectly fit there."""
        grid = self._grid({"epA": 1, "epB": 2})
        # Cell 1's only ranking: POP_HOT (forced overflow); Cell 2's
        # only ranking: POP_SHARED whose primary egress is the SAME
        # sat as cell 1's baseline.
        rankings = {
            (1, "destX"): [("POP_HOT", 10.0)],
            (2, "destX"): [("POP_SHARED", 10.0)],
        }
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: "POP_NEAR1", 2: "POP_NEAR2"}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "POP_NEAR1"): 50.0, (2, "POP_NEAR2"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        # Cell 1: POP_HOT egress = sat 5 (will saturate). Baseline
        # NEAR1 egress = sat 7. Cell 2: POP_SHARED egress = sat 7
        # (SAME as cell 1's baseline egress). Baseline NEAR2 egress
        # = sat 8.
        cell_pop_egress = self._stub_egress({
            (1, "POP_HOT"): 5, (1, "POP_NEAR1"): 7,
            (2, "POP_SHARED"): 7, (2, "POP_NEAR2"): 8,
        })
        # Cell 1 priority = 90 × 12 = 1080. Cell 2 priority = 90 × 5
        # = 450. Cell 1 first; demand 12 exceeds the 10-Gbps sat cap
        # so POP_HOT (sat 5) won't fit → fallback to baseline NEAR1
        # via sat 7. Sat 7 then carries 12 Gbps. Cell 2 then tries
        # POP_SHARED via the SAME sat 7 → 12 + 5 > 10 → fallback to
        # baseline NEAR2.
        demand_per_pair = {("epA", "destX"): 12.0, ("epB", "destX"): 5.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=10.0,
        )

        assert result[(1, "destX")] == "POP_NEAR1", (
            f"cell 1's POP_HOT (sat 5) cannot hold 12 Gbps under cap "
            f"10 → must overflow to NEAR1; got {result[(1, 'destX')]}"
        )
        assert result[(2, "destX")] == "POP_NEAR2", (
            f"cell 1's overflow on sat 7 (12 Gbps) must displace cell "
            f"2 (would push sat 7 to 17 > 10) → NEAR2; got "
            f"{result[(2, 'destX')]}"
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
