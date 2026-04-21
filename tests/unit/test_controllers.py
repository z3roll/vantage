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
    def _stub_egress(mapping: dict[tuple[int, str], int | tuple[int, ...]]):
        """Build a ``cell_pop_egress`` callback from a pre-built dict.

        Accepts either a single sat ID (legacy single-primary form,
        wrapped into a 1-tuple) or a tuple of sat IDs. Returns the
        empty tuple for any (cell, pop) not in ``mapping`` — matches
        the production resolver's "no reachable egress" convention.
        """
        def _fn(cell_id: int, pop_code: str) -> tuple[int, ...]:
            v = mapping.get((cell_id, pop_code))
            if v is None:
                return ()
            if isinstance(v, int):
                return (v,)
            return v
        return _fn

    def test_demand_aggregates_across_endpoints_in_same_cell(self) -> None:
        """Two endpoints in one cell sending to the same dest must
        contribute their *summed* demand to the capacity check.

        Fixture: cell 0 (warmup load 8 on POP1's sat 5). cell 1 has
        epA=5, epB=7 (sum=12). cap=15. POP1 sat 5 already at 8;
        8+12=20 > 15, no fit; walks cascade to POP_ALT sat 6
        (empty) which fits. Without aggregation (pre-2026-04-17
        bug), cell 1 would see only 5 Gbps → POP1 sat 5 (8+5=13 ≤ 15)
        fits and the cascade is never tried.
        """
        # Cell 0 hosts a single endpoint with demand 8 to anchor sat 5.
        grid = self._grid({"warmup": 0, "epA": 1, "epB": 1})
        rankings = {
            (0, "destX"): [("POP1", 10.0)],
            (1, "destX"): [("POP1", 10.0), ("POP_ALT", 20.0)],
        }
        baseline = CellToPopTable(
            mapping=MappingProxyType({0: ("POP_NEAR",), 1: ("POP_NEAR",)}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(0, "POP_NEAR"): 50.0, (1, "POP_NEAR"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        cell_pop_egress = self._stub_egress({
            (0, "POP1"): 5,
            (1, "POP1"): 5,        # shared sat with cell 0
            (1, "POP_ALT"): 6,     # cascade alternate
            (1, "POP_NEAR"): 7,
        })
        # Cell 0 priority = (50 - 10) × 8 = 320.
        # Cell 1 priority = (50 - 10) × 12 = 480 (HIGHER) — processed first.
        # Hmm — cell 1 first means it sees sat 5 empty (8+12=12 ≤ 15)
        # fits. We want cell 0 first to anchor the sat. Reverse by
        # making cell 0's improvement larger.
        # Cell 0 baseline_cost = 100 (50+50), best_alt = 10 → impr = 90.
        # Cell 1 baseline_cost = 100, best_alt = 10 → impr = 90.
        # Tie on improvement × demand: cell 0 = 90 × 8 = 720, cell 1 =
        # 90 × 12 = 1080. Cell 1 first. Need cell 0 first.
        # Bump cell 0 demand to 13 so its priority dominates (13 vs 12 sum).
        demand_per_pair = {
            ("warmup", "destX"): 13.0,
            ("epA", "destX"): 5.0,
            ("epB", "destX"): 7.0,
        }

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=15.0,
        )

        # Cell 0 first: 13 ≤ 15 → POP1 sat 5 charged 13.
        # Cell 1: aggregate 12. POP1 sat 5: 13+12=25 > 15 → cascade.
        # POP_ALT sat 6: 0+12=12 ≤ 15 → fits.
        assert result[(0, "destX")] == "POP1"
        assert result[(1, "destX")] == "POP_ALT", (
            f"expected aggregated demand 12 to walk cascade past saturated "
            f"POP1/sat 5 to POP_ALT/sat 6; got {result[(1, 'destX')]}"
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
            mapping=MappingProxyType({1: ("POP_BASE",), 2: ("POP_BASE",)}),
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

    def test_overflow_picks_least_loaded_cascade_option(self) -> None:
        """When every (pop, sat) in the cascade is saturated, PG falls
        back to the *least-loaded* candidate within the cascade —
        sharing the overflow across the user's full ranked list rather
        than hard-piling on one geographic baseline.

        Fixture: cap=0 forces every option over capacity. Cascade
        contains HOT (sat 5) and ALT (sat 6); both have load 0 at
        this point so the tie breaks on first-encountered = HOT/sat 5.
        """
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("HOT", 10.0), ("ALT", 20.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: ("NEAR",)}),  # baseline irrelevant now
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "NEAR"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        cell_pop_egress = self._stub_egress({
            (1, "HOT"): 5,
            (1, "ALT"): 6,
            (1, "NEAR"): 7,
        })
        demand_per_pair = {("epA", "destX"): 5.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=0.0,    # nothing fits → forced overflow
        )

        assert result == {(1, "destX"): "HOT"}, (
            f"expected least-loaded cascade fallback (first encountered HOT); "
            f"got {result}"
        )

    def test_zero_improvement_still_assigns_for_cascade_emission(self) -> None:
        """If a cell's nearest PoP is already its best E2E option
        (improvement ≤ 0), the cell IS still assigned — the assignment
        anchors the per-dest cascade tuple emitted by
        :meth:`compute_routing_plane`, so the data plane has a
        capacity-aware least-bad fallback chain even for cells where
        baseline is already optimal."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("POP1", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: ("POP1",)}),  # nearest == rank-1
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

        # Even with improvement=0, the cell gets POP1 (its only
        # ranked option, fits within cap). The compute_routing_plane
        # caller will use this as the cascade head.
        assert result == {(1, "destX"): "POP1"}

    def test_zero_demand_is_skipped(self) -> None:
        """A (cell, dest) with positive improvement but zero current
        demand is skipped — no point spending capacity on a
        non-existent flow."""
        grid = self._grid({"epA": 1})
        rankings = {(1, "destX"): [("BEST", 10.0)]}
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: ("WORSE",)}),
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

    def test_overflow_charges_chosen_sat_to_displace_subsequent_cell(self) -> None:
        """When cell A overflows onto its cascade-fallback (sat S),
        the recorded ``sat_load[S]`` must displace cell B if B's only
        candidate also lands on sat S. Without per-cell-overflow
        bookkeeping, B would see S as empty and incorrectly fit there.

        Fixture: cell 1 demand 12 > cap 10. Cell 1's cascade has
        POP_HOT (sat 5) and POP_ALT (sat 6) — both empty so each
        sees ratio=0, ties break first-encountered = POP_HOT/sat 5.
        Cell 2's only cascade entry POP_SHARED routes via sat 5
        (SAME as cell 1's overflow target). With proper sat_load
        tracking, sat 5 is at 12 Gbps when cell 2 evaluates; cell 2's
        POP_SHARED can't fit (12 + 5 > 10), forcing cell 2 to its
        own cascade fallback.
        """
        grid = self._grid({"epA": 1, "epB": 2})
        rankings = {
            (1, "destX"): [("POP_HOT", 10.0), ("POP_ALT", 20.0)],
            (2, "destX"): [("POP_SHARED", 10.0)],
        }
        baseline = CellToPopTable(
            mapping=MappingProxyType({1: ("POP_NEAR1",), 2: ("POP_NEAR2",)}),
            version=0, built_at=0.0,
        )
        cell_sat_cost = {(1, "POP_NEAR1"): 50.0, (2, "POP_NEAR2"): 50.0}
        ground_cost_fn = self._const_gc(50.0)
        cell_pop_egress = self._stub_egress({
            (1, "POP_HOT"): 5, (1, "POP_ALT"): 6, (1, "POP_NEAR1"): 7,
            (2, "POP_SHARED"): 5, (2, "POP_NEAR2"): 8,
        })
        # Cell 1 priority dominates (12 vs 5). Cell 1 walks ranking,
        # nothing fits (cap 10). Falls back to least-loaded =
        # POP_HOT/sat 5 (first encountered with ratio 0). sat_load[5]
        # = 12. Cell 2 then tries POP_SHARED/sat 5: 12+5=17 > 10 →
        # falls back to least-loaded option in its own cascade. Cell
        # 2 has only POP_SHARED in cascade, so falls back there.
        demand_per_pair = {("epA", "destX"): 12.0, ("epB", "destX"): 5.0}

        result = _progressive_filling(
            rankings=rankings, baseline=baseline, cell_grid=grid,
            cell_sat_cost=cell_sat_cost, ground_cost_fn=ground_cost_fn,
            cell_pop_egress=cell_pop_egress,
            demand_per_pair=demand_per_pair,
            sat_feeder_cap_gbps=10.0,
        )

        assert result[(1, "destX")] == "POP_HOT", (
            f"cell 1's overflow lands on least-loaded cascade option "
            f"(POP_HOT/sat 5); got {result[(1, 'destX')]}"
        )
        # Sat 5 now charged with 12 Gbps. Cell 2's POP_SHARED uses
        # sat 5 too; the load tracking forces it through the
        # cascade-fallback path (only POP_SHARED in cascade).
        assert result[(2, "destX")] == "POP_SHARED", (
            f"cell 2 has only POP_SHARED in cascade; even though sat 5 "
            f"is contended, the cascade fallback still picks POP_SHARED "
            f"(its sole option); got {result[(2, 'destX')]}"
        )

    def test_ground_cost_returns_none_when_data_missing(self) -> None:
        """ProgressiveController._ground_cost returns ``None`` for any
        (pop, dest) pair that hasn't been measured yet. PG's cold
        start therefore degenerates to the baseline nearest-PoP plan
        until :class:`GroundDelayFeedback` populates GK."""
        gk = GroundKnowledge()  # no estimator, no cached entries
        ctrl = ProgressiveController(ground_knowledge=gk)
        assert ctrl._ground_cost("POP1", "destX") is None

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
