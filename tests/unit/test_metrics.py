"""Tests for metrics analysis module."""

from __future__ import annotations

import pytest

from vantage.analysis import (
    ComparisonResult,
    LatencyStats,
    SegmentBreakdown,
    compare_controllers,
    compute_latency_stats,
    compute_latency_stats_by_dest,
    compute_segment_breakdown,
)
from vantage.domain import EpochResult, FlowKey, FlowOutcome


def _make_outcome(src: str, dst: str, total_ms: float, **kwargs: float) -> FlowOutcome:
    defaults = dict(
        pop_code="sea", gs_id="gs1", user_sat=0, egress_sat=1,
        satellite_rtt=10.0, ground_rtt=10.0, demand_gbps=0.01,
    )
    defaults.update(kwargs)
    return FlowOutcome(
        flow_key=FlowKey(src, dst),
        total_rtt=total_ms,
        **defaults,
    )


def _make_epoch(flows: list[FlowOutcome]) -> EpochResult:
    total = sum(f.demand_gbps for f in flows)
    return EpochResult(
        epoch=0, flow_outcomes=tuple(flows),
        total_demand_gbps=total, routed_demand_gbps=total, unrouted_demand_gbps=0,
    )


@pytest.mark.unit
class TestLatencyStats:

    def test_basic_stats(self) -> None:
        flows = [
            _make_outcome("a", "g", 10.0),
            _make_outcome("b", "g", 20.0),
            _make_outcome("c", "g", 30.0),
        ]
        stats = compute_latency_stats([_make_epoch(flows)])
        assert stats.count == 3
        assert abs(stats.mean - 20.0) < 0.1
        assert stats.min == 10.0
        assert stats.max == 30.0

    def test_empty(self) -> None:
        stats = compute_latency_stats([])
        assert stats.count == 0

    def test_by_dest(self) -> None:
        flows = [
            _make_outcome("a", "google", 10.0),
            _make_outcome("a", "google", 20.0),
            _make_outcome("a", "wiki", 50.0),
        ]
        by_dest = compute_latency_stats_by_dest([_make_epoch(flows)])
        assert "google" in by_dest
        assert "wiki" in by_dest
        assert by_dest["google"].count == 2
        assert by_dest["wiki"].count == 1


@pytest.mark.unit
class TestSegmentBreakdown:

    def test_breakdown(self) -> None:
        flows = [
            _make_outcome("a", "g", 21.0,
                          satellite_rtt=11.0,
                          ground_rtt=10.0),
        ]
        seg = compute_segment_breakdown([_make_epoch(flows)])
        assert abs(seg.satellite - 11.0) < 0.1
        assert abs(seg.ground - 10.0) < 0.1
        assert seg.satellite_pct > 0
        assert seg.ground_pct > 0

    def test_empty(self) -> None:
        seg = compute_segment_breakdown([])
        assert seg.total == 0


@pytest.mark.unit
class TestCompareControllers:

    def test_improvement(self) -> None:
        baseline = [_make_epoch([
            _make_outcome("a", "g", 50.0),
            _make_outcome("b", "g", 30.0),
        ])]
        improved = [_make_epoch([
            _make_outcome("a", "g", 20.0),  # improved
            _make_outcome("b", "g", 30.0),  # same
        ])]
        cmp = compare_controllers(baseline, improved, "base", "new")
        assert cmp.flows_improved == 1
        assert cmp.flows_same == 1
        assert cmp.flows_worsened == 0
        assert cmp.improvement > 0
        assert cmp.improvement_pct > 0

    def test_no_change(self) -> None:
        epoch = [_make_epoch([_make_outcome("a", "g", 20.0)])]
        cmp = compare_controllers(epoch, epoch)
        assert cmp.flows_improved == 0
        assert cmp.flows_worsened == 0
        assert abs(cmp.improvement) < 0.1

    def test_multi_epoch_no_overwrite(self) -> None:
        """Same flow in two epochs must NOT overwrite — both are counted."""
        epoch0_base = EpochResult(
            epoch=0,
            flow_outcomes=(_make_outcome("a", "g", 50.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        epoch1_base = EpochResult(
            epoch=1,
            flow_outcomes=(_make_outcome("a", "g", 60.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        epoch0_imp = EpochResult(
            epoch=0,
            flow_outcomes=(_make_outcome("a", "g", 20.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        epoch1_imp = EpochResult(
            epoch=1,
            flow_outcomes=(_make_outcome("a", "g", 30.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        cmp = compare_controllers(
            [epoch0_base, epoch1_base],
            [epoch0_imp, epoch1_imp],
            "base", "imp",
        )
        # Both epoch entries counted (total_flows=2, not 1)
        assert cmp.total_flows == 2
        assert cmp.flows_improved == 2
        # avg baseline = (50+60)/2 = 55, avg improved = (20+30)/2 = 25
        assert abs(cmp.baseline_avg - 55.0) < 0.1
        assert abs(cmp.improved_avg - 25.0) < 0.1

    def test_multi_epoch_partial_overlap(self) -> None:
        """Epochs not fully aligned: only matching (epoch, flow) pairs compared."""
        epoch0_base = EpochResult(
            epoch=0,
            flow_outcomes=(_make_outcome("a", "g", 40.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        epoch1_base = EpochResult(
            epoch=1,
            flow_outcomes=(_make_outcome("a", "g", 50.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        # improved only has epoch 0
        epoch0_imp = EpochResult(
            epoch=0,
            flow_outcomes=(_make_outcome("a", "g", 10.0),),
            total_demand_gbps=0.01,
            routed_demand_gbps=0.01,
            unrouted_demand_gbps=0,
        )
        cmp = compare_controllers(
            [epoch0_base, epoch1_base],
            [epoch0_imp],
            "base", "imp",
        )
        # Only epoch 0 overlaps
        assert cmp.total_flows == 1
        assert cmp.flows_improved == 1

    def test_multi_flow_multi_epoch(self) -> None:
        """Multiple flows across multiple epochs: all (epoch, flow) pairs counted."""
        epoch0_base = EpochResult(
            epoch=0,
            flow_outcomes=(
                _make_outcome("a", "g", 50.0),
                _make_outcome("b", "g", 40.0),
            ),
            total_demand_gbps=0.02,
            routed_demand_gbps=0.02,
            unrouted_demand_gbps=0,
        )
        epoch1_base = EpochResult(
            epoch=1,
            flow_outcomes=(
                _make_outcome("a", "g", 55.0),
                _make_outcome("b", "g", 45.0),
            ),
            total_demand_gbps=0.02,
            routed_demand_gbps=0.02,
            unrouted_demand_gbps=0,
        )
        epoch0_imp = EpochResult(
            epoch=0,
            flow_outcomes=(
                _make_outcome("a", "g", 20.0),
                _make_outcome("b", "g", 15.0),
            ),
            total_demand_gbps=0.02,
            routed_demand_gbps=0.02,
            unrouted_demand_gbps=0,
        )
        epoch1_imp = EpochResult(
            epoch=1,
            flow_outcomes=(
                _make_outcome("a", "g", 25.0),
                _make_outcome("b", "g", 20.0),
            ),
            total_demand_gbps=0.02,
            routed_demand_gbps=0.02,
            unrouted_demand_gbps=0,
        )
        cmp = compare_controllers(
            [epoch0_base, epoch1_base],
            [epoch0_imp, epoch1_imp],
            "base", "imp",
        )
        # 2 flows × 2 epochs = 4 total comparisons
        assert cmp.total_flows == 4
        assert cmp.flows_improved == 4
        # avg baseline = (50+40+55+45)/4 = 47.5
        # avg improved = (20+15+25+20)/4 = 20.0
        assert abs(cmp.baseline_avg - 47.5) < 0.1
        assert abs(cmp.improved_avg - 20.0) < 0.1
