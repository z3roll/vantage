"""Metrics analysis: pure functions over EpochResult sequences.

Layer boundary: **analysis** — offline post-hoc computation over results.
Does NOT participate in online decision-making or affect the epoch loop.
Computes latency statistics, segment decomposition, and controller comparison.
All RTT values in ms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vantage.domain import EpochResult


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency distribution statistics in ms."""

    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    max: float


@dataclass(frozen=True, slots=True)
class SegmentBreakdown:
    """Average delay decomposition: satellite (terminal→PoP) vs ground (PoP→dest), in ms."""

    satellite: float
    ground: float
    total: float

    @property
    def satellite_pct(self) -> float:
        return self.satellite / self.total * 100 if self.total > 0 else 0

    @property
    def ground_pct(self) -> float:
        return self.ground / self.total * 100 if self.total > 0 else 0


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Comparison between two controllers."""

    baseline_name: str
    improved_name: str
    baseline_avg: float
    improved_avg: float
    improvement: float
    improvement_pct: float
    flows_improved: int
    flows_same: int
    flows_worsened: int
    total_flows: int


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def compute_latency_stats(results: list[EpochResult]) -> LatencyStats:
    """Compute latency distribution across all flows in all epochs."""
    rtts = [f.total_rtt for r in results for f in r.flow_outcomes]
    if not rtts:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    mean = sum(rtts) / len(rtts)
    var = sum((x - mean) ** 2 for x in rtts) / len(rtts)
    return LatencyStats(
        count=len(rtts),
        mean=mean,
        std=math.sqrt(var),
        min=min(rtts),
        p25=_percentile(rtts, 25),
        p50=_percentile(rtts, 50),
        p75=_percentile(rtts, 75),
        p95=_percentile(rtts, 95),
        p99=_percentile(rtts, 99),
        max=max(rtts),
    )


def compute_segment_breakdown(results: list[EpochResult]) -> SegmentBreakdown:
    """Compute average delay decomposition: satellite vs ground."""
    flows = [f for r in results for f in r.flow_outcomes]
    if not flows:
        return SegmentBreakdown(0, 0, 0)

    n = len(flows)
    return SegmentBreakdown(
        satellite=sum(f.satellite_rtt for f in flows) / n,
        ground=sum(f.ground_rtt for f in flows) / n,
        total=sum(f.total_rtt for f in flows) / n,
    )


def compute_latency_stats_by_dest(
    results: list[EpochResult],
) -> dict[str, LatencyStats]:
    """Compute latency stats grouped by destination."""
    by_dest: dict[str, list[float]] = {}
    for r in results:
        for f in r.flow_outcomes:
            by_dest.setdefault(f.flow_key.dst, []).append(f.total_rtt)

    out: dict[str, LatencyStats] = {}
    for dest, rtts in sorted(by_dest.items()):
        mean = sum(rtts) / len(rtts)
        var = sum((x - mean) ** 2 for x in rtts) / len(rtts)
        out[dest] = LatencyStats(
            count=len(rtts),
            mean=mean,
            std=math.sqrt(var),
            min=min(rtts),
            p25=_percentile(rtts, 25),
            p50=_percentile(rtts, 50),
            p75=_percentile(rtts, 75),
            p95=_percentile(rtts, 95),
            p99=_percentile(rtts, 99),
            max=max(rtts),
        )
    return out


def compare_controllers(
    baseline: list[EpochResult],
    improved: list[EpochResult],
    baseline_name: str = "baseline",
    improved_name: str = "improved",
) -> ComparisonResult:
    """Compare two controllers' results flow by flow.

    Keys by (epoch, flow_key) so that multi-epoch results are compared
    correctly without later epochs overwriting earlier ones.
    """
    b_map = {(r.epoch, f.flow_key): f.total_rtt
             for r in baseline for f in r.flow_outcomes}
    i_map = {(r.epoch, f.flow_key): f.total_rtt
             for r in improved for f in r.flow_outcomes}

    common = set(b_map.keys()) & set(i_map.keys())
    b_avg = sum(b_map[k] for k in common) / len(common) if common else 0
    i_avg = sum(i_map[k] for k in common) / len(common) if common else 0

    flows_improved = sum(1 for k in common if i_map[k] < b_map[k] - 0.1)
    flows_worsened = sum(1 for k in common if i_map[k] > b_map[k] + 0.1)
    flows_same = len(common) - flows_improved - flows_worsened

    return ComparisonResult(
        baseline_name=baseline_name,
        improved_name=improved_name,
        baseline_avg=b_avg,
        improved_avg=i_avg,
        improvement=b_avg - i_avg,
        improvement_pct=(b_avg - i_avg) / b_avg * 100 if b_avg > 0 else 0,
        flows_improved=flows_improved,
        flows_same=flows_same,
        flows_worsened=flows_worsened,
        total_flows=len(common),
    )
