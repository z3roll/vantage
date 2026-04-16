"""Metrics analysis: pure functions over EpochResult sequences.

Layer boundary: **analysis** — offline post-hoc computation over results.
Does NOT participate in online decision-making or affect the epoch loop.
Computes latency, throughput, loss, congestion, fairness statistics,
segment decomposition, and controller comparison.
All RTT values in ms, bandwidth in Gbps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vantage.domain import EpochResult


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
    """Average delay decomposition: satellite (terminal->PoP) vs ground (PoP->dest), in ms."""

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


@dataclass(frozen=True, slots=True)
class ThroughputStats:
    """Throughput distribution across flows."""

    count: int
    total_demand_gbps: float
    total_effective_gbps: float
    mean_demand_gbps: float
    mean_effective_gbps: float
    mean_bottleneck_gbps: float
    utilization_ratio: float


@dataclass(frozen=True, slots=True)
class LossStats:
    """Packet loss distribution across flows."""

    count: int
    mean_loss: float
    max_loss: float
    p95_loss: float
    p99_loss: float
    flows_with_loss: int
    flows_with_loss_pct: float


@dataclass(frozen=True, slots=True)
class CongestionStats:
    """Congestion summary across epochs."""

    total_epochs: int
    mean_queuing_rtt: float
    max_queuing_rtt: float
    p95_queuing_rtt: float
    queuing_rtt_fraction: float
    flows_with_queuing: int
    flows_with_queuing_pct: float


@dataclass(frozen=True, slots=True)
class FairnessMetrics:
    """Jain's Fairness Index over effective throughput."""

    jains_index: float
    flow_count: int
    min_throughput_gbps: float
    max_throughput_gbps: float
    cv: float


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_controllers(
    baseline: list[EpochResult],
    improved: list[EpochResult],
    baseline_name: str = "baseline",
    improved_name: str = "improved",
) -> ComparisonResult:
    """Compare two controllers' results flow by flow."""
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


# ---------------------------------------------------------------------------
# Throughput, loss, congestion, fairness
# ---------------------------------------------------------------------------


def compute_throughput_stats(results: list[EpochResult]) -> ThroughputStats:
    """Throughput statistics across all flows and epochs."""
    flows = [f for r in results for f in r.flow_outcomes]
    if not flows:
        return ThroughputStats(0, 0, 0, 0, 0, 0, 0)

    n = len(flows)
    total_demand = sum(f.demand_gbps for f in flows)
    total_eff = sum(f.effective_throughput_gbps for f in flows)
    mean_bneck = sum(f.bottleneck_gbps for f in flows) / n

    return ThroughputStats(
        count=n,
        total_demand_gbps=total_demand,
        total_effective_gbps=total_eff,
        mean_demand_gbps=total_demand / n,
        mean_effective_gbps=total_eff / n,
        mean_bottleneck_gbps=mean_bneck,
        utilization_ratio=total_eff / total_demand if total_demand > 0 else 0,
    )


def compute_loss_stats(results: list[EpochResult]) -> LossStats:
    """Packet loss distribution across all flows and epochs."""
    losses = [f.loss_probability for r in results for f in r.flow_outcomes]
    if not losses:
        return LossStats(0, 0, 0, 0, 0, 0, 0)

    n = len(losses)
    with_loss = sum(1 for p in losses if p > 1e-9)

    return LossStats(
        count=n,
        mean_loss=sum(losses) / n,
        max_loss=max(losses),
        p95_loss=_percentile(losses, 95),
        p99_loss=_percentile(losses, 99),
        flows_with_loss=with_loss,
        flows_with_loss_pct=with_loss / n * 100,
    )


def compute_congestion_stats(results: list[EpochResult]) -> CongestionStats:
    """Queuing delay and congestion statistics."""
    flows = [f for r in results for f in r.flow_outcomes]
    if not flows:
        return CongestionStats(0, 0, 0, 0, 0, 0, 0)

    q_rtts = [f.queuing_rtt for f in flows]
    sat_rtts = [f.satellite_rtt for f in flows]
    n = len(flows)
    total_q = sum(q_rtts)
    total_sat = sum(sat_rtts)
    with_q = sum(1 for q in q_rtts if q > 0.01)

    return CongestionStats(
        total_epochs=len(results),
        mean_queuing_rtt=total_q / n,
        max_queuing_rtt=max(q_rtts),
        p95_queuing_rtt=_percentile(q_rtts, 95),
        queuing_rtt_fraction=total_q / total_sat if total_sat > 0 else 0,
        flows_with_queuing=with_q,
        flows_with_queuing_pct=with_q / n * 100,
    )


def compute_fairness(results: list[EpochResult]) -> FairnessMetrics:
    """Jain's Fairness Index over effective throughput.

    J(x) = (sum x_i)^2 / (n * sum x_i^2)
    """
    tputs = [f.effective_throughput_gbps for r in results for f in r.flow_outcomes]
    if not tputs:
        return FairnessMetrics(1.0, 0, 0, 0, 0)

    n = len(tputs)
    s = sum(tputs)
    sq = sum(x * x for x in tputs)
    jains = (s * s) / (n * sq) if sq > 0 else 1.0

    mean = s / n
    var = sum((x - mean) ** 2 for x in tputs) / n
    cv = math.sqrt(var) / mean if mean > 0 else 0.0

    return FairnessMetrics(
        jains_index=jains,
        flow_count=n,
        min_throughput_gbps=min(tputs),
        max_throughput_gbps=max(tputs),
        cv=cv,
    )
