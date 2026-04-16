"""Analysis subsystem: offline metrics and comparison.

Layer boundary: analysis is post-hoc computation over results.
Does NOT participate in online decision-making.
"""

from vantage.analysis.metrics import (
    ComparisonResult,
    CongestionStats,
    FairnessMetrics,
    LatencyStats,
    LossStats,
    SegmentBreakdown,
    ThroughputStats,
    compare_controllers,
    compute_congestion_stats,
    compute_fairness,
    compute_latency_stats,
    compute_latency_stats_by_dest,
    compute_loss_stats,
    compute_segment_breakdown,
    compute_throughput_stats,
)

__all__ = [
    "ComparisonResult",
    "CongestionStats",
    "FairnessMetrics",
    "LatencyStats",
    "LossStats",
    "SegmentBreakdown",
    "ThroughputStats",
    "compare_controllers",
    "compute_congestion_stats",
    "compute_fairness",
    "compute_latency_stats",
    "compute_latency_stats_by_dest",
    "compute_loss_stats",
    "compute_segment_breakdown",
    "compute_throughput_stats",
]
