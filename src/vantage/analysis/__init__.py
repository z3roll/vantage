"""Analysis subsystem: offline metrics and comparison.

Layer boundary: analysis is post-hoc computation over results.
Does NOT participate in online decision-making.
"""

from vantage.analysis.metrics import (
    ComparisonResult,
    LatencyStats,
    SegmentBreakdown,
    compare_controllers,
    compute_latency_stats,
    compute_latency_stats_by_dest,
    compute_segment_breakdown,
)

__all__ = [
    "ComparisonResult",
    "LatencyStats",
    "SegmentBreakdown",
    "compare_controllers",
    "compute_latency_stats",
    "compute_latency_stats_by_dest",
    "compute_segment_breakdown",
]
