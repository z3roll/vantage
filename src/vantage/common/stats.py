"""Small statistical helpers shared by control and reporting code."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = ["percentile", "weighted_mean", "weighted_percentile"]


def percentile(data: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile for an unweighted sample."""
    if not data:
        return 0.0
    ordered = sorted(data)
    k = (len(ordered) - 1) * p / 100
    floor = int(k)
    ceil = min(floor + 1, len(ordered) - 1)
    return ordered[floor] * (ceil - k) + ordered[ceil] * (k - floor)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Demand-weighted arithmetic mean: ``sum(value * weight) / sum(weight)``."""
    if not values or not weights:
        return 0.0
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def weighted_percentile(pairs: Iterable[tuple[float, float]], p: float) -> float:
    """Weighted percentile over ``(value, weight)`` pairs."""
    ordered = sorted(pairs, key=lambda item: item[0])
    if not ordered:
        return 0.0
    total = sum(weight for _, weight in ordered)
    if total <= 0.0:
        return 0.0
    threshold = total * p / 100.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]
