"""Unified ground delay knowledge service.

Single source of truth for ground segment delays. Combines:
- L1 cache: observed/feedback values (written by engine each epoch)
- L2/L3 estimator fallback: bootstraps when cache is empty

All consumers (controller, forward, engine) depend on this one service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vantage.world.ground.delay import GroundDelay


class GroundKnowledge:
    """Unified ground delay knowledge store.

    Provides four capabilities:
    1. put(): write delay values (from feedback or estimation)
    2. get(): read cached delay (None if unknown)
    3. get_or_estimate(): read with L2/L3 fallback (bootstraps feedback loop)
    4. best_pop_for(): find PoP with lowest known delay to a destination

    The estimator is injected at construction. Without one, cache misses
    return 0.0 and the feedback loop cannot bootstrap.
    """

    def __init__(self, estimator: GroundDelay | None = None) -> None:
        self._cache: dict[tuple[str, str], float] = {}
        self._known_dests: set[str] = set()
        self._estimator = estimator

    @property
    def estimator(self) -> GroundDelay | None:
        return self._estimator

    def put(self, pop_code: str, dest: str, delay_rtt: float) -> None:
        """Write a ground delay RTT value."""
        self._cache[(pop_code, dest)] = delay_rtt
        self._known_dests.add(dest)

    def get(self, pop_code: str, dest: str) -> float | None:
        """Read cached ground delay RTT. Returns None if not known."""
        return self._cache.get((pop_code, dest))

    def get_or_estimate(
        self,
        pop_code: str,
        dest: str,
        pop_lat: float,
        pop_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> float:
        """Read cached value, falling back to L2/L3 estimation.

        Returns RTT in ms. Returns 0.0 only if no estimator is
        configured and cache misses.
        """
        cached = self._cache.get((pop_code, dest))
        if cached is not None:
            return cached
        if self._estimator is not None:
            # estimate() returns one-way delay; multiply by 2 for RTT
            return self._estimator.estimate(
                pop_lat, pop_lon, dest_lat, dest_lon
            ) * 2
        return 0.0

    def has(self, dest: str) -> bool:
        """Check if any PoP has delay data for this destination."""
        return dest in self._known_dests

    def all_entries(self) -> dict[tuple[str, str], float]:
        """Return a copy of all cached (pop_code, dest) → delay entries."""
        return dict(self._cache)

    def best_pop_for(self, dest: str) -> tuple[str, float] | None:
        """Find the PoP with lowest delay to a destination."""
        best_pop: str | None = None
        best_delay = float("inf")
        for (pop_code, d), delay in self._cache.items():
            if d == dest and delay < best_delay:
                best_delay = delay
                best_pop = pop_code
        if best_pop is None:
            return None
        return best_pop, best_delay


# Backward-compatible alias
GroundDelayCache = GroundKnowledge
