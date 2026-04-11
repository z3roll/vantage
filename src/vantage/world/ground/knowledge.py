"""Unified ground delay knowledge service.

Single source of truth for ground segment delays.
Supports per-PoP capacity limits with pluggable eviction policy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

from vantage.world.ground.cache_keys import encode_class_key, is_class_key

if TYPE_CHECKING:
    from vantage.world.ground.delay import GroundDelay
    from vantage.world.ground.profiled_delay import ServiceGroundDelay


class CacheEvictionPolicy(Protocol):
    """Decides which entry to evict when a PoP's cache is full."""

    def record_access(self, pop_code: str, dest: str) -> None: ...
    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None: ...


class LRUEviction:
    """Evict least recently used entry."""

    def __init__(self) -> None:
        self._access_order: dict[str, list[str]] = defaultdict(list)

    def record_access(self, pop_code: str, dest: str) -> None:
        order = self._access_order[pop_code]
        if dest in order:
            order.remove(dest)
        order.append(dest)

    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None:
        for dest in self._access_order.get(pop_code, []):
            if dest in entries:
                return dest
        return next(iter(entries), None)


class LFUEviction:
    """Evict least frequently used entry."""

    def __init__(self) -> None:
        self._freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_access(self, pop_code: str, dest: str) -> None:
        self._freq[pop_code][dest] += 1

    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None:
        freq = self._freq.get(pop_code, {})
        return min(entries, key=lambda d: freq.get(d, 0), default=None)


class TTLEviction:
    """Evict oldest entry (by insertion time)."""

    def __init__(self) -> None:
        self._insert_time: dict[str, dict[str, float]] = defaultdict(dict)
        self._clock = 0.0

    def set_time(self, t: float) -> None:
        self._clock = t

    def record_access(self, pop_code: str, dest: str) -> None:
        if dest not in self._insert_time.get(pop_code, {}):
            self._insert_time[pop_code][dest] = self._clock

    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None:
        times = self._insert_time.get(pop_code, {})
        return min(entries, key=lambda d: times.get(d, 0.0), default=None)


class TrafficWeightedEviction:
    """Evict entry with lowest traffic weight."""

    def __init__(self) -> None:
        self._weight: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_access(self, pop_code: str, dest: str) -> None:
        self._weight[pop_code][dest] += 1

    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None:
        w = self._weight.get(pop_code, {})
        return min(entries, key=lambda d: w.get(d, 0), default=None)


class GroundKnowledge:
    """Unified ground delay knowledge store with per-PoP capacity limits.

    Each PoP has a bounded cache of (dest → delay_rtt) entries.
    When a PoP's cache exceeds capacity, the eviction policy decides
    which entry to remove.
    """

    def __init__(
        self,
        estimator: GroundDelay | None = None,
        pop_capacity: int = 0,
        eviction: CacheEvictionPolicy | None = None,
    ) -> None:
        # Per-PoP caches: {pop_code → {dest → delay_rtt}}
        self._per_pop: dict[str, dict[str, float]] = defaultdict(dict)
        self._known_dests: set[str] = set()
        self._estimator = estimator
        self._pop_capacity = pop_capacity  # 0 = unlimited
        self._eviction = eviction

    @property
    def estimator(self) -> GroundDelay | None:
        return self._estimator

    def put(self, pop_code: str, dest: str, delay_rtt: float) -> None:
        """Write a ground delay RTT value."""
        pop_cache = self._per_pop[pop_code]

        if dest in pop_cache:
            # Update existing entry
            pop_cache[dest] = delay_rtt
            if self._eviction is not None:
                self._eviction.record_access(pop_code, dest)
            return

        # New entry — check capacity
        if (
            self._pop_capacity > 0
            and len(pop_cache) >= self._pop_capacity
            and self._eviction is not None
        ):
            # Class-level keys are protected from eviction
            evictable = {k: v for k, v in pop_cache.items() if not is_class_key(k)}
            if evictable:
                victim = self._eviction.select_victim(pop_code, evictable)
                if victim is not None:
                    del pop_cache[victim]

        pop_cache[dest] = delay_rtt
        self._known_dests.add(dest)
        if self._eviction is not None:
            self._eviction.record_access(pop_code, dest)

    def get(self, pop_code: str, dest: str) -> float | None:
        """Read cached ground delay RTT. Returns None if not known."""
        pop_cache = self._per_pop.get(pop_code)
        if pop_cache is None:
            return None
        val = pop_cache.get(dest)
        if val is not None and self._eviction is not None:
            self._eviction.record_access(pop_code, dest)
        return val

    def get_or_estimate(self, pop_code: str, dest: str) -> float:
        """Read cached value, falling back to the measurement estimator.

        The estimator (when present) is a :class:`GroundDelay` that
        returns **one-way** RTT in ms. We double it here so consumers
        get a round-trip value consistent with the cached values.

        Raises:
            KeyError: If the pair has neither a cached entry nor a
                measurement available in the estimator. Callers that
                want a graceful miss should call :meth:`get` and
                handle ``None`` themselves.
        """
        cached = self.get(pop_code, dest)
        if cached is not None:
            return cached
        if self._estimator is None:
            raise KeyError(
                f"no cached RTT and no estimator for "
                f"(pop={pop_code!r}, dest={dest!r})"
            )
        return self._estimator.estimate(pop_code, dest) * 2

    def has(self, dest: str) -> bool:
        """Check if any PoP has delay data for this destination."""
        return dest in self._known_dests

    def all_entries(self) -> dict[tuple[str, str], float]:
        """Return all cached (pop_code, dest) → delay entries."""
        result: dict[tuple[str, str], float] = {}
        for pop_code, pop_cache in self._per_pop.items():
            for dest, delay in pop_cache.items():
                result[(pop_code, dest)] = delay
        return result

    def pop_entries(self, pop_code: str) -> dict[str, float]:
        """Return {dest → delay} for a single PoP."""
        return dict(self._per_pop.get(pop_code, {}))

    def pop_size(self, pop_code: str) -> int:
        """Number of cached destinations for a PoP."""
        return len(self._per_pop.get(pop_code, {}))

    def total_size(self) -> int:
        """Total entries across all PoPs."""
        return sum(len(v) for v in self._per_pop.values())

    def best_pop_for(self, dest: str) -> tuple[str, float] | None:
        """Find the PoP with lowest delay to a destination."""
        best_pop: str | None = None
        best_delay = float("inf")
        for pop_code, pop_cache in self._per_pop.items():
            delay = pop_cache.get(dest)
            if delay is not None and delay < best_delay:
                best_delay = delay
                best_pop = pop_code
        if best_pop is None:
            return None
        return best_pop, best_delay

    # ── Service-class cache methods ──────────────────────

    def put_class(self, pop_code: str, service_class: str, delay_rtt: float) -> None:
        """Write a class-level ground delay RTT value.

        Class-level keys are protected from eviction by LRU/LFU policies.
        """
        key = encode_class_key(service_class)
        self.put(pop_code, key, delay_rtt)

    def put_class_time(
        self,
        pop_code: str,
        service_class: str,
        day_type: str,
        local_hour: int,
        delay_rtt: float,
    ) -> None:
        """Write a time-scoped class-level ground delay RTT value."""
        key = encode_class_key(service_class, day_type, local_hour)
        self.put(pop_code, key, delay_rtt)

    def get_class(
        self,
        pop_code: str,
        service_class: str,
        day_type: str | None = None,
        local_hour: int | None = None,
    ) -> float | None:
        """Read a class-level ground delay RTT."""
        if day_type is not None and local_hour is not None:
            key = encode_class_key(service_class, day_type, local_hour)
            val = self.get(pop_code, key)
            if val is not None:
                return val
        key = encode_class_key(service_class)
        return self.get(pop_code, key)

    def get_class_or_estimate(
        self,
        pop_code: str,
        service_class: str,
        service_estimator: ServiceGroundDelay | None = None,
        local_hour: int = 12,
        day_type: str = "weekday",
    ) -> float:
        """Read class-level cache, falling back to ServiceGroundDelay estimation.

        If an estimate is computed, it is cached for future lookups.

        Raises:
            KeyError: If the pair has no cached class-level entry and
                no ``service_estimator`` is provided. The strict
                measurement contract in this codebase forbids a silent
                zero fallback; callers that want a graceful miss should
                call :meth:`get_class` and handle ``None`` themselves.
        """
        cached = self.get_class(pop_code, service_class, day_type, local_hour)
        if cached is not None:
            return cached
        if service_estimator is None:
            raise KeyError(
                f"no cached class-level RTT and no service_estimator for "
                f"(pop={pop_code!r}, service={service_class!r})"
            )
        rtt = service_estimator.estimate_service(pop_code, service_class, local_hour, day_type)
        self.put_class_time(pop_code, service_class, day_type, local_hour, rtt)
        return rtt
