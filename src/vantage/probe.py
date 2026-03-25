"""Ground delay measurement: active probing + passive sampling.

PoPs collect ground delay data through two mechanisms:
1. Passive sampling: extract RTT from actual user traffic (zero cost)
2. Active probing: PoP pings destinations on Controller's instruction

Data flows:
  PoPs → (traffic stats) → Controller → (hot dest list) → PoPs → (probe) → Controller
  PoPs → (passive RTT)  → Controller

Both write to GroundKnowledge, which the Controller reads for cost tables.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Protocol

from vantage.domain import EpochResult, Endpoint, PoP
from vantage.world.ground import GroundDelay, GroundKnowledge


# ---------------------------------------------------------------------------
# Probe target policy: decides WHAT each PoP should probe
# ---------------------------------------------------------------------------


class ProbeTargetPolicy(Protocol):
    """Decides which destinations a PoP should actively probe."""

    def select(
        self,
        pop_code: str,
        candidates: list[str],
        budget: int,
    ) -> list[str]:
        """Select up to `budget` destinations to probe from candidates."""
        ...


class HotListPolicy:
    """Probe from a fixed list of known important destinations."""

    def __init__(self, hot_destinations: list[str]) -> None:
        self._hot = hot_destinations

    def select(
        self, pop_code: str, candidates: list[str], budget: int
    ) -> list[str]:
        # Prioritize hot list, then fill with other candidates
        targets = [d for d in self._hot if d in candidates]
        for c in candidates:
            if c not in targets:
                targets.append(c)
            if len(targets) >= budget:
                break
        return targets[:budget]


class TrafficDrivenPolicy:
    """Probe destinations ranked by global traffic frequency.

    Controller aggregates traffic stats from all PoPs, then
    distributes the global hot list. PoPs probe what's globally
    popular, even if they haven't seen local traffic for it.
    """

    def __init__(self) -> None:
        self._global_freq: dict[str, int] = defaultdict(int)

    def update_stats(self, pop_traffic: dict[str, dict[str, int]]) -> None:
        """Update with per-PoP traffic stats: {pop → {dest → count}}.

        Called by Controller after collecting stats from all PoPs.
        """
        self._global_freq.clear()
        for per_pop in pop_traffic.values():
            for dest, count in per_pop.items():
                self._global_freq[dest] += count

    def select(
        self, pop_code: str, candidates: list[str], budget: int
    ) -> list[str]:
        # Rank candidates by global frequency (highest first)
        ranked = sorted(
            candidates,
            key=lambda d: self._global_freq.get(d, 0),
            reverse=True,
        )
        return ranked[:budget]


# ---------------------------------------------------------------------------
# Cache eviction policy: decides WHAT to remove when cache is full
# ---------------------------------------------------------------------------


class CacheEvictionPolicy(Protocol):
    """Decides which entries to evict when a PoP's cache is full."""

    def record_access(self, pop_code: str, dest: str) -> None:
        """Record an access (read or write) for eviction tracking."""
        ...

    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None:
        """Select one entry to evict. Returns dest key, or None if empty."""
        ...


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
        order = self._access_order.get(pop_code, [])
        for dest in order:
            if dest in entries:
                return dest
        # Fallback: any entry
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
    """Evict entry with lowest traffic weight (least demanded destination)."""

    def __init__(self) -> None:
        self._weight: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_access(self, pop_code: str, dest: str) -> None:
        self._weight[pop_code][dest] += 1

    def select_victim(self, pop_code: str, entries: dict[str, float]) -> str | None:
        w = self._weight.get(pop_code, {})
        return min(entries, key=lambda d: w.get(d, 0), default=None)


# ---------------------------------------------------------------------------
# Per-PoP cache with capacity limit
# ---------------------------------------------------------------------------


class PoPCache:
    """Bounded ground delay cache for a single PoP.

    Stores dest → delay_rtt. When full, evicts according to policy.
    """

    def __init__(
        self,
        pop_code: str,
        capacity: int,
        eviction: CacheEvictionPolicy,
    ) -> None:
        self.pop_code = pop_code
        self.capacity = capacity
        self._entries: dict[str, float] = {}
        self._eviction = eviction

    def get(self, dest: str) -> float | None:
        val = self._entries.get(dest)
        if val is not None:
            self._eviction.record_access(self.pop_code, dest)
        return val

    def put(self, dest: str, delay_rtt: float) -> None:
        if dest in self._entries:
            self._entries[dest] = delay_rtt
            self._eviction.record_access(self.pop_code, dest)
            return
        # Need to insert — check capacity
        if len(self._entries) >= self.capacity:
            victim = self._eviction.select_victim(self.pop_code, self._entries)
            if victim is not None:
                del self._entries[victim]
        self._entries[dest] = delay_rtt
        self._eviction.record_access(self.pop_code, dest)

    @property
    def entries(self) -> dict[str, float]:
        return dict(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Probe Manager: orchestrates the full measurement pipeline
# ---------------------------------------------------------------------------


class ProbeManager:
    """Orchestrates ground delay measurement for all PoPs.

    Manages per-PoP caches, executes active probes and passive sampling,
    and syncs results to the centralized GroundKnowledge.

    Flow per epoch:
    1. Passive sampling: extract (pop, dest, rtt) from user traffic results
    2. Collect traffic stats from all PoPs → Controller aggregates
    3. Controller distributes hot destination list
    4. Active probing: each PoP probes destinations it's missing
    5. Sync all PoP caches → GroundKnowledge (for Controller to read)
    """

    def __init__(
        self,
        ground_truth: GroundDelay,
        knowledge: GroundKnowledge,
        pops: tuple[PoP, ...],
        endpoints: dict[str, Endpoint],
        target_policy: ProbeTargetPolicy,
        eviction_policy: CacheEvictionPolicy,
        pop_cache_capacity: int = 1000,
        probe_budget_per_pop: int = 10,
        passive_sample_rate: float = 1.0,
        probe_interval_s: float = 30.0,
    ) -> None:
        self._ground_truth = ground_truth
        self._knowledge = knowledge
        self._pops = pops
        self._endpoints = endpoints
        self._target_policy = target_policy
        self._eviction = eviction_policy
        self._probe_budget = probe_budget_per_pop
        self._sample_rate = passive_sample_rate
        self._probe_interval_s = probe_interval_s

        # Per-PoP caches
        self._caches: dict[str, PoPCache] = {
            p.code: PoPCache(p.code, pop_cache_capacity, eviction_policy)
            for p in pops
        }

        # Per-PoP local traffic stats: {pop → {dest → count}}
        self._traffic_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Probe timing
        self._last_probe_time: dict[str, float] = {}

        # All known destination names
        self._all_dests = [
            e.name for e in endpoints.values()
            if not e.name.startswith("terminal_")
        ]

    def collect(
        self,
        method: Literal["active_probe", "passive_sample"],
        current_time_s: float = 0.0,
        epoch_result: EpochResult | None = None,
    ) -> int:
        """Unified measurement entry point.

        Args:
            method: "active_probe" or "passive_sample"
            current_time_s: simulation time (for probe interval checking)
            epoch_result: required for passive_sample

        Returns:
            Number of cache entries written.
        """
        if method == "passive_sample":
            return self._passive_sample(epoch_result)
        elif method == "active_probe":
            return self._active_probe(current_time_s)
        else:
            raise ValueError(f"Unknown method: {method}")

    def sync_to_knowledge(self) -> None:
        """Push all PoP cache data to centralized GroundKnowledge."""
        for pop_code, cache in self._caches.items():
            for dest, delay in cache.entries.items():
                self._knowledge.put(pop_code, dest, delay)

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Return per-PoP traffic stats for Controller to aggregate."""
        return dict(self._traffic_stats)

    def get_cache_summary(self) -> dict[str, int]:
        """Return {pop_code: num_entries} for monitoring."""
        return {code: cache.size for code, cache in self._caches.items()}

    # ── Private methods ─────────────────────────────────────

    def _passive_sample(self, epoch_result: EpochResult | None) -> int:
        """Extract ground delay measurements from user traffic."""
        if epoch_result is None:
            return 0

        import random
        written = 0
        for flow in epoch_result.flow_outcomes:
            if flow.ground_rtt <= 0:
                continue
            # Sample
            if random.random() > self._sample_rate:
                continue
            # Update PoP's local cache
            cache = self._caches.get(flow.pop_code)
            if cache is not None:
                cache.put(flow.flow_key.dst, flow.ground_rtt)
                written += 1
            # Update traffic stats
            self._traffic_stats[flow.pop_code][flow.flow_key.dst] += 1

        return written

    def _active_probe(self, current_time_s: float) -> int:
        """Execute active probes for PoPs whose interval has elapsed."""
        written = 0
        for pop in self._pops:
            last = self._last_probe_time.get(pop.code, -float("inf"))
            if current_time_s - last < self._probe_interval_s:
                continue

            self._last_probe_time[pop.code] = current_time_s
            cache = self._caches[pop.code]

            # Ask target policy what to probe
            targets = self._target_policy.select(
                pop.code, self._all_dests, self._probe_budget
            )

            for dest_name in targets:
                # Skip if already cached
                if cache.get(dest_name) is not None:
                    continue
                # Simulate measurement using ground truth model
                dst_ep = self._endpoints.get(dest_name)
                if dst_ep is None:
                    continue
                delay_rtt = self._ground_truth.estimate(
                    pop.lat_deg, pop.lon_deg,
                    dst_ep.lat_deg, dst_ep.lon_deg,
                ) * 2  # one-way → RTT
                cache.put(dest_name, delay_rtt)
                written += 1

        return written
