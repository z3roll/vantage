"""Ground delay measurement: active probing + passive sampling.

PoPs collect ground delay data through two mechanisms:
1. Passive sampling: extract RTT from actual user traffic (zero cost)
2. Active probing: PoP pings destinations on Controller's instruction

Data flows:
  PoPs → (traffic stats) → Controller → (hot dest list) → PoPs → (probe) → Controller
  PoPs → (passive RTT)  → Controller

Both write to GroundKnowledge (which has per-PoP capacity + eviction).
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
    ) -> list[str]: ...


class HotListPolicy:
    """Probe from a fixed list of known important destinations."""

    def __init__(self, hot_destinations: list[str]) -> None:
        self._hot = hot_destinations

    def select(
        self, pop_code: str, candidates: list[str], budget: int
    ) -> list[str]:
        targets = [d for d in self._hot if d in candidates]
        for c in candidates:
            if c not in targets:
                targets.append(c)
            if len(targets) >= budget:
                break
        return targets[:budget]


class TrafficDrivenPolicy:
    """Probe destinations ranked by global traffic frequency.

    Controller aggregates traffic stats from all PoPs, then distributes
    the global hot list. PoPs probe what's globally popular.
    """

    def __init__(self) -> None:
        self._global_freq: dict[str, int] = defaultdict(int)

    def update_stats(self, pop_traffic: dict[str, dict[str, int]]) -> None:
        """Update with per-PoP traffic stats: {pop → {dest → count}}."""
        self._global_freq.clear()
        for per_pop in pop_traffic.values():
            for dest, count in per_pop.items():
                self._global_freq[dest] += count

    def select(
        self, pop_code: str, candidates: list[str], budget: int
    ) -> list[str]:
        ranked = sorted(
            candidates,
            key=lambda d: self._global_freq.get(d, 0),
            reverse=True,
        )
        return ranked[:budget]


# ---------------------------------------------------------------------------
# Probe Manager: orchestrates the full measurement pipeline
# ---------------------------------------------------------------------------


class ProbeManager:
    """Orchestrates ground delay measurement for all PoPs.

    Writes directly to GroundKnowledge, which handles per-PoP capacity
    and eviction internally. No separate PoPCache needed.
    """

    def __init__(
        self,
        ground_truth: GroundDelay,
        knowledge: GroundKnowledge,
        pops: tuple[PoP, ...],
        endpoints: dict[str, Endpoint],
        target_policy: ProbeTargetPolicy,
        probe_budget_per_pop: int = 10,
        passive_sample_rate: float = 1.0,
        probe_interval_s: float = 30.0,
    ) -> None:
        self._ground_truth = ground_truth
        self._knowledge = knowledge
        self._pops = pops
        self._endpoints = endpoints
        self._target_policy = target_policy
        self._probe_budget = probe_budget_per_pop
        self._sample_rate = passive_sample_rate
        self._probe_interval_s = probe_interval_s

        # Per-PoP traffic stats: {pop → {dest → count}}
        self._traffic_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Probe timing
        self._last_probe_time: dict[str, float] = {}

        # All destination names
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
        """Unified measurement entry point. Returns entries written."""
        if method == "passive_sample":
            return self._passive_sample(epoch_result)
        elif method == "active_probe":
            return self._active_probe(current_time_s)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Per-PoP traffic stats for Controller to aggregate."""
        return dict(self._traffic_stats)

    # ── Private ─────────────────────────────────────────────

    def _passive_sample(self, epoch_result: EpochResult | None) -> int:
        if epoch_result is None:
            return 0

        import random
        written = 0
        for flow in epoch_result.flow_outcomes:
            if flow.ground_rtt <= 0:
                continue
            if random.random() > self._sample_rate:
                continue
            self._knowledge.put(flow.pop_code, flow.flow_key.dst, flow.ground_rtt)
            self._traffic_stats[flow.pop_code][flow.flow_key.dst] += 1
            written += 1
        return written

    def _active_probe(self, current_time_s: float) -> int:
        written = 0
        for pop in self._pops:
            last = self._last_probe_time.get(pop.code, -float("inf"))
            if current_time_s - last < self._probe_interval_s:
                continue

            self._last_probe_time[pop.code] = current_time_s

            targets = self._target_policy.select(
                pop.code, self._all_dests, self._probe_budget
            )

            for dest_name in targets:
                # Skip if already known
                if self._knowledge.get(pop.code, dest_name) is not None:
                    continue
                dst_ep = self._endpoints.get(dest_name)
                if dst_ep is None:
                    continue
                delay_rtt = self._ground_truth.estimate(
                    pop.lat_deg, pop.lon_deg,
                    dst_ep.lat_deg, dst_ep.lon_deg,
                ) * 2
                self._knowledge.put(pop.code, dest_name, delay_rtt)
                written += 1
        return written
