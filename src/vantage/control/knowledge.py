"""Per-PoP learned ground-RTT statistics (epoch-aggregated).

Pre-refactor this module held a scalar per ``(pop, dest)``. The
current design stores a tiny per-epoch aggregate summary:

    * ``mu_ms``       — EWMA across epochs of the per-epoch **mean**
                        realised RTT.
    * ``dev_ms``      — EWMA across epochs of the per-epoch **within-
                        epoch spread** (population stddev of every
                        flow's realised RTT inside that epoch).
    * ``epoch_count`` — number of *epochs* absorbed, not the number
                        of flows. One :meth:`update` call per
                        ``(pop, dest, epoch)`` bumps this by one
                        regardless of how many flows the feedback
                        layer aggregated into the call.
    * ``last_epoch``  — epoch of the most recent observation, ``-1``
                        for prior-only entries that have never been
                        updated by feedback.

The planner consumes this via :meth:`score`, which composes mean,
deviation, and staleness into a single comparable number:

    ``score = mu_ms + λ · dev_ms + stale_per_epoch · max(0, now - last_epoch)``

Callers that just want the plain RTT (to fill a cascade entry,
render a chart, etc.) use :meth:`get_mean`, which returns ``mu_ms``
or ``None`` on a miss. The legacy scalar API (``put`` / ``get``)
stays as thin wrappers so forward/controller code can migrate
incrementally — ``put(pop, dest, rtt)`` records an epoch-mean of
``rtt`` with dev ``0`` at ``last_epoch = -1``.

Per-PoP capacity limits + eviction policies are preserved: each PoP
holds a bounded ``dest → GroundStat`` cache, and the policies only
ever iterate keys so their implementations didn't need changes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from vantage.model.ground.latency import GroundDelay


DELIMITER = ":"
CLASS_PREFIX = "class"
SERVICE_PREFIX = "service"


def encode_class_key(
    service_class: str,
    day_type: str | None = None,
    hour: int | None = None,
) -> str:
    if day_type is None or hour is None:
        return f"{CLASS_PREFIX}{DELIMITER}{service_class}"
    return f"{CLASS_PREFIX}{DELIMITER}{service_class}{DELIMITER}{day_type}{DELIMITER}{hour}"


def encode_service_key(
    service_name: str,
    day_type: str | None = None,
    hour: int | None = None,
) -> str:
    if day_type is None or hour is None:
        return f"{SERVICE_PREFIX}{DELIMITER}{service_name}"
    return f"{SERVICE_PREFIX}{DELIMITER}{service_name}{DELIMITER}{day_type}{DELIMITER}{hour}"


def decode_key(key: str) -> tuple[str, str, str | None, int | None]:
    if DELIMITER in key:
        parts = key.split(DELIMITER)
        tier = parts[0]
        if tier in (CLASS_PREFIX, SERVICE_PREFIX):
            if len(parts) == 2:
                return tier, parts[1], None, None
            if len(parts) == 4:
                return tier, parts[1], parts[2], int(parts[3])
    return "legacy", key, None, None


def is_class_key(key: str) -> bool:
    return key.startswith(CLASS_PREFIX + DELIMITER)


def is_service_key(key: str) -> bool:
    return key.startswith(SERVICE_PREFIX + DELIMITER)


@dataclass(slots=True)
class GroundStat:
    """Lightweight per-(pop, dest) learned summary (epoch-level).

    Fields are mutated in place by :meth:`GroundKnowledge.update` — a
    flat mutable dataclass is the cheapest storage shape at the
    hundreds-of-thousands-of-pairs scale we hit in production runs.

    ``epoch_count`` tracks the number of *epochs* absorbed, not the
    number of flows. One :meth:`GroundKnowledge.update` call lands
    per ``(pop, dest, epoch)`` tuple, so a route that observes 10 k
    flows in one epoch bumps this by 1 — identical to a route that
    saw a single flow. This keeps scoring decisions immune to demand
    skew (noisy PoPs with huge fan-out would otherwise overwhelm
    thin PoPs).
    """

    mu_ms: float
    dev_ms: float
    epoch_count: int
    last_epoch: int

    def as_tuple(self) -> tuple[float, float, int, int]:
        return (self.mu_ms, self.dev_ms, self.epoch_count, self.last_epoch)


class CacheEvictionPolicy(Protocol):
    """Decides which entry to evict when a PoP's cache is full."""

    def record_access(self, pop_code: str, dest: str) -> None: ...
    def select_victim(
        self, pop_code: str, entries: Mapping[str, Any],
    ) -> str | None: ...


class LRUEviction:
    """Evict least recently used entry."""

    def __init__(self) -> None:
        self._access_order: dict[str, list[str]] = defaultdict(list)

    def record_access(self, pop_code: str, dest: str) -> None:
        order = self._access_order[pop_code]
        if dest in order:
            order.remove(dest)
        order.append(dest)

    def select_victim(
        self, pop_code: str, entries: Mapping[str, Any],
    ) -> str | None:
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

    def select_victim(
        self, pop_code: str, entries: Mapping[str, Any],
    ) -> str | None:
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

    def select_victim(
        self, pop_code: str, entries: Mapping[str, Any],
    ) -> str | None:
        times = self._insert_time.get(pop_code, {})
        return min(entries, key=lambda d: times.get(d, 0.0), default=None)


class TrafficWeightedEviction:
    """Evict entry with lowest traffic weight."""

    def __init__(self) -> None:
        self._weight: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_access(self, pop_code: str, dest: str) -> None:
        self._weight[pop_code][dest] += 1

    def select_victim(
        self, pop_code: str, entries: Mapping[str, Any],
    ) -> str | None:
        w = self._weight.get(pop_code, {})
        return min(entries, key=lambda d: w.get(d, 0), default=None)


class GroundKnowledge:
    """Per-PoP learned ground-RTT statistics with optional capacity cap.

    Holds one :class:`GroundStat` per ``(pop, dest)`` pair. The
    update rule is EWMA on both the mean and the mean absolute
    deviation:

        ``new_mu  = (1 - α)·old_mu + α·sample``
        ``new_dev = (1 - α)·old_dev + α·|sample - old_mu|``

    which tracks a noisy signal without storing any history. ``α``
    is the constructor's ``ewma_alpha``.

    Capacity and eviction are per-PoP — setting ``pop_capacity > 0``
    bounds each PoP's dest-dict; when full, the injected
    :class:`CacheEvictionPolicy` chooses a victim to drop.
    """

    def __init__(
        self,
        estimator: GroundDelay | None = None,
        pop_capacity: int = 0,
        eviction: CacheEvictionPolicy | None = None,
        ewma_alpha: float = 0.3,
        ttl_s: float = 0.0,
    ) -> None:
        self._per_pop: dict[str, dict[str, GroundStat]] = defaultdict(dict)
        self._timestamps: dict[str, dict[str, float]] = defaultdict(dict)
        self._known_dests: set[str] = set()
        self._estimator = estimator
        self._pop_capacity = pop_capacity  # 0 = unlimited
        self._eviction = eviction
        self._ewma_alpha = float(ewma_alpha)
        self._ttl_s = ttl_s
        self._clock: float = 0.0  # seconds; maintained by the engine

    def set_clock(self, t: float) -> None:
        """Record current simulation time (seconds). Used only for TTL."""
        self._clock = t

    @property
    def estimator(self) -> GroundDelay | None:
        return self._estimator

    @property
    def ewma_alpha(self) -> float:
        return self._ewma_alpha

    # ── core stats API ────────────────────────────────────────────

    def update(
        self,
        pop_code: str,
        dest: str,
        epoch_mean_rtt: float,
        epoch_dev_rtt: float,
        epoch: int,
    ) -> GroundStat:
        """Absorb one epoch's aggregate into the stats for ``(pop, dest)``.

        Parameters
        ----------
        epoch_mean_rtt:
            Mean RTT (ms) across every realised flow on this
            ``(pop, dest)`` during ``epoch``. Feedback computes this
            before calling ``update`` — knowledge never sees raw
            per-flow values.
        epoch_dev_rtt:
            Within-epoch spread (population stddev, ms) across the
            same flows. Zero when only one flow landed on the pair
            this epoch.
        epoch:
            Epoch identifier. Callers without a real epoch (bootstrap
            writers) should use :meth:`put` instead, which sets
            ``last_epoch = -1``.

        Returns the updated :class:`GroundStat`. Exactly one call lands
        per ``(pop, dest, epoch)`` — ``epoch_count`` increments by one
        regardless of the underlying flow count.
        """
        pop_cache = self._per_pop[pop_code]
        stat = pop_cache.get(dest)
        if stat is None:
            # Capacity check before inserting a fresh entry.
            if (
                self._pop_capacity > 0
                and len(pop_cache) >= self._pop_capacity
                and self._eviction is not None
            ):
                evictable = {
                    k: v for k, v in pop_cache.items() if not is_class_key(k)
                }
                if evictable:
                    victim = self._eviction.select_victim(pop_code, evictable)
                    if victim is not None:
                        del pop_cache[victim]
                        self._timestamps[pop_code].pop(victim, None)
            stat = GroundStat(
                mu_ms=float(epoch_mean_rtt),
                dev_ms=float(epoch_dev_rtt),
                epoch_count=1,
                last_epoch=int(epoch),
            )
            pop_cache[dest] = stat
            self._known_dests.add(dest)
        else:
            a = self._ewma_alpha
            prev_mu = stat.mu_ms
            new_mean = float(epoch_mean_rtt)
            new_dev = float(epoch_dev_rtt)
            # On abrupt jumps (>3× or <⅓) the previous mean is almost
            # certainly stale — reset to the new epoch aggregate so
            # the planner sees the regime change immediately instead
            # of crawling toward it over many EWMA steps.
            if prev_mu > 0 and (
                new_mean / prev_mu > 3.0 or prev_mu / new_mean > 3.0
            ):
                stat.mu_ms = new_mean
                stat.dev_ms = new_dev
            else:
                stat.mu_ms = (1.0 - a) * prev_mu + a * new_mean
                stat.dev_ms = (1.0 - a) * stat.dev_ms + a * new_dev
            stat.epoch_count += 1
            stat.last_epoch = int(epoch)
        self._timestamps[pop_code][dest] = self._clock
        if self._eviction is not None:
            self._eviction.record_access(pop_code, dest)
        return stat

    def stat(self, pop_code: str, dest: str) -> GroundStat | None:
        """Return the raw :class:`GroundStat` for ``(pop, dest)``.

        ``None`` on a cache miss or when the entry has expired under
        ``ttl_s``. Callers in hot-path code should prefer
        :meth:`get_mean` (returns a bare float) to avoid the object
        allocation and attribute access.
        """
        pop_cache = self._per_pop.get(pop_code)
        if pop_cache is None:
            return None
        stat = pop_cache.get(dest)
        if stat is None:
            return None
        if self._ttl_s > 0:
            ts = self._timestamps.get(pop_code, {}).get(dest, 0.0)
            if self._clock - ts > self._ttl_s:
                return None
        if self._eviction is not None:
            self._eviction.record_access(pop_code, dest)
        return stat

    def get_mean(self, pop_code: str, dest: str) -> float | None:
        """Return the current EWMA-smoothed mean RTT (ms), or ``None``."""
        stat = self.stat(pop_code, dest)
        return stat.mu_ms if stat is not None else None

    def score(
        self,
        pop_code: str,
        dest: str,
        *,
        current_epoch: int,
        lambda_dev: float = 1.0,
        stale_per_epoch_ms: float = 0.0,
    ) -> float | None:
        """Return ``mu + λ·dev + stale_penalty`` or ``None`` on miss.

        ``lambda_dev`` controls how strongly deviation penalises a
        noisy pair. ``stale_per_epoch_ms`` adds a linear penalty per
        epoch since the last observation; with ``last_epoch == -1``
        (prior-only, never observed) the penalty is treated as zero —
        a prior is not "stale" in the same sense as a measurement
        that rotted.
        """
        stat = self.stat(pop_code, dest)
        if stat is None:
            return None
        penalty = 0.0
        if stale_per_epoch_ms > 0.0 and stat.last_epoch >= 0:
            staleness = max(0, int(current_epoch) - stat.last_epoch)
            penalty = stale_per_epoch_ms * staleness
        return stat.mu_ms + lambda_dev * stat.dev_ms + penalty

    # ── legacy scalar API (bootstrap / callers that want a plain RTT) ──

    def put(self, pop_code: str, dest: str, delay_rtt: float) -> None:
        """Legacy scalar writer; shim over :meth:`update`.

        Used by bootstrap paths that don't yet know a real epoch
        (e.g. class-level defaults). Entries written through ``put``
        record an epoch mean of ``delay_rtt`` with within-epoch dev
        ``0.0`` at ``last_epoch = -1``, so :meth:`score`'s staleness
        penalty leaves them alone.
        """
        self.update(pop_code, dest, float(delay_rtt), 0.0, epoch=-1)

    def get(self, pop_code: str, dest: str) -> float | None:
        """Legacy scalar reader; returns the current mean or ``None``."""
        return self.get_mean(pop_code, dest)

    def get_or_estimate(self, pop_code: str, dest: str) -> float:
        """Return the learned mean, falling back to the estimator × 2.

        The estimator returns a one-way RTT; we double it here so the
        caller always gets a round-trip value, matching the format of
        the stored ``mu_ms``. Raises :class:`KeyError` when neither a
        stat nor an estimator is available — identical to the
        pre-refactor contract so existing callers don't need changes.
        """
        mean = self.get_mean(pop_code, dest)
        if mean is not None:
            return mean
        if self._estimator is None:
            raise KeyError(
                f"no cached RTT and no estimator for "
                f"(pop={pop_code!r}, dest={dest!r})"
            )
        return self._estimator.estimate(pop_code, dest) * 2

    def has(self, dest: str) -> bool:
        return dest in self._known_dests

    def all_entries(self) -> dict[tuple[str, str], float]:
        """Flatten to ``(pop, dest) → mu_ms`` for cache dumps / UI."""
        result: dict[tuple[str, str], float] = {}
        for pop_code, pop_cache in self._per_pop.items():
            for dest, stat in pop_cache.items():
                result[(pop_code, dest)] = stat.mu_ms
        return result

    def all_stats(self) -> dict[tuple[str, str], GroundStat]:
        """Same as :meth:`all_entries` but exposes the full stats."""
        out: dict[tuple[str, str], GroundStat] = {}
        for pop_code, pop_cache in self._per_pop.items():
            for dest, stat in pop_cache.items():
                out[(pop_code, dest)] = stat
        return out

    def pop_entries(self, pop_code: str) -> dict[str, float]:
        return {dest: stat.mu_ms for dest, stat in self._per_pop.get(pop_code, {}).items()}

    def pop_size(self, pop_code: str) -> int:
        return len(self._per_pop.get(pop_code, {}))

    def total_size(self) -> int:
        return sum(len(v) for v in self._per_pop.values())

    def best_pop_for(self, dest: str) -> tuple[str, float] | None:
        best_pop: str | None = None
        best_delay = float("inf")
        for pop_code, pop_cache in self._per_pop.items():
            stat = pop_cache.get(dest)
            if stat is not None and stat.mu_ms < best_delay:
                best_delay = stat.mu_ms
                best_pop = pop_code
        if best_pop is None:
            return None
        return best_pop, best_delay

    # ── Service-class cache methods ──────────────────────

    def put_class(self, pop_code: str, service_class: str, delay_rtt: float) -> None:
        self.put(pop_code, encode_class_key(service_class), delay_rtt)

    def put_class_time(
        self,
        pop_code: str,
        service_class: str,
        day_type: str,
        local_hour: int,
        delay_rtt: float,
    ) -> None:
        self.put(
            pop_code,
            encode_class_key(service_class, day_type, local_hour),
            delay_rtt,
        )

    def get_class(
        self,
        pop_code: str,
        service_class: str,
        day_type: str | None = None,
        local_hour: int | None = None,
    ) -> float | None:
        if day_type is not None and local_hour is not None:
            key = encode_class_key(service_class, day_type, local_hour)
            val = self.get(pop_code, key)
            if val is not None:
                return val
        return self.get(pop_code, encode_class_key(service_class))

    def get_class_or_raise(
        self,
        pop_code: str,
        service_class: str,
        day_type: str = "weekday",
        local_hour: int = 12,
    ) -> float:
        cached = self.get_class(pop_code, service_class, day_type, local_hour)
        if cached is not None:
            return cached
        raise KeyError(
            f"no cached class-level RTT for "
            f"(pop={pop_code!r}, service={service_class!r})"
        )
