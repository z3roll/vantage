"""Capacity view and per-epoch usage bookkeeping.

Design contract:

    * **Single source of truth for capacity**: physical cap values live
      on the resource types themselves —
      :class:`vantage.model.ISLEdge` (``capacity_gbps``),
      :class:`vantage.model.ShellConfig` (``feeder_capacity_gbps``),
      :class:`vantage.model.GroundStation` (``max_capacity``).
      Nothing in this module re-stores those values; we only build
      *lookups* over them.

    * **Derived view**: :class:`CapacityView` is a lightweight read-only
      façade that takes references to a :class:`SatelliteState`,
      :class:`ShellConfig`, and a ``gs_id → GroundStation`` map and
      exposes ``isl_cap / sat_feeder_cap / gs_feeder_cap`` in O(1). The
      ISL cap index is pre-built at construction because the data plane
      will query it in a hot loop.

    * **Per-epoch accounting**: :class:`UsageBook` is the mutable
      accounting object in :mod:`vantage.forward`. It holds a reference to a
      :class:`CapacityView` so it can compute utilization / saturation
      / residual headroom without the caller passing caps at every
      site. Instantiate one book per epoch; never share across epochs.

All bandwidth values in Gbps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from vantage.model.ground.infrastructure import GroundStation
from vantage.model.satellite.state import SatelliteState, ShellConfig

__all__ = [
    "CapacityView",
    "UsageBook",
]


# --- CapacityView ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapacityView:
    """Read-only lookup façade over physical capacity fields.

    Instances are cheap to construct — the ISL index is the only
    derived state, and it's built once at construction so ``isl_cap``
    is O(1) inside forward.py's per-flow loop.

    The three data sources (``isl_cap_index`` / ``sat_feeder_gbps`` /
    ``gs_by_id``) are treated as authoritative; this view never stores
    a second copy. Use :meth:`from_snapshot` as the primary constructor
    — it threads together the three underlying records.
    """

    isl_cap_index: Mapping[tuple[int, int], float]  # (min, max) → Gbps
    sat_feeder_gbps: float
    gs_by_id: Mapping[str, GroundStation]

    def __post_init__(self) -> None:
        if not isinstance(self.isl_cap_index, MappingProxyType):
            object.__setattr__(
                self, "isl_cap_index", MappingProxyType(dict(self.isl_cap_index))
            )
        if not isinstance(self.gs_by_id, MappingProxyType):
            object.__setattr__(self, "gs_by_id", MappingProxyType(dict(self.gs_by_id)))

    @classmethod
    def from_snapshot(
        cls,
        sat_state: SatelliteState,
        shell: ShellConfig,
        ground_stations: Mapping[str, GroundStation],
    ) -> CapacityView:
        """Build a view over the given snapshot state.

        The ISL index is derived from ``sat_state.graph.edges`` in a
        direction-agnostic ``(min, max)`` form so both orderings of a
        lookup yield the same cap.
        """
        isl_index: dict[tuple[int, int], float] = {}
        for edge in sat_state.graph.edges:
            key = (edge.sat_a, edge.sat_b) if edge.sat_a <= edge.sat_b else (edge.sat_b, edge.sat_a)
            isl_index[key] = edge.capacity_gbps
        return cls(
            isl_cap_index=MappingProxyType(isl_index),
            sat_feeder_gbps=shell.feeder_capacity_gbps,
            gs_by_id=MappingProxyType(dict(ground_stations)),
        )

    # --- lookups ----------------------------------------------------------

    @staticmethod
    def _isl_key(sat_a: int, sat_b: int) -> tuple[int, int]:
        return (sat_a, sat_b) if sat_a <= sat_b else (sat_b, sat_a)

    def isl_cap(self, sat_a: int, sat_b: int) -> float:
        """Return the capacity of the ISL ``(sat_a, sat_b)``.

        Raises :class:`KeyError` if no such ISL exists in the snapshot —
        unlike a silently-defaulting registry, a missing ISL is a real
        routing bug the caller should surface.
        """
        return self.isl_cap_index[self._isl_key(sat_a, sat_b)]

    def sat_feeder_cap(self, sat_id: int) -> float:
        """Return the per-sat Ka feeder aggregate cap.

        Currently a flat value across all sats in the snapshot (all
        belong to the same shell). Accepts ``sat_id`` anyway so the
        signature is future-proof for multi-shell constellations.
        """
        del sat_id  # uniform for now
        return self.sat_feeder_gbps

    def gs_feeder_cap(self, gs_id: str) -> float:
        """Return the per-GS feeder aggregate cap (``GroundStation.max_capacity``)."""
        return self.gs_by_id[gs_id].max_capacity


# --- UsageBook --------------------------------------------------------------


@dataclass(slots=True)
class UsageBook:
    """Mutable per-epoch running tally of resource consumption.

    Instantiate one per epoch. Never share a book across epochs — the
    dicts are accumulators that assume monotonic growth within a single
    forward pass and are expected to be discarded afterwards.

    The book holds a :class:`CapacityView` reference so caller code can
    ask for utilization / saturation / residual headroom without
    looking up caps itself. This is the *only* place that combines
    "used" with "cap"; the cap side is still owned exclusively by the
    underlying resource types.
    """

    view: CapacityView
    isl_used: dict[tuple[int, int], float] = field(default_factory=dict)
    sat_feeder_used: dict[int, float] = field(default_factory=dict)
    gs_feeder_used: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def isl_key(sat_a: int, sat_b: int) -> tuple[int, int]:
        """Canonical ordering of an ISL pair: always ``(min, max)``."""
        return (sat_a, sat_b) if sat_a <= sat_b else (sat_b, sat_a)

    # --- charge / release --------------------------------------------------
    # Charging never refuses; it just accumulates. A separate fairness pass
    # inspects the book to decide who gets throttled. All charge/release
    # inputs must be non-negative — use the complementary method instead of
    # a signed value, so bookkeeping bugs surface loudly.

    @staticmethod
    def _check_non_negative(gbps: float) -> None:
        if gbps < 0.0:
            raise ValueError(f"gbps must be non-negative, got {gbps}")

    def charge_isl(self, sat_a: int, sat_b: int, gbps: float) -> None:
        self._check_non_negative(gbps)
        key = self.isl_key(sat_a, sat_b)
        self.isl_used[key] = self.isl_used.get(key, 0.0) + gbps

    def charge_sat_feeder(self, sat_id: int, gbps: float) -> None:
        self._check_non_negative(gbps)
        self.sat_feeder_used[sat_id] = self.sat_feeder_used.get(sat_id, 0.0) + gbps

    def charge_gs_feeder(self, gs_id: str, gbps: float) -> None:
        self._check_non_negative(gbps)
        self.gs_feeder_used[gs_id] = self.gs_feeder_used.get(gs_id, 0.0) + gbps

    def release_isl(self, sat_a: int, sat_b: int, gbps: float) -> None:
        self._check_non_negative(gbps)
        key = self.isl_key(sat_a, sat_b)
        self.isl_used[key] = max(0.0, self.isl_used.get(key, 0.0) - gbps)

    def release_sat_feeder(self, sat_id: int, gbps: float) -> None:
        self._check_non_negative(gbps)
        self.sat_feeder_used[sat_id] = max(0.0, self.sat_feeder_used.get(sat_id, 0.0) - gbps)

    def release_gs_feeder(self, gs_id: str, gbps: float) -> None:
        self._check_non_negative(gbps)
        self.gs_feeder_used[gs_id] = max(0.0, self.gs_feeder_used.get(gs_id, 0.0) - gbps)

    # --- introspection -----------------------------------------------------

    def isl_utilization(self, sat_a: int, sat_b: int) -> float:
        used = self.isl_used.get(self.isl_key(sat_a, sat_b), 0.0)
        cap = self.view.isl_cap(sat_a, sat_b)
        return used / cap if cap > 0 else float("inf")

    def sat_feeder_utilization(self, sat_id: int) -> float:
        used = self.sat_feeder_used.get(sat_id, 0.0)
        cap = self.view.sat_feeder_cap(sat_id)
        return used / cap if cap > 0 else float("inf")

    def gs_feeder_utilization(self, gs_id: str) -> float:
        used = self.gs_feeder_used.get(gs_id, 0.0)
        cap = self.view.gs_feeder_cap(gs_id)
        return used / cap if cap > 0 else float("inf")

    def is_isl_saturated(self, sat_a: int, sat_b: int) -> bool:
        return self.isl_utilization(sat_a, sat_b) > 1.0

    def is_sat_feeder_saturated(self, sat_id: int) -> bool:
        return self.sat_feeder_utilization(sat_id) > 1.0

    def is_gs_feeder_saturated(self, gs_id: str) -> bool:
        return self.gs_feeder_utilization(gs_id) > 1.0

    # --- residual capacity -------------------------------------------------
    # Needed by max-min fair-share / Greedy Filling solvers so they
    # can decide how much more traffic a resource can still absorb. All
    # three clamp at zero (over-subscribed resources report no headroom
    # rather than a negative number).

    def remaining_isl(self, sat_a: int, sat_b: int) -> float:
        cap = self.view.isl_cap(sat_a, sat_b)
        used = self.isl_used.get(self.isl_key(sat_a, sat_b), 0.0)
        return max(0.0, cap - used)

    def remaining_sat_feeder(self, sat_id: int) -> float:
        cap = self.view.sat_feeder_cap(sat_id)
        used = self.sat_feeder_used.get(sat_id, 0.0)
        return max(0.0, cap - used)

    def remaining_gs_feeder(self, gs_id: str) -> float:
        cap = self.view.gs_feeder_cap(gs_id)
        used = self.gs_feeder_used.get(gs_id, 0.0)
        return max(0.0, cap - used)
