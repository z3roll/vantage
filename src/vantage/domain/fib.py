"""Forwarding-plane domain types (FIB = Forwarding Information Base).

These types describe the data-plane lookup structures that the PPT's
control center pushes to every satellite every 15 seconds. They are
*consumers* of the routing computation owned by
:mod:`vantage.world.satellite.routing` — the Dijkstra step there
produces the raw shortest-path predecessors; a FIB builder (to be
added under ``world/satellite/``) reuses that output to populate each
satellite's :class:`SatelliteFIB`.

Two tables per refresh:

    * :class:`CellToPopTable` — a *global* ``cell_id → pop_code`` map.
      For the nearest-PoP baseline this is purely geographic and
      changes only when the PoP set changes.
    * :class:`SatelliteFIB` — a *per-satellite* ``pop_code → FIBEntry``
      table. Each entry either terminates at a local GS (``EGRESS``)
      or forwards to an ISL neighbor (``FORWARD``).

:class:`RoutingPlane` bundles both tables with the timestamp of the
last refresh so the engine can enforce the 15 s cadence declared in
:data:`ROUTING_PLANE_REFRESH_S`.

The file is named ``fib.py`` rather than ``routing.py`` to avoid a
name collision with :mod:`vantage.world.satellite.routing`, which
owns the Dijkstra computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from vantage.domain.cell import CellId

__all__ = [
    "ROUTING_PLANE_REFRESH_S",
    "CellToPopTable",
    "FIBEntry",
    "FIBEntryKind",
    "RoutingPlane",
    "SatelliteFIB",
]


# Control-center → satellite sync cadence in seconds. The routing plane
# is considered stale when ``now - built_at >= ROUTING_PLANE_REFRESH_S``.
# Per the PPT (Slide 13), this is 15 s for Argus.
ROUTING_PLANE_REFRESH_S: float = 15.0


class FIBEntryKind(Enum):
    """Tag for :class:`FIBEntry`."""

    EGRESS = "egress"
    """This satellite egresses traffic locally to a ground station."""

    FORWARD = "forward"
    """This satellite forwards traffic over ISL to a neighbor."""


@dataclass(frozen=True, slots=True)
class FIBEntry:
    """One entry in a :class:`SatelliteFIB`.

    Semantics depend on :attr:`kind`:

    * ``EGRESS``:  :attr:`target` is a :class:`str` ground-station id.
      :attr:`cost_ms` is the downlink + backhaul cost from this sat
      all the way to the PoP.
    * ``FORWARD``: :attr:`target` is an ``int`` next-hop satellite id.
      :attr:`cost_ms` is the remaining shortest-path cost from this
      sat to the target PoP (inclusive of the next ISL hop).

    Using an ``int | str`` target lets the two variants share the same
    storage shape; ``__post_init__`` enforces the kind→target type
    invariant so direct construction can't build a malformed entry.
    Consumers should branch on :attr:`kind` (or the :attr:`is_egress` /
    :attr:`is_forward` predicates) before interpreting :attr:`target`.
    """

    kind: FIBEntryKind
    target: int | str
    cost_ms: float

    def __post_init__(self) -> None:
        if self.kind is FIBEntryKind.EGRESS and not isinstance(self.target, str):
            raise TypeError(
                f"EGRESS FIBEntry must have a str target (gs_id), got {type(self.target).__name__}"
            )
        if self.kind is FIBEntryKind.FORWARD and not isinstance(self.target, int):
            raise TypeError(
                f"FORWARD FIBEntry must have an int target (sat_id), "
                f"got {type(self.target).__name__}"
            )

    @classmethod
    def egress(cls, gs_id: str, cost_ms: float) -> FIBEntry:
        """Build an EGRESS entry terminating at ``gs_id``."""
        return cls(kind=FIBEntryKind.EGRESS, target=gs_id, cost_ms=cost_ms)

    @classmethod
    def forward(cls, next_hop_sat: int, cost_ms: float) -> FIBEntry:
        """Build a FORWARD entry handing off to ``next_hop_sat``."""
        return cls(kind=FIBEntryKind.FORWARD, target=next_hop_sat, cost_ms=cost_ms)

    @property
    def is_egress(self) -> bool:
        return self.kind is FIBEntryKind.EGRESS

    @property
    def is_forward(self) -> bool:
        return self.kind is FIBEntryKind.FORWARD

    @property
    def next_hop_sat(self) -> int:
        """Return the next-hop sat id; raises if this is not a FORWARD entry."""
        if self.kind is not FIBEntryKind.FORWARD:
            raise ValueError(f"FIBEntry is {self.kind}, not FORWARD")
        # Type invariant guaranteed by __post_init__.
        return self.target  # type: ignore[return-value]

    @property
    def egress_gs(self) -> str:
        """Return the egress GS id; raises if this is not an EGRESS entry."""
        if self.kind is not FIBEntryKind.EGRESS:
            raise ValueError(f"FIBEntry is {self.kind}, not EGRESS")
        # Type invariant guaranteed by __post_init__.
        return self.target  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SatelliteFIB:
    """One satellite's full forwarding table.

    ``fib`` maps a :class:`str` PoP code to the :class:`FIBEntry` to use
    for traffic destined to that PoP. Lookups that miss the map mean
    "no route to that PoP from here" — callers should handle the
    :class:`KeyError` explicitly. ``__post_init__`` freezes the mapping
    into a :class:`MappingProxyType` so callers cannot mutate it through
    a retained dict reference after construction.
    """

    sat_id: int
    fib: Mapping[str, FIBEntry]
    version: int
    built_at: float  # simulation time (s) when this FIB was computed

    def __post_init__(self) -> None:
        if not isinstance(self.fib, MappingProxyType):
            object.__setattr__(self, "fib", MappingProxyType(dict(self.fib)))

    def route(self, pop_code: str) -> FIBEntry:
        """Return the :class:`FIBEntry` for ``pop_code`` or raise ``KeyError``."""
        return self.fib[pop_code]


@dataclass(frozen=True, slots=True)
class CellToPopTable:
    """Cell-to-PoP assignment pushed by the controller.

    ``mapping``: default ``cell → ranked PoP tuple`` (baseline /
    fallback). The tuple is ordered by increasing preference cost
    (geographic distance for the nearest-PoP baseline, E2E RTT for
    capacity-aware policies). ``mapping[cell][0]`` is the controller's
    primary pick; the data plane walks the rest as cascading fallbacks
    when upstream sat feeders saturate.

    ``per_dest``: optional ``(cell, dest) → ranked PoP tuple``
    overrides for performance-aware routing — different destinations
    may rank PoPs differently from the same cell. Same head-of-tuple
    semantics as ``mapping``.
    """

    mapping: Mapping[CellId, tuple[str, ...]]
    version: int
    built_at: float
    per_dest: Mapping[tuple[CellId, str], tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, MappingProxyType):
            object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))
        if not isinstance(self.per_dest, MappingProxyType):
            object.__setattr__(self, "per_dest", MappingProxyType(dict(self.per_dest)))

    def pop_of(self, cell_id: CellId, dest: str | None = None) -> str:
        """Primary PoP for (cell, dest) — head of the ranked tuple."""
        return self.pops_of(cell_id, dest)[0]

    def pops_of(
        self, cell_id: CellId, dest: str | None = None,
    ) -> tuple[str, ...]:
        """Full ranked PoP tuple for (cell, dest).

        Falls back to the ``mapping`` entry if no per-dest override is
        present. Raises ``KeyError`` if the cell itself is unmapped.
        """
        if dest is not None:
            override = self.per_dest.get((cell_id, dest))
            if override is not None:
                return override
        return self.mapping[cell_id]


@dataclass(frozen=True, slots=True)
class RoutingPlane:
    """The full per-epoch routing state visible to the data plane.

    Bundled as one immutable record so the engine can swap it atomically
    when the refresh cadence is due. Construction freezes ``sat_fibs``
    into :class:`MappingProxyType`.
    """

    cell_to_pop: CellToPopTable
    sat_fibs: Mapping[int, SatelliteFIB]
    version: int
    built_at: float  # simulation time (s) of the most recent refresh

    def __post_init__(self) -> None:
        if not isinstance(self.sat_fibs, MappingProxyType):
            object.__setattr__(self, "sat_fibs", MappingProxyType(dict(self.sat_fibs)))

    def fib_of(self, sat_id: int) -> SatelliteFIB:
        """Return the :class:`SatelliteFIB` for ``sat_id`` or raise ``KeyError``."""
        return self.sat_fibs[sat_id]

    def is_stale(self, now_s: float, cadence_s: float = ROUTING_PLANE_REFRESH_S) -> bool:
        """Return ``True`` if the plane needs a refresh at simulation time ``now_s``."""
        return (now_s - self.built_at) >= cadence_s
