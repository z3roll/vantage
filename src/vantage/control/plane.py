"""Routing-plane domain types.

Historically this module defined a per-satellite FIB (Forwarding
Information Base). The PPT spec calls for per-satellite FIBs, but in
Argus the data plane walks controller-pinned PoP cascades rather than
hop-by-hop FIB entries, so the FIB abstraction ended up unused in the
hot path while `forward.RoutingPlaneForward` kept re-reading
`snapshot.satellite.delay_matrix` / `predecessor_matrix` directly. The
data plane therefore was (de facto) recomputing satellite routing
every realize call.

The types now reflect what the data plane actually consumes from the
controller at refresh time:

    * :class:`CellToPopTable` — global ``cell → ranked PoP cascade``
      map (unchanged). Primary head + cascading fallbacks.
    * :class:`SatPathTable` — per-snapshot ISL shortest-path artifact:
      the one-way ``delay_matrix`` and ``predecessor_matrix`` used for
      ISL-segment RTT lookups and hop reconstruction. These are the
      shortest-path outputs the controller computes every 15 s; the
      data plane reads this artifact rather than the raw
      ``SatelliteState`` matrices.
    * :class:`PopEgressTable` — per-PoP precomputed candidate arrays
      ``(egress_sat_ids, base_cost, gs_ids)`` where ``base_cost`` is
      ``2·downlink + 2·backhaul`` per sat. Built once by the
      controller from the snapshot's ground infrastructure and
      gateway attachments, so ``forward.decide`` does one vectorised
      add over the precomputed arrays instead of walking infrastructure
      + gateway attachments per flow.

:class:`RoutingPlane` bundles all three with the refresh timestamp
and a monotonic version; the simulation loop swaps it atomically when the
cadence is due. :data:`ROUTING_PLANE_REFRESH_S` is the 15 s
sync cadence declared in the PPT (Slide 13).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from vantage.model.coverage import CellId

__all__ = [
    "ROUTING_PLANE_REFRESH_S",
    "CellToPopTable",
    "PopEgressTable",
    "RoutingPlane",
    "SatPathTable",
]


# Control-center → satellite sync cadence in seconds. The routing plane
# is considered stale when ``now - built_at >= ROUTING_PLANE_REFRESH_S``.
# Per the PPT (Slide 13), this is 15 s for Argus.
ROUTING_PLANE_REFRESH_S: float = 15.0


# Frozen empty arrays used when ``PopEgressTable.for_pop`` is queried
# for a PoP the controller did not build candidates for (e.g., zero
# visible GS attachments). Keeping them as module singletons avoids
# reallocating each miss.
_EMPTY_INT32: NDArray[np.int32] = np.empty(0, dtype=np.int32)
_EMPTY_INT32.flags.writeable = False
_EMPTY_FLOAT64: NDArray[np.float64] = np.empty(0, dtype=np.float64)
_EMPTY_FLOAT64.flags.writeable = False


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
class SatPathTable:
    """Per-snapshot ISL shortest-path artifact owned by the controller.

    Wraps the two Dijkstra outputs (``delay_matrix`` and
    ``predecessor_matrix``) behind the routing-plane interface so the
    data plane can look up ISL RTTs and reconstruct hop sequences
    without reaching back into ``SatelliteState``.

    The arrays are treated as immutable — the controller writes them
    once at refresh time and then hands them out via this table. For
    forward compatibility they are exposed as read-only views, which
    lets callers slice and ``tolist()`` without risk of mutating the
    shared routing state.

    Accessors:

        * :meth:`isl_delay` — scalar one-way ISL delay in ms.
        * :meth:`delay_row` — the 1-D column of one-way delays from a
          given ingress to every other sat; consumed by the data
          plane's vectorised per-PoP cost computation.
        * :meth:`pred_row` — the 1-D row of shortest-path predecessors
          from a given ingress; consumed by the data-plane ISL hop
          reconstruction.
    """

    delay_matrix: NDArray[np.float64]
    predecessor_matrix: NDArray[np.int32]
    version: int
    built_at: float

    def __post_init__(self) -> None:
        dm = self.delay_matrix
        pm = self.predecessor_matrix
        if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
            raise ValueError(
                f"SatPathTable.delay_matrix must be square (n, n); got {dm.shape}"
            )
        if pm.shape != dm.shape:
            raise ValueError(
                f"SatPathTable.predecessor_matrix shape {pm.shape} "
                f"!= delay_matrix shape {dm.shape}"
            )
        # Ensure the shared references cannot be mutated through a
        # retained handle. Arrays that are already non-writable are
        # left alone.
        if dm.flags.writeable:
            dm.flags.writeable = False
        if pm.flags.writeable:
            pm.flags.writeable = False

    @property
    def num_sats(self) -> int:
        return int(self.delay_matrix.shape[0])

    def isl_delay(self, sat_a: int, sat_b: int) -> float:
        """One-way ISL propagation delay (ms) between a pair of sats."""
        return float(self.delay_matrix[sat_a, sat_b])

    def delay_row(self, ingress: int) -> NDArray[np.float64]:
        """One-way ISL delay from ``ingress`` to every other sat."""
        return self.delay_matrix[ingress]

    def pred_row(self, ingress: int) -> NDArray[np.int32]:
        """Shortest-path predecessors rooted at ``ingress``."""
        return self.predecessor_matrix[ingress]


@dataclass(frozen=True, slots=True)
class PopEgressTable:
    """Per-PoP controller-built egress candidate table.

    For each PoP, stores a flattened table of viable (sat, gs)
    downlink pairs: for each candidate, the egress sat id, the
    ``base_cost`` contributed by the ground segment (=
    ``2·downlink_rtt + 2·backhaul_rtt``, RTT already doubled), and
    the GS id hosting the antenna. Built once per refresh from
    ``snapshot.infra.pop_gs_edges`` + ``snapshot.satellite.gateway_attachments``.

    The data plane adds ``delay_row[egress_ids] * 2`` from the
    accompanying :class:`SatPathTable` to get the full sat-segment
    RTT from a specific ingress — one numpy fancy-index + add
    replaces the per-flow infrastructure walk that used to happen
    inside ``RoutingPlaneForward``.

    PoPs not present in :attr:`candidates` have no viable egress in
    this plane — :meth:`for_pop` returns empty arrays so the data
    plane can skip them without branching on ``None``.
    """

    candidates: Mapping[
        str, tuple[NDArray[np.int32], NDArray[np.float64], tuple[str, ...]]
    ]
    version: int
    built_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, MappingProxyType):
            object.__setattr__(
                self, "candidates", MappingProxyType(dict(self.candidates)),
            )
        # Freeze every per-PoP array so downstream callers cannot
        # mutate the controller's artifact by side effect.
        for egress_ids, base_cost, _gs_ids in self.candidates.values():
            if egress_ids.flags.writeable:
                egress_ids.flags.writeable = False
            if base_cost.flags.writeable:
                base_cost.flags.writeable = False

    def for_pop(
        self, pop_code: str,
    ) -> tuple[NDArray[np.int32], NDArray[np.float64], tuple[str, ...]]:
        """Return ``(egress_sat_ids, base_cost, gs_ids)`` for ``pop_code``.

        Empty arrays + empty tuple when the PoP has no viable egress
        candidates in this plane (no attached GS, or no visible sat
        on any attached GS at refresh time).
        """
        return self.candidates.get(pop_code, (_EMPTY_INT32, _EMPTY_FLOAT64, ()))

    def has_pop(self, pop_code: str) -> bool:
        return pop_code in self.candidates


@dataclass(frozen=True, slots=True)
class RoutingPlane:
    """The full per-epoch routing state visible to the data plane.

    Bundled as one immutable record so the engine can swap it
    atomically when the refresh cadence is due. All three artifacts
    (``cell_to_pop``, ``sat_paths``, ``pop_egress``) share the same
    ``version`` and ``built_at`` so the data plane can reason about
    freshness in one place.
    """

    cell_to_pop: CellToPopTable
    sat_paths: SatPathTable
    pop_egress: PopEgressTable
    version: int
    built_at: float  # simulation time (s) of the most recent refresh

    def is_stale(self, now_s: float, cadence_s: float = ROUTING_PLANE_REFRESH_S) -> bool:
        """Return ``True`` if the plane needs a refresh at simulation time ``now_s``."""
        return (now_s - self.built_at) >= cadence_s
