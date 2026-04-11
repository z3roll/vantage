"""Policy-agnostic helpers for assembling a :class:`RoutingPlane`.

Two pure functions, both reused across baseline and future policies:

    * :func:`build_satellite_fibs` — turns a :class:`PerSatRouting`
      (which already encodes "for each ingress sat, which egress sat
      and GS minimizes cost to each PoP") into a per-satellite
      :class:`SatelliteFIB` by looking up the first ISL hop through
      :func:`vantage.world.satellite.routing.first_hop_on_path`.
    * :func:`build_cell_to_pop_nearest` — geographic argmin from every
      :class:`Cell` center to the closest :class:`PoP`. This is the
      baseline ``cell_to_pop`` assignment; capacity-aware policies will
      replace it with a TE solver later.

Both functions are stateless and take explicit inputs, so they can be
unit-tested without spinning up a full snapshot.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from types import MappingProxyType

import numpy as np

from vantage.control.policy.common.sat_cost import (
    PerSatRouting,
    precompute_per_sat_routing,
)
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    FIBEntry,
    NetworkSnapshot,
    PoP,
    RoutingPlane,
    SatelliteFIB,
)
from vantage.world.satellite.routing import first_hop_on_path

__all__ = [
    "build_cell_to_pop_nearest",
    "build_routing_plane_nearest_pop",
    "build_satellite_fibs",
]

_log = logging.getLogger(__name__)


def build_satellite_fibs(
    snapshot: NetworkSnapshot,
    per_sat_routing: PerSatRouting,
    *,
    version: int = 0,
) -> dict[int, SatelliteFIB]:
    """Assemble one :class:`SatelliteFIB` per satellite.

    For each ``(ingress_sat, pop_code)`` where the PoP is reachable,
    emit a :class:`FIBEntry`:

        * **EGRESS** if ``chosen_egress_sat == ingress_sat`` — the
          local satellite is itself the egress point (bent-pipe case,
          or we already are the terminating hop). The entry target is
          the chosen GS id and the cost is the full RTT.
        * **FORWARD** otherwise — walk the predecessor matrix once via
          :func:`first_hop_on_path` to find the first ISL neighbor on
          the shortest path to the egress, then record that neighbor.

    Unreachable PoPs are silently skipped (no FIB entry) — callers
    handle the missing key as "no route from here".

    Args:
        snapshot: Current network state (for ``num_sats``, predecessor
            matrix, and time stamp).
        per_sat_routing: Output of
            :func:`vantage.control.policy.common.sat_cost.precompute_per_sat_routing`
            for the same snapshot.
        version: Monotonic version tag attached to every produced FIB
            so downstream consumers can tell refreshes apart.

    Returns:
        Mapping from satellite id to its :class:`SatelliteFIB`.
    """
    sat = snapshot.satellite
    predecessor = sat.predecessor_matrix
    built_at = snapshot.time_s
    pop_codes = tuple(per_sat_routing.cost_ms.keys())

    fibs: dict[int, SatelliteFIB] = {}
    for ingress in range(sat.num_sats):
        entries: dict[str, FIBEntry] = {}
        for pop_code in pop_codes:
            if not per_sat_routing.is_reachable(pop_code, ingress):
                continue
            egress = int(per_sat_routing.egress_sat[pop_code][ingress])
            cost = float(per_sat_routing.cost_ms[pop_code][ingress])
            gs_id = per_sat_routing.chosen_gs(pop_code, ingress)
            # ``is_reachable`` already guarantees a concrete egress/gs. If
            # the invariant is broken (e.g. gs_index contains -1 but
            # egress_sat does not), surface it as a hard error rather than
            # relying on ``assert`` — ``python -O`` would strip the assert.
            if gs_id is None:
                raise ValueError(
                    f"PerSatRouting inconsistency: is_reachable={True} for "
                    f"({pop_code}, ingress={ingress}) but chosen_gs returned None"
                )

            if egress == ingress:
                entries[pop_code] = FIBEntry.egress(gs_id=gs_id, cost_ms=cost)
            else:
                next_hop = first_hop_on_path(predecessor, ingress, egress)
                if next_hop < 0 or next_hop == ingress:
                    # Shouldn't happen — is_reachable said there is a path
                    # and first_hop_on_path should always advance. Prefer a
                    # partial FIB over a crashed controller, but log the
                    # anomaly so it surfaces in production runs instead of
                    # silently dropping routes.
                    _log.warning(
                        "build_satellite_fibs: dropping FIB entry for "
                        "(ingress=%d, pop=%s): first_hop_on_path returned "
                        "%d (egress=%d). PerSatRouting said reachable but "
                        "the predecessor matrix disagrees.",
                        ingress, pop_code, next_hop, egress,
                    )
                    continue
                entries[pop_code] = FIBEntry.forward(
                    next_hop_sat=next_hop, cost_ms=cost
                )

        fibs[ingress] = SatelliteFIB(
            sat_id=ingress,
            fib=MappingProxyType(entries),
            version=version,
            built_at=built_at,
        )
    return fibs


def build_cell_to_pop_nearest(
    cell_grid: CellGrid,
    pops: Iterable[PoP],
    *,
    built_at: float,
    version: int = 0,
) -> CellToPopTable:
    """Assign every cell to its geographically-nearest PoP.

    Vectorized haversine over the full ``(|cells|, |pops|)`` cartesian
    product, then ``argmin`` along the PoP axis. Baseline scale
    (~10 k cells × ~50 PoPs) is ~10 ms end-to-end — cheap enough to
    re-run per routing-plane refresh should the PoP set ever become
    dynamic, though in practice the baseline plan calls this once at
    setup time.

    The output assignment depends only on geography, so it never
    changes between refreshes for a fixed ``(cell_grid, pops)`` pair.
    Capacity-aware policies (Progressive Filling etc.) will replace
    this with a TE solver in a later phase.

    Args:
        cell_grid: The set of cells in play (from endpoints).
        pops: All PoPs eligible as destinations.
        built_at: Simulation time stamp to record on the produced table.
        version: Monotonic version tag.

    Raises:
        ValueError: If ``pops`` is empty (no PoPs means no assignment
            is meaningful and callers should surface the mistake).
    """
    pop_list = tuple(pops)
    if not pop_list:
        raise ValueError("build_cell_to_pop_nearest: pops must be non-empty")

    # Materialize parallel cell / pop coordinate arrays once. dict.values()
    # is iteration-order-stable in CPython ≥ 3.7, so ``cell_ids[i]`` aligns
    # with ``cell_lats[i]`` / ``cell_lons[i]`` without an explicit zip.
    cell_ids = tuple(cell_grid.cells.keys())
    cell_coords = np.fromiter(
        (coord for cell in cell_grid.cells.values() for coord in (cell.lat_deg, cell.lon_deg)),
        dtype=np.float64,
        count=len(cell_ids) * 2,
    ).reshape(-1, 2)

    pop_coords = np.array(
        [(p.lat_deg, p.lon_deg) for p in pop_list], dtype=np.float64
    )

    nearest_idx = _vectorized_nearest_index(cell_coords, pop_coords)

    mapping: dict[int, str] = {
        cell_id: pop_list[int(nearest_idx[i])].code
        for i, cell_id in enumerate(cell_ids)
    }

    return CellToPopTable(
        mapping=MappingProxyType(mapping),
        version=version,
        built_at=built_at,
    )


def _vectorized_nearest_index(
    cell_coords: np.ndarray,
    pop_coords: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine argmin ``cell → pop``.

    Args:
        cell_coords: shape ``(n_cells, 2)`` ``(lat_deg, lon_deg)``.
        pop_coords:  shape ``(n_pops, 2)`` ``(lat_deg, lon_deg)``.

    Returns:
        ``(n_cells,)`` int array of the PoP index each cell maps to.
    """
    # Broadcast ``(n_cells, 1, 2)`` vs ``(1, n_pops, 2)`` into
    # ``(n_cells, n_pops, 2)``, then reduce along the last axis for
    # the haversine. We don't need the actual distance — argmin on
    # ``sin²(Δ/2)`` gives the same answer monotonically because ``arcsin``
    # and multiplication by a positive constant preserve ordering — so we
    # skip ``arcsin`` / Earth-radius multiplication entirely.
    cell_rad = np.deg2rad(cell_coords)[:, None, :]  # (n_cells, 1, 2)
    pop_rad = np.deg2rad(pop_coords)[None, :, :]    # (1, n_pops, 2)

    lat1 = cell_rad[..., 0]
    lat2 = pop_rad[..., 0]
    dlat = lat2 - lat1
    dlon = pop_rad[..., 1] - cell_rad[..., 1]

    # ``a`` is the classic haversine kernel. Monotonic in distance.
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    return a.argmin(axis=1)


def build_routing_plane_nearest_pop(
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
    *,
    version: int = 0,
) -> RoutingPlane:
    """Assemble a full nearest-PoP baseline :class:`RoutingPlane`.

    Composes :func:`build_cell_to_pop_nearest` (static geographic
    assignment) with :func:`build_satellite_fibs` (per-sat FIB derived
    from the Dijkstra output inside ``snapshot.satellite``). Intended
    to be called by :class:`NearestPoPController.compute_routing_plane`
    — exposed here so other policies that reuse the baseline as a
    fallback (e.g., warm-start for Progressive Filling) can call it
    directly.
    """
    cell_to_pop = build_cell_to_pop_nearest(
        cell_grid=cell_grid,
        pops=snapshot.infra.pops,
        built_at=snapshot.time_s,
        version=version,
    )
    per_sat_routing = precompute_per_sat_routing(snapshot)
    sat_fibs = build_satellite_fibs(snapshot, per_sat_routing, version=version)
    return RoutingPlane(
        cell_to_pop=cell_to_pop,
        sat_fibs=MappingProxyType(sat_fibs),
        version=version,
        built_at=snapshot.time_s,
    )
