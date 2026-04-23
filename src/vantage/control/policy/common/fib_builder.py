"""Policy-agnostic helpers for assembling a :class:`RoutingPlane`.

Pure functions reused across baseline and future policies:

    * :func:`build_sat_path_table` — wraps the snapshot's
      ``delay_matrix`` / ``predecessor_matrix`` in a controller-owned
      :class:`SatPathTable`. The data plane reads sat-level routing
      through this artifact rather than reaching into
      ``SatelliteState`` directly, matching the PPT's 15 s refresh
      model.
    * :func:`build_pop_egress_table` — per PoP, flattens ``gs_pop_edges``
      × ``gateway_attachments`` into ``(egress_ids, base_cost, gs_ids)``
      numpy-ready arrays where ``base_cost = 2·downlink + 2·backhaul``.
      Replaces the former per-flow walk inside
      :class:`RoutingPlaneForward` with a controller-side precompute.
    * :func:`build_cell_to_pop_nearest` — geographic argmin from every
      :class:`Cell` center to the closest :class:`PoP`. This is the
      baseline ``cell_to_pop`` assignment; capacity-aware policies
      override it with a TE solver later.

Each function is stateless and takes explicit inputs, so they can be
unit-tested without spinning up a full snapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    NetworkSnapshot,
    PoP,
    PopEgressTable,
    RoutingPlane,
    SatPathTable,
)

__all__ = [
    "build_cell_to_pop_nearest",
    "build_pop_egress_table",
    "build_routing_plane_nearest_pop",
    "build_sat_path_table",
    "compute_cell_ingress",
    "compute_cell_sat_cost",
    "compute_pop_capacity",
    "rank_pops_by_e2e",
]


def compute_pop_capacity(snapshot: NetworkSnapshot) -> dict[str, float]:
    """Per-PoP aggregate ingress capacity (Gbps) from the ground segment.

    For each :class:`~vantage.domain.PoP`, sums
    :attr:`~vantage.domain.GroundStation.max_capacity` across every
    GS attached to the PoP via ``gs_pop_edges``. This is the
    coarse-grained envelope the planner uses to rank PoPs against
    aggregate demand — it is **not** a realize-time enforcement
    knob. Fine-grained per-sat-feeder / per-GS-feeder limits remain
    the data plane's responsibility (see
    :class:`~vantage.forward.RoutingPlaneForward`).
    """
    infra = snapshot.infra
    caps: dict[str, float] = {}
    for pop in infra.pops:
        total = 0.0
        for gs_id, _ in infra.pop_gs_edges(pop.code):
            gs = infra.gs_by_id(gs_id)
            if gs is not None:
                total += gs.max_capacity
        caps[pop.code] = total
    return caps

def build_sat_path_table(
    snapshot: NetworkSnapshot,
    *,
    version: int = 0,
) -> SatPathTable:
    """Wrap the snapshot's ISL shortest-path outputs in a controller artifact.

    Assumes the Dijkstra step already ran as part of the snapshot
    build (``compute_all_pairs`` populated
    ``SatelliteState.delay_matrix`` / ``predecessor_matrix``). This
    function just republishes those arrays through
    :class:`SatPathTable`, which is the view the data plane is
    allowed to read from.

    Why the indirection? The data plane previously reached into
    ``snapshot.satellite`` directly on the hot path, so it was
    effectively recomputing "which neighbour do I hand off to" every
    realize call. The routing plane now owns the artifact; the data
    plane consumes a reference, and the control layer is the only
    place that decides when/how the matrices get rebuilt.
    """
    sat = snapshot.satellite
    return SatPathTable(
        delay_matrix=sat.delay_matrix,
        predecessor_matrix=sat.predecessor_matrix,
        version=version,
        built_at=snapshot.time_s,
    )


def build_pop_egress_table(
    snapshot: NetworkSnapshot,
    *,
    version: int = 0,
) -> PopEgressTable:
    """Precompute per-PoP downlink candidate tables for the data plane.

    For each PoP in ``snapshot.infra``, walk ``pop_gs_edges`` ×
    ``gateway_attachments`` once and record:

        * ``egress_ids`` — int32 array of candidate egress sat ids.
        * ``base_cost`` — float64 array of ``2·downlink + 2·backhaul``
          RTT contributions per candidate. A flow's total sat-segment
          RTT from ``ingress`` to this PoP is then
          ``sat_paths.delay_row(ingress)[egress_ids] * 2 + base_cost``
          in one numpy op — replacing the per-flow walk over
          infrastructure that used to live in
          :class:`RoutingPlaneForward`.
        * ``gs_ids`` — the per-candidate GS id (so the data plane can
          charge the right GS feeder).

    PoPs with no attached GS, and GS rows with no visible sats in the
    current snapshot, contribute nothing and the PoP simply does not
    appear in the resulting table. :meth:`PopEgressTable.for_pop`
    returns empty arrays in that case so the data plane treats those
    PoPs as unreachable without special-casing ``None``.
    """
    sat = snapshot.satellite
    infra = snapshot.infra
    attach = sat.gateway_attachments.attachments

    candidates: dict[
        str, tuple[NDArray[np.int32], NDArray[np.float64], tuple[str, ...]]
    ] = {}

    for pop in infra.pops:
        egress_list: list[int] = []
        base_list: list[float] = []
        gs_list: list[str] = []
        for gs_id, backhaul_oneway in infra.pop_gs_edges(pop.code):
            if infra.gs_by_id(gs_id) is None:
                continue
            gs_links = attach.get(gs_id)
            if not gs_links:
                continue
            backhaul_rtt = backhaul_oneway * 2.0
            for link in gs_links:
                egress_list.append(link.sat_id)
                base_list.append(link.delay * 2.0 + backhaul_rtt)
                gs_list.append(gs_id)
        if not egress_list:
            continue
        egress_ids = np.asarray(egress_list, dtype=np.int32)
        base_cost = np.asarray(base_list, dtype=np.float64)
        egress_ids.flags.writeable = False
        base_cost.flags.writeable = False
        candidates[pop.code] = (egress_ids, base_cost, tuple(gs_list))

    return PopEgressTable(
        candidates=MappingProxyType(candidates),
        version=version,
        built_at=snapshot.time_s,
    )


def build_cell_to_pop_nearest(
    cell_grid: CellGrid,
    pops: Iterable[PoP],
    *,
    built_at: float,
    version: int = 0,
    top_n: int | None = None,
) -> CellToPopTable:
    """Assign every active cell to its full ranked PoP cascade by distance.

    Returns a :class:`CellToPopTable` whose ``mapping`` value is a
    ``tuple[str, ...]`` of length ``min(top_n, |pops|)`` — by default
    *all* PoPs sorted by haversine distance ASC. The head is the
    closest PoP; the tail is the cascading fallback chain the data
    plane walks when nearer PoPs' Ka feeders saturate.

    Only *active* cells (those that host at least one endpoint) are
    materialised in ``mapping``: the data plane only ever queries
    source endpoints' cells, and stub cells covering empty land
    would otherwise blow the table up by 300× at the production
    528 k-cell scale.

    Vectorized haversine + per-row sort. Production scale
    (~1.7 k active cells × ~50 PoPs) is < 5 ms end-to-end and ~700 KB
    of memory.

    The output ranking depends only on geography, so it never changes
    between refreshes for a fixed ``(cell_grid, pops)`` pair.
    Capacity-aware policies (Progressive Filling etc.) consume this
    output as their reference baseline AND emit their own per-(cell,
    dest) ranked overrides on top.

    Args:
        cell_grid: The set of cells in play (from endpoints).
        pops: All PoPs eligible as destinations.
        built_at: Simulation time stamp to record on the produced table.
        version: Monotonic version tag.
        top_n: Maximum length of each cell's ranked PoP tuple. ``None``
            (the default) means "all PoPs" — the controller's full
            cascade. Pass an explicit smaller integer to truncate
            (useful for tests or memory-constrained experiments).

    Raises:
        ValueError: If ``pops`` is empty, or if ``top_n`` is non-positive.
    """
    pop_list = tuple(pops)
    if not pop_list:
        raise ValueError("build_cell_to_pop_nearest: pops must be non-empty")
    if top_n is not None and top_n <= 0:
        raise ValueError(
            f"build_cell_to_pop_nearest: top_n must be positive (got {top_n})"
        )
    effective_n = min(top_n if top_n is not None else len(pop_list), len(pop_list))

    # Restrict to active cells: data plane only queries cells of
    # source endpoints. cell_grid.endpoint_to_cell.values() is the
    # full active set; intersect with cells.keys() defensively in
    # case any endpoint maps to a stripped cell.
    active_cell_ids = set(cell_grid.endpoint_to_cell.values())
    cell_ids = tuple(c for c in active_cell_ids if c in cell_grid.cells)

    if not cell_ids:
        return CellToPopTable(
            mapping=MappingProxyType({}),
            version=version,
            built_at=built_at,
        )

    cell_coords = np.array(
        [
            (cell_grid.cells[cid].lat_deg, cell_grid.cells[cid].lon_deg)
            for cid in cell_ids
        ],
        dtype=np.float64,
    )

    pop_coords = np.array(
        [(p.lat_deg, p.lon_deg) for p in pop_list], dtype=np.float64
    )

    ranked_idx = _vectorized_nearest_indices(
        cell_coords, pop_coords, top_n=effective_n,
    )
    pop_codes = tuple(p.code for p in pop_list)

    mapping: dict[int, tuple[str, ...]] = {
        cell_id: tuple(pop_codes[int(j)] for j in ranked_idx[i])
        for i, cell_id in enumerate(cell_ids)
    }

    return CellToPopTable(
        mapping=MappingProxyType(mapping),
        version=version,
        built_at=built_at,
    )


def _vectorized_nearest_indices(
    cell_coords: np.ndarray,
    pop_coords: np.ndarray,
    *,
    top_n: int,
) -> np.ndarray:
    """Vectorized haversine top-N PoPs per cell, sorted ASC.

    Args:
        cell_coords: shape ``(n_cells, 2)`` ``(lat_deg, lon_deg)``.
        pop_coords:  shape ``(n_pops, 2)`` ``(lat_deg, lon_deg)``.
        top_n: Number of nearest PoP indices to return per cell. Must
            satisfy ``1 <= top_n <= n_pops``.

    Returns:
        ``(n_cells, top_n)`` int array; ``out[i, k]`` is the index of
        cell ``i``'s ``k``-th nearest PoP (0-th = closest).
    """
    # Broadcast ``(n_cells, 1, 2)`` vs ``(1, n_pops, 2)`` into
    # ``(n_cells, n_pops, 2)``, then reduce along the last axis for
    # the haversine. We don't need the actual distance — sorting on
    # ``sin²(Δ/2)`` gives the same order monotonically because ``arcsin``
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

    n_pops = a.shape[1]
    if top_n >= n_pops:
        # Fewer / equal PoPs than requested — full sort is exactly
        # what we want and avoids the argpartition split.
        return np.argsort(a, axis=1)[:, :top_n]

    # argpartition pulls the top-N indices to the front (unsorted),
    # then we argsort just within that slice. O(n_pops + top_n log
    # top_n) per row vs O(n_pops log n_pops) for a full sort.
    part = np.argpartition(a, kth=top_n - 1, axis=1)[:, :top_n]
    rows = np.arange(a.shape[0])[:, None]
    order = np.argsort(a[rows, part], axis=1)
    return part[rows, order]


def compute_cell_ingress(
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
) -> dict[int, int]:
    """Pick a representative ingress satellite for each active cell.

    For each cell that hosts at least one endpoint, returns the
    top-elevation visible satellite from the cell centre. Used by
    both :func:`compute_cell_sat_cost` (to look up per-PoP sat-segment
    costs) and the controller's per-sat-feeder capacity tracking
    (which needs to know which egress sat each cell's flows will
    contend for).

    Cells whose centre has no visible satellite are silently omitted
    from the result; callers that iterate ``cell_grid`` should treat
    a missing key the same way they treat a missing endpoint.

    Vectorised over all active cells in a single numpy pass: the old
    per-cell ``find_ingress_satellite`` loop re-ran the ``(n_sats, 3)``
    geometry ~1 800 times at production scale (~200 ms); the batched
    version resolves the same argmax in one ``(n_cells, n_sats)``
    elevation matrix (≤30 ms).
    """
    import numpy as np

    from vantage.common import DEFAULT_MIN_ELEVATION_DEG
    from vantage.common.constants import EARTH_RADIUS_KM

    sat_positions = snapshot.satellite.positions
    if sat_positions.ndim != 2 or sat_positions.shape[1] != 3:
        raise ValueError(
            f"sat_positions must have shape (n_sats, 3); got {sat_positions.shape}"
        )

    active_cell_ids = set(cell_grid.endpoint_to_cell.values())
    items: list[tuple[int, float, float]] = []
    for cid in active_cell_ids:
        cell = cell_grid.cells.get(cid)
        if cell is not None:
            items.append((cid, cell.lat_deg, cell.lon_deg))
    if not items:
        return {}

    cell_ids = np.fromiter((it[0] for it in items), dtype=np.int64, count=len(items))
    lat = np.fromiter((it[1] for it in items), dtype=np.float64, count=len(items))
    lon = np.fromiter((it[2] for it in items), dtype=np.float64, count=len(items))

    g_lat = np.deg2rad(lat)
    g_lon = np.deg2rad(lon)
    cos_g_lat = np.cos(g_lat)
    gx = EARTH_RADIUS_KM * cos_g_lat * np.cos(g_lon)
    gy = EARTH_RADIUS_KM * cos_g_lat * np.sin(g_lon)
    gz = EARTH_RADIUS_KM * np.sin(g_lat)

    s_lat = np.deg2rad(sat_positions[:, 0])
    s_lon = np.deg2rad(sat_positions[:, 1])
    s_r = EARTH_RADIUS_KM + sat_positions[:, 2]
    cos_s_lat = np.cos(s_lat)
    sx = s_r * cos_s_lat * np.cos(s_lon)
    sy = s_r * cos_s_lat * np.sin(s_lon)
    sz = s_r * np.sin(s_lat)

    # (n_cells, n_sats) slant range
    dx = sx[None, :] - gx[:, None]
    dy = sy[None, :] - gy[:, None]
    dz = sz[None, :] - gz[:, None]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    ux = cos_g_lat * np.cos(g_lon)
    uy = cos_g_lat * np.sin(g_lon)
    uz = np.sin(g_lat)
    sin_elev = np.clip(
        (dx * ux[:, None] + dy * uy[:, None] + dz * uz[:, None])
        / np.maximum(dist, 1e-10),
        -1.0,
        1.0,
    )
    elev_deg = np.degrees(np.arcsin(sin_elev))

    # Mask invisible sats with -inf so argmax picks the best visible.
    masked = np.where(elev_deg >= DEFAULT_MIN_ELEVATION_DEG, elev_deg, -np.inf)
    best = np.argmax(masked, axis=1)
    valid = masked[np.arange(len(items)), best] > -np.inf

    out: dict[int, int] = {}
    for i in range(len(items)):
        if valid[i]:
            out[int(cell_ids[i])] = int(best[i])
    return out


def compute_cell_sat_cost(
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
) -> dict[tuple[int, str], float]:
    """Compute sat_cost from each *cell*'s representative ingress to every PoP.

    Iterates the unique active cells (via :func:`compute_cell_ingress`),
    looks up ``sat_cost[(ingress, pop_code)]`` for every PoP, and
    returns a ``{(cell_id, pop_code) → sat_rtt_ms}`` table.

    Earlier code iterated over every endpoint and overwrote the same
    ``(cell, pop)`` keys repeatedly — same data, just wasted work
    (and pre-Bug-9 it also advanced the shared module RNG once per
    endpoint, perturbing forward.realize's stochastic ingress).
    """
    sat_cost_table = precompute_sat_cost(snapshot)
    pop_codes = [p.code for p in snapshot.infra.pops]
    cell_ingress = compute_cell_ingress(snapshot, cell_grid)

    result: dict[tuple[int, str], float] = {}
    for cell_id, ingress_sat in cell_ingress.items():
        for pc in pop_codes:
            sc = sat_cost_table.get((ingress_sat, pc))
            if sc is not None:
                result[(cell_id, pc)] = sc
    return result


def rank_pops_by_e2e(
    cell_grid: CellGrid,
    pops: tuple[PoP, ...],
    baseline: CellToPopTable,
    cell_sat_cost: dict[tuple[int, str], float],
    ground_cost_fn: Callable[[str, str], float | None],
    dest_names: Iterable[str],
) -> dict[tuple[int, str], list[tuple[str, float]]]:
    """Rank all reachable PoPs by E2E cost for each (cell, dest).

    Returns ``{(cell_id, dest) → [(pop_code, e2e_cost), ...]}``
    sorted ascending by cost. The first entry is the best PoP.
    Only includes PoPs for which both sat_cost and ground_cost exist.
    """
    active_cells = set(cell_grid.endpoint_to_cell.values())
    rankings: dict[tuple[int, str], list[tuple[str, float]]] = {}

    for cell_id in active_cells:
        if cell_id not in baseline.mapping:
            continue

        for dest in dest_names:
            scored: list[tuple[str, float]] = []

            for pop in pops:
                gc = ground_cost_fn(pop.code, dest)
                if gc is None:
                    continue
                sc = cell_sat_cost.get((cell_id, pop.code))
                if sc is None:
                    continue
                scored.append((pop.code, sc + gc))

            if scored:
                scored.sort(key=lambda x: x[1])
                rankings[(cell_id, dest)] = scored

    return rankings


def build_routing_plane_nearest_pop(
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
    *,
    version: int = 0,
) -> RoutingPlane:
    """Assemble a full nearest-PoP baseline :class:`RoutingPlane`.

    Composes the three controller-owned artifacts:

        * :func:`build_cell_to_pop_nearest` — static geographic
          cell → ranked PoP cascade.
        * :func:`build_sat_path_table` — republishes Dijkstra output
          as the data plane's ISL shortest-path artifact.
        * :func:`build_pop_egress_table` — per-PoP precomputed downlink
          candidates for the data plane to vectorise over.

    Intended to be called by
    :meth:`NearestPoPController.compute_routing_plane` — exposed here
    so other policies that reuse the baseline (e.g., warm-start for
    Progressive Filling) can call it directly.
    """
    cell_to_pop = build_cell_to_pop_nearest(
        cell_grid=cell_grid,
        pops=snapshot.infra.pops,
        built_at=snapshot.time_s,
        version=version,
    )
    sat_paths = build_sat_path_table(snapshot, version=version)
    pop_egress = build_pop_egress_table(snapshot, version=version)
    return RoutingPlane(
        cell_to_pop=cell_to_pop,
        sat_paths=sat_paths,
        pop_egress=pop_egress,
        version=version,
        built_at=snapshot.time_s,
    )
