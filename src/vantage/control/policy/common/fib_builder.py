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
from collections.abc import Callable, Iterable
from types import MappingProxyType

import numpy as np

from vantage.control.policy.common.sat_cost import (
    PerSatRouting,
    precompute_per_sat_routing,
    precompute_sat_cost,
)
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    Endpoint,
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
    "build_routing_plane_with_overrides",
    "build_satellite_fibs",
    "compute_cell_ingress",
    "compute_cell_sat_cost",
    "compute_e2e_overrides",
    "rank_pops_by_e2e",
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
    """
    from vantage.control.policy.common.utils import find_ingress_satellite

    sat_positions = snapshot.satellite.positions
    active_cell_ids = set(cell_grid.endpoint_to_cell.values())

    out: dict[int, int] = {}
    for cell_id in active_cell_ids:
        cell = cell_grid.cells.get(cell_id)
        if cell is None:
            continue
        # Endpoint name is irrelevant — find_ingress_satellite reads
        # only lat/lon. Use a synthetic name so future readers don't
        # mistakenly think it's an actual endpoint.
        ep = Endpoint(name="_cell_centre", lat_deg=cell.lat_deg, lon_deg=cell.lon_deg)
        uplink = find_ingress_satellite(ep, sat_positions, top_prob=1.0)
        if uplink is not None:
            out[cell_id] = uplink.sat_id
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


def compute_e2e_overrides(
    cell_grid: CellGrid,
    pops: tuple[PoP, ...],
    baseline: CellToPopTable,
    cell_sat_cost: dict[tuple[int, str], float],
    ground_cost_fn: Callable[[str, str], float | None],
    dest_names: Iterable[str],
    *,
    min_improvement_ms: float = 2.0,
) -> dict[tuple[int, str], str]:
    """Find per-(cell, dest) PoP overrides that improve E2E latency.

    For each active cell and destination, checks whether a non-default
    PoP gives lower ``sat_cost + ground_cost``. Only overrides with
    improvement exceeding *min_improvement_ms* are emitted (avoids
    noise from near-ties).

    Args:
        ground_cost_fn: ``(pop_code, dest) → RTT_ms | None``.
            Return ``None`` for unknown pairs.
        dest_names: Destinations to optimise for.
    """
    active_cells = set(cell_grid.endpoint_to_cell.values())
    overrides: dict[tuple[int, str], str] = {}

    for cell_id in active_cells:
        # ``baseline.mapping`` is now ``cell → ranked PoP tuple``; the
        # head is the geographic-nearest, which is what this legacy
        # override picker treats as "default".
        default_ranked = baseline.mapping.get(cell_id)
        if not default_ranked:
            continue
        default_pop = default_ranked[0]

        for dest in dest_names:
            default_ground = ground_cost_fn(default_pop, dest)
            if default_ground is None:
                continue
            default_sat = cell_sat_cost.get((cell_id, default_pop), 50.0)
            default_cost = default_sat + default_ground

            best_pop = default_pop
            best_cost = default_cost

            for pop in pops:
                if pop.code == default_pop:
                    continue
                gc = ground_cost_fn(pop.code, dest)
                if gc is None:
                    continue
                sc = cell_sat_cost.get((cell_id, pop.code), 50.0)
                total = sc + gc
                if total < best_cost:
                    best_cost = total
                    best_pop = pop.code

            if best_pop != default_pop and default_cost - best_cost > min_improvement_ms:
                overrides[(cell_id, dest)] = best_pop

    return overrides


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


def build_routing_plane_with_overrides(
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
    per_dest_overrides: dict[tuple[int, str], str],
    *,
    baseline: CellToPopTable | None = None,
    version: int = 0,
) -> RoutingPlane:
    """Build RoutingPlane with per-dest overrides on top of nearest-PoP baseline.

    Composes :func:`build_cell_to_pop_nearest` with the given overrides
    and :func:`build_satellite_fibs`. Used by all E2E-aware controllers
    (service-aware, greedy, ground-only, static-pop).

    Pass *baseline* to reuse an already-computed :class:`CellToPopTable`
    and avoid a redundant :func:`build_cell_to_pop_nearest` call.

    ``per_dest_overrides`` is the legacy single-PoP override shape
    (``(cell, dest) → str``). Each value is wrapped into a 1-tuple
    before being stored on the :class:`CellToPopTable`, which now
    expects ranked tuples per (cell, dest); fallback alternates for
    these legacy controllers come from the baseline ``mapping`` walk
    that the data plane does after the override is exhausted.
    """
    cell_to_pop_base = baseline or build_cell_to_pop_nearest(
        cell_grid=cell_grid,
        pops=snapshot.infra.pops,
        built_at=snapshot.time_s,
        version=version,
    )
    ranked_overrides: dict[tuple[int, str], tuple[str, ...]] = {
        key: (pop_code,) for key, pop_code in per_dest_overrides.items()
    }
    cell_to_pop = CellToPopTable(
        mapping=cell_to_pop_base.mapping,
        version=version,
        built_at=snapshot.time_s,
        per_dest=MappingProxyType(ranked_overrides),
    )
    per_sat = precompute_per_sat_routing(snapshot)
    sat_fibs = build_satellite_fibs(snapshot, per_sat, version=version)
    return RoutingPlane(
        cell_to_pop=cell_to_pop,
        sat_fibs=MappingProxyType(sat_fibs),
        version=version,
        built_at=snapshot.time_s,
    )


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
