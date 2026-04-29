"""Geographic cell domain types.

A ``Cell`` is a static, discrete patch of Earth surface used as the
aggregation unit for traffic-engineering decisions. We use `Uber's H3
<https://h3geo.org>`_ hexagonal grid at resolution 5 (~252 km² per cell)
to match the cell definition used by Starlink (see Mike Puchol,
"Modeling Starlink capacity").

Conventions:
    * ``CellId`` is the 64-bit int form of an H3 index. It is hashable
      and cheap to store. Use :func:`latlng_to_cell_id` / :func:`cell_id_to_str`
      to convert at boundaries.
    * ``Cell`` is a frozen dataclass carrying the id and the center
      (lat, lon) of the hex, pre-computed once.
    * ``CellGrid`` is an immutable lookup structure that (a) holds every
      ``Cell`` used by a simulation run, and (b) memoizes the
      endpoint-name → cell_id mapping so downstream code can resolve
      a flow's source cell in O(1).

Two constructors with different coverage semantics:

    * :meth:`CellGrid.from_endpoints` — materializes only the hexes
      that contain at least one endpoint. Tiny (~100 cells), suitable
      for tests or minimal demos.
    * :meth:`CellGrid.from_polygon_coverage` — enumerates **every**
      res-5 hex whose center falls inside one of a set of polygons
      (e.g. Natural Earth country multipolygons ⇒ global land mask).
      Large (~500 k cells at res 5 for global land) but physically
      meaningful: a satellite's visible-cell count then matches the
      ~10 k per FOR reported by Puchol.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import h3

__all__ = [
    "CELL_RESOLUTION",
    "Cell",
    "CellGrid",
    "CellId",
    "cell_id_to_str",
    "latlng_to_cell_id",
]

_log = logging.getLogger(__name__)


# H3 resolution used throughout vantage. Res 5 → ~252 km² average hex area,
# matching Starlink's operational cell definition per Mike Puchol's analysis.
CELL_RESOLUTION: int = 5

# A CellId is the 64-bit int form of an H3 index.
CellId = int


def latlng_to_cell_id(lat_deg: float, lon_deg: float, resolution: int = CELL_RESOLUTION) -> CellId:
    """Resolve a geographic point to its containing H3 cell id.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        resolution: H3 resolution (default :data:`CELL_RESOLUTION`).

    Returns:
        The 64-bit int form of the H3 index. H3 v4's native ``latlng_to_cell``
        returns a hex string; we normalize to int for hashing efficiency.
    """
    token = h3.latlng_to_cell(lat_deg, lon_deg, resolution)
    return h3.str_to_int(token)


def cell_id_to_str(cell_id: CellId) -> str:
    """Render a :data:`CellId` as its canonical H3 hex string.

    Useful for logging, serialization, and H3 library calls that expect
    the string form.
    """
    return h3.int_to_str(cell_id)


@dataclass(frozen=True, slots=True)
class Cell:
    """A geographic cell with precomputed center coordinates.

    Coordinates are in WGS-84 degrees. ``cell_id`` is the int form of
    the H3 index (see :data:`CellId`).
    """

    cell_id: CellId
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True)
class CellGrid:
    """Immutable lookup structure for all cells used by a simulation run.

    Holds two maps:
        * ``cells``: :data:`CellId` → :class:`Cell`. The authoritative
          per-cell metadata.
        * ``endpoint_to_cell``: endpoint-name → :data:`CellId`. Populated
          once at init so downstream code can resolve a flow's source cell
          without re-running H3.

    Two supported constructors, each tuned for a different coverage
    semantic:
        * :meth:`from_endpoints` — tiny grid containing only hexes
          that host at least one endpoint.
        * :meth:`from_polygon_coverage` — every hex whose centre falls
          inside one of a set of polygons, typically a global land
          mask. Use for any query that needs to know what a satellite
          *physically sees* rather than what a specific simulation
          set of endpoints happens to touch.

    Direct construction is for deserialization or tests only.
    """

    cells: Mapping[CellId, Cell]
    endpoint_to_cell: Mapping[str, CellId]

    @classmethod
    def from_endpoints(
        cls,
        endpoints: Iterable[tuple[str, float, float]],
        resolution: int = CELL_RESOLUTION,
    ) -> CellGrid:
        """Build a grid from ``(name, lat, lon)`` triples.

        Any endpoints that fall into the same hex share a :class:`Cell`.
        The cell's stored ``(lat_deg, lon_deg)`` is the hex center — not
        any one endpoint — so it's stable across runs.
        """
        cells: dict[CellId, Cell] = {}
        endpoint_to_cell: dict[str, CellId] = {}
        for name, lat, lon in endpoints:
            cid = latlng_to_cell_id(lat, lon, resolution)
            if cid not in cells:
                center_lat, center_lon = h3.cell_to_latlng(cell_id_to_str(cid))
                cells[cid] = Cell(cell_id=cid, lat_deg=center_lat, lon_deg=center_lon)
            endpoint_to_cell[name] = cid
        return cls(
            cells=MappingProxyType(cells),
            endpoint_to_cell=MappingProxyType(endpoint_to_cell),
        )

    @classmethod
    def from_polygon_coverage(
        cls,
        geojson_path: str | Path,
        resolution: int = CELL_RESOLUTION,
        endpoints: Iterable[tuple[str, float, float]] = (),
        cache_path: str | Path | None = None,
    ) -> CellGrid:
        """Build a grid from every cell whose centre falls inside a
        polygon in ``geojson_path``.

        The intended input is Natural Earth's country boundary
        GeoJSON (``ne_countries.geojson`` or equivalent): pointing
        this at the countries feature collection yields a **global
        land mask** of roughly 500 k res-5 cells. Any other polygon
        set works too — the loader has no opinion about "land"
        specifically, it just enumerates cells inside the shape(s).

        Args:
            geojson_path: Path to a GeoJSON ``FeatureCollection``
                whose features are ``Polygon`` or ``MultiPolygon``.
            resolution: H3 resolution (default :data:`CELL_RESOLUTION`).
            endpoints: Optional ``(name, lat, lon)`` triples. After
                the polygon sweep, each endpoint is mapped to its
                cell; any cell not already in the grid (for example
                a coastal hex whose centre is just offshore) is
                materialised so every endpoint resolves. **If this
                argument is omitted, the returned grid's
                ``endpoint_to_cell`` is empty and :meth:`cell_of`
                will raise :class:`KeyError` for every query — the
                grid is then only useful for containment checks via
                ``__contains__`` and for bulk-iterating ``cells``.**
            cache_path: Optional path to a JSON file that caches the
                resolved cell id set. On load, the cache is trusted
                only if its recorded ``geojson_mtime`` matches the
                current file mtime; stale caches are rebuilt
                automatically. The first run takes ~30–60 s for the
                global land mask; subsequent runs take <1 s.

        Raises:
            FileNotFoundError: If ``geojson_path`` does not exist.
            ValueError: If the file is not a ``FeatureCollection``.
        """
        geojson_path = Path(geojson_path)
        if not geojson_path.exists():
            raise FileNotFoundError(f"geojson not found: {geojson_path}")

        cell_ids: set[CellId]
        cached = _load_cell_id_cache(cache_path, geojson_path, resolution)
        if cached is not None:
            cell_ids = cached
        else:
            cell_ids = _enumerate_polygon_cells(geojson_path, resolution)
            _save_cell_id_cache(cache_path, geojson_path, resolution, cell_ids)

        # Materialize Cell objects for every id in the set.
        cells: dict[CellId, Cell] = {}
        for cid in cell_ids:
            center_lat, center_lon = h3.cell_to_latlng(cell_id_to_str(cid))
            cells[cid] = Cell(cell_id=cid, lat_deg=center_lat, lon_deg=center_lon)

        # Augment with endpoint cells so every endpoint resolves even
        # if its enclosing hex was missed by the polygon sweep (common
        # for coastal endpoints whose hex centre is offshore).
        endpoint_to_cell: dict[str, CellId] = {}
        for name, lat, lon in endpoints:
            cid = latlng_to_cell_id(lat, lon, resolution)
            if cid not in cells:
                c_lat, c_lon = h3.cell_to_latlng(cell_id_to_str(cid))
                cells[cid] = Cell(cell_id=cid, lat_deg=c_lat, lon_deg=c_lon)
            endpoint_to_cell[name] = cid

        return cls(
            cells=MappingProxyType(cells),
            endpoint_to_cell=MappingProxyType(endpoint_to_cell),
        )

    def cell_of(self, endpoint_name: str) -> CellId:
        """Return the :data:`CellId` hosting ``endpoint_name``.

        Raises :class:`KeyError` if the endpoint was not registered at
        construction time.
        """
        return self.endpoint_to_cell[endpoint_name]

    def __len__(self) -> int:
        return len(self.cells)

    def __contains__(self, cell_id: object) -> bool:
        return cell_id in self.cells


# --- Polygon enumeration helpers --------------------------------------------


def _enumerate_polygon_cells(geojson_path: Path, resolution: int) -> set[CellId]:
    """Return every res-N cell whose centre is inside any polygon in
    the GeoJSON feature collection.

    Handles both ``Polygon`` and ``MultiPolygon`` geometry types, and
    respects holes (lakes, enclaves). The polygon sweep is done
    country-by-country; we print progress roughly every 50 features
    so the operator sees the loader isn't stuck.

    Polygons that crash ``h3.polygon_to_cells`` (for example shapes
    that straddle the ±180° antimeridian, such as Antarctica in
    Natural Earth) are skipped with a warning — Starlink's user base
    doesn't live on those polygons so the loss of coverage is
    irrelevant for flow routing.
    """
    with geojson_path.open() as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError(
            f"{geojson_path}: expected GeoJSON FeatureCollection, "
            f"got {data.get('type')!r}"
        )

    features = data.get("features", [])
    _log.info(
        "enumerating res-%d cells in %d polygons from %s",
        resolution, len(features), geojson_path.name,
    )
    acc: set[CellId] = set()
    skipped: list[str] = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry")
        if geom is None:
            continue
        name = feat.get("properties", {}).get("NAME", f"feature_{i}")
        gtype = geom.get("type")
        if gtype == "Polygon":
            cells = _polygon_to_cell_ids_safe(geom["coordinates"], resolution, name)
            if cells is None:
                skipped.append(name)
            else:
                acc.update(cells)
        elif gtype == "MultiPolygon":
            for j, poly_coords in enumerate(geom["coordinates"]):
                label = f"{name}[{j}]"
                cells = _polygon_to_cell_ids_safe(poly_coords, resolution, label)
                if cells is None:
                    skipped.append(label)
                else:
                    acc.update(cells)
        # silently skip non-polygon geometries
        if (i + 1) % 50 == 0:
            _log.info(
                "polygon sweep progress: %d/%d features, %d cells so far",
                i + 1, len(features), len(acc),
            )
    if skipped:
        _log.warning(
            "skipped %d polygon(s) that h3 couldn't enumerate "
            "(likely antimeridian crossings): %s%s",
            len(skipped),
            skipped[:5],
            "..." if len(skipped) > 5 else "",
        )
    _log.info("polygon sweep done: %d unique cells", len(acc))
    return acc


def _polygon_to_cell_ids_safe(
    coords: list[list[list[float]]], resolution: int, label: str
) -> set[CellId] | None:
    """Call h3.polygon_to_cells on one polygon with outer ring + holes.

    GeoJSON polygon coordinates are ``[outer_ring, hole1, hole2, ...]``,
    each ring being a list of ``[lon, lat]`` pairs. H3 wants
    ``[lat, lng]`` instead, so we flip them here.

    Returns ``None`` if ``polygon_to_cells`` raises an
    :class:`h3.H3BaseException` — the caller records the label as
    skipped. Narrowing the catch to h3's own exception hierarchy
    keeps programming errors (TypeError / AttributeError from
    malformed coordinate structures) visible instead of silently
    dropping countries.
    """
    if not coords:
        return set()
    outer = [[lat, lon] for lon, lat in coords[0]]
    holes = [
        [[lat, lon] for lon, lat in ring]
        for ring in coords[1:]
    ]
    try:
        poly = h3.LatLngPoly(outer, *holes)
        return {h3.str_to_int(c) for c in h3.polygon_to_cells(poly, resolution)}
    except h3.H3BaseException as exc:
        _log.debug("polygon %s skipped: %s", label, type(exc).__name__)
        return None


# --- On-disk cache ----------------------------------------------------------
#
# The polygon sweep takes 30–60 s on a global land mask, so we cache the
# resolved cell id set to a JSON file. The cache carries the source
# geojson mtime and resolution; any mismatch invalidates it.


def _load_cell_id_cache(
    cache_path: str | Path | None,
    geojson_path: Path,
    resolution: int,
) -> set[CellId] | None:
    if cache_path is None:
        return None
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("resolution") != resolution:
        return None
    if payload.get("geojson_mtime") != geojson_path.stat().st_mtime:
        return None
    if payload.get("geojson_path") != str(geojson_path):
        return None
    ids = payload.get("cell_ids")
    if not isinstance(ids, list):
        return None
    _log.info(
        "loaded %d cells from cache %s (geojson mtime match)",
        len(ids), cache_path.name,
    )
    return {int(x) for x in ids}


def _save_cell_id_cache(
    cache_path: str | Path | None,
    geojson_path: Path,
    resolution: int,
    cell_ids: set[CellId],
) -> None:
    """Write cache atomically: temp file + os.rename.

    An interrupted write would otherwise leave a partial JSON on
    disk forever, causing every subsequent run to silently re-sweep
    the polygons. The temp file lives in the same directory so
    os.rename can be an atomic in-filesystem operation.
    """
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resolution": resolution,
        "geojson_path": str(geojson_path),
        "geojson_mtime": geojson_path.stat().st_mtime,
        "cell_ids": sorted(cell_ids),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, cache_path)
    _log.info("saved cache to %s", cache_path.name)
