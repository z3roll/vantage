"""Unit tests for ``vantage.domain.cell``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vantage.domain.cell import (
    CELL_RESOLUTION,
    Cell,
    CellGrid,
    cell_id_to_str,
    latlng_to_cell_id,
)


@pytest.mark.unit
class TestLatLngToCellId:
    """Conversion between (lat, lon) and H3 cell ids."""

    def test_returns_int(self) -> None:
        cid = latlng_to_cell_id(37.77, -122.42)
        assert isinstance(cid, int)
        assert cid > 0

    def test_round_trip_through_str(self) -> None:
        cid = latlng_to_cell_id(37.77, -122.42)
        token = cell_id_to_str(cid)
        # An H3 res-5 token is 15 lowercase hex chars.
        assert isinstance(token, str)
        assert len(token) == 15
        assert all(c in "0123456789abcdef" for c in token)

    def test_same_hex_same_id(self) -> None:
        """Two points in the same hex share the same cell id."""
        # Tiny delta, well inside any H3 res-5 hex (~252 km²).
        cid_a = latlng_to_cell_id(37.770000, -122.420000)
        cid_b = latlng_to_cell_id(37.770005, -122.420005)
        assert cid_a == cid_b

    def test_different_hemispheres_different_id(self) -> None:
        cid_a = latlng_to_cell_id(37.77, -122.42)
        cid_b = latlng_to_cell_id(-37.77, 122.42)
        assert cid_a != cid_b


@pytest.mark.unit
class TestCellFrozen:
    """``Cell`` is an immutable frozen dataclass."""

    def test_cannot_mutate(self) -> None:
        cell = Cell(cell_id=1, lat_deg=0.0, lon_deg=0.0)
        with pytest.raises(AttributeError):
            cell.lat_deg = 1.0  # type: ignore[misc]

    def test_hashable(self) -> None:
        cell = Cell(cell_id=1, lat_deg=0.0, lon_deg=0.0)
        assert hash(cell) == hash(Cell(cell_id=1, lat_deg=0.0, lon_deg=0.0))


@pytest.mark.unit
class TestCellGridFromEndpoints:
    """``CellGrid.from_endpoints`` correctly materialises the lookup tables."""

    def test_each_endpoint_registered(self) -> None:
        grid = CellGrid.from_endpoints(
            [
                ("sf", 37.77, -122.42),
                ("nyc", 40.71, -74.01),
            ]
        )
        assert "sf" in grid.endpoint_to_cell
        assert "nyc" in grid.endpoint_to_cell
        assert grid.endpoint_to_cell["sf"] != grid.endpoint_to_cell["nyc"]

    def test_cell_count_deduped(self) -> None:
        """Two endpoints inside the same hex produce only one Cell."""
        grid = CellGrid.from_endpoints(
            [
                ("a", 37.770000, -122.420000),
                ("b", 37.770005, -122.420005),
            ]
        )
        assert len(grid) == 1
        # Both endpoints map to the same cell id.
        assert grid.cell_of("a") == grid.cell_of("b")

    def test_cell_center_is_hex_center_not_endpoint(self) -> None:
        """Stored (lat, lon) must be the hex center — stable across runs.

        We use two points that are proven to share a hex by the earlier
        ``test_same_hex_same_id`` test, so the equality assertion below
        is *always* exercised (no conditional guard).
        """
        grid_a = CellGrid.from_endpoints([("a", 37.770000, -122.420000)])
        grid_b = CellGrid.from_endpoints([("b", 37.770005, -122.420005)])
        cid_a = grid_a.cell_of("a")
        cid_b = grid_b.cell_of("b")
        # Invariant from test_same_hex_same_id: these two points land in
        # the same hex.
        assert cid_a == cid_b
        cell_a = grid_a.cells[cid_a]
        cell_b = grid_b.cells[cid_b]
        # Stored center must match exactly across runs — independent of
        # whichever interior point was fed into from_endpoints.
        assert cell_a == cell_b
        # And the stored coordinates must not equal either input point
        # (the center of a hex is essentially never one of its interior
        # sample points).
        assert (cell_a.lat_deg, cell_a.lon_deg) != (37.770000, -122.420000)

    def test_cell_of_unknown_endpoint_raises(self) -> None:
        grid = CellGrid.from_endpoints([("sf", 37.77, -122.42)])
        with pytest.raises(KeyError):
            grid.cell_of("unknown")

    def test_contains(self) -> None:
        grid = CellGrid.from_endpoints([("sf", 37.77, -122.42)])
        cid = grid.cell_of("sf")
        assert cid in grid
        assert 0 not in grid

    def test_default_resolution_is_five(self) -> None:
        assert CELL_RESOLUTION == 5

    def test_lookup_table_is_readonly(self) -> None:
        grid = CellGrid.from_endpoints([("sf", 37.77, -122.42)])
        # Must not be possible to mutate the underlying mapping through
        # the public attribute — we return a MappingProxy.
        with pytest.raises(TypeError):
            grid.endpoint_to_cell["hack"] = 42  # type: ignore[index]


# --- from_polygon_coverage --------------------------------------------------


def _write_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )


def _square_feature(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> dict:
    """Build a GeoJSON Polygon feature for an axis-aligned lat/lon
    rectangle. Coordinates follow the GeoJSON convention ``[lon, lat]``."""
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        },
    }


@pytest.mark.unit
class TestFromPolygonCoverage:
    """``from_polygon_coverage`` enumerates cells via ``h3.polygon_to_cells``."""

    def test_small_polygon_yields_nonzero_cells(self, tmp_path: Path) -> None:
        """A 1°×1° square around the equator at res 5 covers ~40+ hexes."""
        geojson = tmp_path / "square.geojson"
        _write_geojson(geojson, [_square_feature(-0.5, -0.5, 0.5, 0.5)])
        grid = CellGrid.from_polygon_coverage(geojson)
        # 1°×1° ≈ 111 km × 111 km ≈ 12_300 km². Res-5 hex ≈ 252 km².
        # Expected: ~40 cells, give or take shape irregularities.
        assert 20 < len(grid) < 100

    def test_polygon_cell_centers_are_hex_centers(self, tmp_path: Path) -> None:
        geojson = tmp_path / "square.geojson"
        _write_geojson(geojson, [_square_feature(-0.5, -0.5, 0.5, 0.5)])
        grid = CellGrid.from_polygon_coverage(geojson)
        # Sanity: every materialized Cell's (lat,lon) round-trips through
        # H3 to the same cell id.
        for cid, cell in grid.cells.items():
            assert latlng_to_cell_id(cell.lat_deg, cell.lon_deg) == cid

    def test_multipolygon_feature(self, tmp_path: Path) -> None:
        """Two disjoint squares via a single MultiPolygon feature."""
        geojson = tmp_path / "multi.geojson"
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[
                        [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                        [-0.5, 0.5], [-0.5, -0.5],
                    ]],
                    [[
                        [100.0, 0.0], [101.0, 0.0], [101.0, 1.0],
                        [100.0, 1.0], [100.0, 0.0],
                    ]],
                ],
            },
        }
        _write_geojson(geojson, [feature])
        grid = CellGrid.from_polygon_coverage(geojson)
        # Both squares should contribute cells.
        assert len(grid) > 40  # at least one square's worth

    def test_endpoint_augmentation_adds_missing_cell(self, tmp_path: Path) -> None:
        """An endpoint outside any polygon still resolves."""
        geojson = tmp_path / "tiny.geojson"
        _write_geojson(geojson, [_square_feature(-0.1, -0.1, 0.1, 0.1)])
        # Endpoint on the opposite side of the globe.
        remote = ("antarctica_probe", -70.0, 120.0)
        grid = CellGrid.from_polygon_coverage(geojson, endpoints=[remote])
        assert "antarctica_probe" in grid.endpoint_to_cell
        cell_id = grid.cell_of("antarctica_probe")
        assert cell_id in grid
        # The cell was not in the polygon coverage but got added anyway.
        assert cell_id == latlng_to_cell_id(-70.0, 120.0)

    def test_endpoint_augmentation_preexisting_cell(self, tmp_path: Path) -> None:
        """Endpoint whose hex is already in the polygon sweep: cell
        count must not grow, but the endpoint mapping must still be
        populated so ``cell_of`` works."""
        geojson = tmp_path / "sq.geojson"
        _write_geojson(geojson, [_square_feature(-0.5, -0.5, 0.5, 0.5)])
        grid_no_ep = CellGrid.from_polygon_coverage(geojson)
        endpoint = ("inside", 0.0, 0.0)
        grid_with_ep = CellGrid.from_polygon_coverage(
            geojson, endpoints=[endpoint]
        )
        # The (0, 0) hex is definitely inside the square → already in
        # the polygon sweep → no new Cell materialised.
        assert len(grid_with_ep) == len(grid_no_ep)
        # But the endpoint mapping is populated.
        assert "inside" in grid_with_ep.endpoint_to_cell
        assert grid_with_ep.cell_of("inside") == latlng_to_cell_id(0.0, 0.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CellGrid.from_polygon_coverage(tmp_path / "nope.geojson")

    def test_non_feature_collection_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bogus.geojson"
        path.write_text(json.dumps({"type": "NotACollection"}))
        with pytest.raises(ValueError, match="FeatureCollection"):
            CellGrid.from_polygon_coverage(path)


@pytest.mark.unit
class TestPolygonCoverageCache:
    """Cache hit/miss semantics."""

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        geojson = tmp_path / "square.geojson"
        _write_geojson(geojson, [_square_feature(-0.5, -0.5, 0.5, 0.5)])
        cache = tmp_path / "cache.json"

        grid1 = CellGrid.from_polygon_coverage(geojson, cache_path=cache)
        assert cache.exists()

        # Second call: cache hit.
        grid2 = CellGrid.from_polygon_coverage(geojson, cache_path=cache)
        assert set(grid1.cells) == set(grid2.cells)

    def test_cache_invalidated_on_geojson_mtime(self, tmp_path: Path) -> None:
        geojson = tmp_path / "square.geojson"
        _write_geojson(geojson, [_square_feature(-0.5, -0.5, 0.5, 0.5)])
        cache = tmp_path / "cache.json"
        CellGrid.from_polygon_coverage(geojson, cache_path=cache)

        # Rewrite the geojson with a different shape; cache must rebuild.
        import time
        time.sleep(0.01)  # ensure mtime ticks forward
        _write_geojson(geojson, [_square_feature(10.0, 10.0, 11.0, 11.0)])
        grid_rebuilt = CellGrid.from_polygon_coverage(geojson, cache_path=cache)

        # New cells are around (10.5, 10.5), not (0, 0).
        sample_cell = next(iter(grid_rebuilt.cells.values()))
        assert sample_cell.lat_deg > 5.0

    def test_cache_invalidated_on_resolution_mismatch(self, tmp_path: Path) -> None:
        geojson = tmp_path / "square.geojson"
        _write_geojson(geojson, [_square_feature(-0.5, -0.5, 0.5, 0.5)])
        cache = tmp_path / "cache.json"
        CellGrid.from_polygon_coverage(geojson, cache_path=cache, resolution=5)

        grid_res6 = CellGrid.from_polygon_coverage(
            geojson, cache_path=cache, resolution=6
        )
        # Res 6 has ~7× more cells than res 5 for the same polygon.
        assert len(grid_res6) > 100
