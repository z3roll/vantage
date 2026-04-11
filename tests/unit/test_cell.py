"""Unit tests for ``vantage.domain.cell``."""

from __future__ import annotations

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
