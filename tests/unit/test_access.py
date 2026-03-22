"""Tests for satellite access link module."""

from __future__ import annotations

import numpy as np
import pytest

from vantage.world.satellite.visibility import (
    C_VACUUM_KM_S,
    EARTH_RADIUS_KM,
    AccessModel,
    SphericalAccessModel,
    _elevation_and_range,
    _to_ecef,
)


# ---------------------------------------------------------------------------
# ECEF conversion tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToECEF:
    """Test spherical ECEF coordinate conversion."""

    def test_equator_prime_meridian(self) -> None:
        """(0°, 0°, 0 km) → (R, 0, 0)."""
        x, y, z = _to_ecef(0.0, 0.0, 0.0)
        assert abs(x - EARTH_RADIUS_KM) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z) < 1e-6

    def test_north_pole(self) -> None:
        """(90°, 0°, 0 km) → (0, 0, R)."""
        x, y, z = _to_ecef(90.0, 0.0, 0.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z - EARTH_RADIUS_KM) < 1e-6

    def test_equator_90e(self) -> None:
        """(0°, 90°, 0 km) → (0, R, 0)."""
        x, y, z = _to_ecef(0.0, 90.0, 0.0)
        assert abs(x) < 1e-6
        assert abs(y - EARTH_RADIUS_KM) < 1e-6
        assert abs(z) < 1e-6

    def test_altitude_adds_to_radius(self) -> None:
        """Altitude should increase the radial distance."""
        alt = 550.0
        x, y, z = _to_ecef(0.0, 0.0, alt)
        assert abs(x - (EARTH_RADIUS_KM + alt)) < 1e-6

    def test_south_pole(self) -> None:
        """(-90°, 0°, 0 km) → (0, 0, -R)."""
        x, y, z = _to_ecef(-90.0, 0.0, 0.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        assert abs(z + EARTH_RADIUS_KM) < 1e-6


# ---------------------------------------------------------------------------
# Slant range tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestElevationAndRange:
    """Test combined elevation angle and slant range computation."""

    def test_same_point_zero_range(self) -> None:
        x, y, z = _to_ecef(0.0, 0.0, 0.0)
        elev, slant = _elevation_and_range(0.0, 0.0, x, y, z, x, y, z)
        assert elev == 90.0
        assert slant == 0.0

    def test_radial_offset_range_equals_altitude(self) -> None:
        """Ground to satellite directly overhead → range = altitude."""
        gx, gy, gz = _to_ecef(0.0, 0.0, 0.0)
        sx, sy, sz = _to_ecef(0.0, 0.0, 550.0)
        _, slant = _elevation_and_range(0.0, 0.0, gx, gy, gz, sx, sy, sz)
        assert abs(slant - 550.0) < 1e-6

    def test_range_symmetric(self) -> None:
        """Slant range is symmetric regardless of direction."""
        gx, gy, gz = _to_ecef(40.0, -74.0, 0.0)
        sx, sy, sz = _to_ecef(42.0, -72.0, 550.0)
        _, slant_fwd = _elevation_and_range(40.0, -74.0, gx, gy, gz, sx, sy, sz)
        _, slant_rev = _elevation_and_range(42.0, -72.0, sx, sy, sz, gx, gy, gz)
        assert abs(slant_fwd - slant_rev) < 1e-10

    def test_directly_overhead_is_90(self) -> None:
        """Satellite directly above ground point → 90° elevation."""
        gx, gy, gz = _to_ecef(0.0, 0.0, 0.0)
        sx, sy, sz = _to_ecef(0.0, 0.0, 550.0)
        elev, _ = _elevation_and_range(0.0, 0.0, gx, gy, gz, sx, sy, sz)
        assert abs(elev - 90.0) < 1e-6

    def test_directly_overhead_polar(self) -> None:
        """Satellite directly above north pole → 90° elevation."""
        gx, gy, gz = _to_ecef(90.0, 0.0, 0.0)
        sx, sy, sz = _to_ecef(90.0, 0.0, 550.0)
        elev, _ = _elevation_and_range(90.0, 0.0, gx, gy, gz, sx, sy, sz)
        assert abs(elev - 90.0) < 1e-6

    def test_horizon_is_near_zero(self) -> None:
        """Satellite near horizon should have low positive elevation."""
        gx, gy, gz = _to_ecef(0.0, 0.0, 0.0)
        # Satellite at 20° longitude offset, 550 km altitude → ~3.2° elevation
        sx, sy, sz = _to_ecef(0.0, 20.0, 550.0)
        elev, _ = _elevation_and_range(0.0, 0.0, gx, gy, gz, sx, sy, sz)
        assert abs(elev - 3.2) < 1.0  # within 1° of reference

    def test_below_horizon_is_negative(self) -> None:
        """Satellite on opposite side of Earth → negative elevation."""
        gx, gy, gz = _to_ecef(0.0, 0.0, 0.0)
        sx, sy, sz = _to_ecef(0.0, 180.0, 550.0)
        elev, _ = _elevation_and_range(0.0, 0.0, gx, gy, gz, sx, sy, sz)
        assert elev < 0

    def test_elevation_decreases_with_angular_distance(self) -> None:
        """Elevation should decrease as satellite moves away from zenith."""
        gx, gy, gz = _to_ecef(0.0, 0.0, 0.0)

        elevations = []
        for lon_offset in [0.0, 5.0, 10.0, 15.0, 20.0]:
            sx, sy, sz = _to_ecef(0.0, lon_offset, 550.0)
            elev, _ = _elevation_and_range(0.0, 0.0, gx, gy, gz, sx, sy, sz)
            elevations.append(elev)

        for i in range(len(elevations) - 1):
            assert elevations[i] > elevations[i + 1]

    def test_range_increases_with_angular_distance(self) -> None:
        """Slant range should increase as satellite moves from zenith."""
        gx, gy, gz = _to_ecef(0.0, 0.0, 0.0)

        ranges = []
        for lon_offset in [0.0, 5.0, 10.0, 15.0, 20.0]:
            sx, sy, sz = _to_ecef(0.0, lon_offset, 550.0)
            _, slant = _elevation_and_range(0.0, 0.0, gx, gy, gz, sx, sy, sz)
            ranges.append(slant)

        for i in range(len(ranges) - 1):
            assert ranges[i] < ranges[i + 1]


# ---------------------------------------------------------------------------
# SphericalAccessModel tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSphericalAccessModel:
    """Test the SphericalAccessModel implementation."""

    @pytest.fixture
    def model(self) -> SphericalAccessModel:
        return SphericalAccessModel()

    @pytest.fixture
    def overhead_constellation(self) -> np.ndarray:
        """3 satellites: one directly overhead, one nearby, one far away."""
        return np.array([
            [0.0, 0.0, 550.0],      # directly overhead (0°, 0°)
            [5.0, 5.0, 550.0],      # nearby
            [0.0, 180.0, 550.0],    # opposite side of Earth
        ])

    def test_compute_access_filters_by_elevation(
        self, model: SphericalAccessModel, overhead_constellation: np.ndarray
    ) -> None:
        """Only satellites above min elevation should be returned."""
        links = model.compute_access(0.0, 0.0, 0.0, overhead_constellation, 25.0)
        # Satellite 2 (opposite side) should be filtered out
        sat_ids = {link.sat_id for link in links}
        assert 2 not in sat_ids
        assert 0 in sat_ids  # directly overhead

    def test_compute_access_sorted_by_elevation(
        self, model: SphericalAccessModel, overhead_constellation: np.ndarray
    ) -> None:
        """Results should be sorted by elevation (highest first)."""
        links = model.compute_access(0.0, 0.0, 0.0, overhead_constellation, 0.0)
        for i in range(len(links) - 1):
            assert links[i].elevation_deg >= links[i + 1].elevation_deg

    def test_nearest_satellite_is_overhead(
        self, model: SphericalAccessModel, overhead_constellation: np.ndarray
    ) -> None:
        """Satellite directly overhead should be selected as nearest."""
        link = model.nearest_satellite(0.0, 0.0, 0.0, overhead_constellation, 25.0)
        assert link is not None
        assert link.sat_id == 0
        assert abs(link.elevation_deg - 90.0) < 1e-6

    def test_nearest_satellite_none_when_no_visible(
        self, model: SphericalAccessModel
    ) -> None:
        """None when no satellite is visible."""
        # All satellites on opposite side
        positions = np.array([
            [0.0, 180.0, 550.0],
            [0.0, -170.0, 550.0],
        ])
        link = model.nearest_satellite(0.0, 0.0, 0.0, positions, 25.0)
        assert link is None

    def test_overhead_delay_is_altitude_over_c(
        self, model: SphericalAccessModel
    ) -> None:
        """Directly overhead → slant range = altitude, delay = alt/c."""
        positions = np.array([[0.0, 0.0, 550.0]])
        link = model.nearest_satellite(0.0, 0.0, 0.0, positions, 0.0)
        assert link is not None
        assert abs(link.slant_range_km - 550.0) < 1e-6
        assert abs(link.delay - 550.0 / C_VACUUM_KM_S * 1000) < 1e-9

    def test_slant_range_increases_with_angle(
        self, model: SphericalAccessModel
    ) -> None:
        """Slant range should increase as satellite is further from zenith."""
        ranges = []
        for lon in [0.0, 5.0, 10.0, 15.0]:
            positions = np.array([[0.0, lon, 550.0]])
            link = model.nearest_satellite(0.0, 0.0, 0.0, positions, 0.0)
            assert link is not None
            ranges.append(link.slant_range_km)
        for i in range(len(ranges) - 1):
            assert ranges[i] < ranges[i + 1]

    def test_delay_positive_for_visible_satellites(
        self, model: SphericalAccessModel, overhead_constellation: np.ndarray
    ) -> None:
        """All visible satellites should have positive delay."""
        links = model.compute_access(0.0, 0.0, 0.0, overhead_constellation, 0.0)
        for link in links:
            assert link.delay > 0

    def test_invalid_positions_shape_raises(
        self, model: SphericalAccessModel
    ) -> None:
        """Invalid positions array shape should raise ValueError."""
        bad_positions = np.array([[0.0, 0.0]])  # shape (1, 2) instead of (n, 3)
        with pytest.raises(ValueError, match="shape"):
            model.compute_access(0.0, 0.0, 0.0, bad_positions, 25.0)

    def test_compute_access_pair_specific(
        self, model: SphericalAccessModel
    ) -> None:
        """compute_access_pair should work for a specific ground-sat pair."""
        link = model.compute_access_pair(
            ground_lat_deg=40.0, ground_lon_deg=-74.0, ground_alt_km=0.0,
            sat_lat_deg=42.0, sat_lon_deg=-72.0, sat_alt_km=550.0,
        )
        assert link.sat_id == -1  # placeholder for direct pair computation
        assert link.slant_range_km > 0
        assert link.delay > 0
        assert link.elevation_deg > 0

    def test_compute_access_pair_overhead(
        self, model: SphericalAccessModel
    ) -> None:
        """Direct overhead pair → 90° elevation, slant = altitude."""
        link = model.compute_access_pair(
            ground_lat_deg=0.0, ground_lon_deg=0.0, ground_alt_km=0.0,
            sat_lat_deg=0.0, sat_lon_deg=0.0, sat_alt_km=550.0,
        )
        assert abs(link.elevation_deg - 90.0) < 1e-6
        assert abs(link.slant_range_km - 550.0) < 1e-6

    def test_min_elevation_25_typical_coverage(
        self, model: SphericalAccessModel
    ) -> None:
        """With min elevation 25°, a satellite at 550km has limited footprint."""
        # At 550 km, 7° lon offset at equator → elevation ~27° (above 25° threshold)
        # At 550 km, 20° lon offset at equator → elevation ~3° (below 25° threshold)
        positions = np.array([
            [0.0, 7.0, 550.0],
            [0.0, 20.0, 550.0],
        ])
        links = model.compute_access(0.0, 0.0, 0.0, positions, 25.0)
        sat_ids = {link.sat_id for link in links}
        assert 0 in sat_ids
        assert 1 not in sat_ids

    def test_access_from_nonzero_altitude(
        self, model: SphericalAccessModel
    ) -> None:
        """Ground station at altitude should have shorter slant range to overhead sat."""
        link_sea = model.compute_access_pair(0.0, 0.0, 0.0, 0.0, 0.0, 550.0)
        link_high = model.compute_access_pair(0.0, 0.0, 5.0, 0.0, 0.0, 550.0)
        # Higher ground → shorter distance to satellite
        assert link_high.slant_range_km < link_sea.slant_range_km

    def test_realistic_starlink_delay_range(
        self, model: SphericalAccessModel
    ) -> None:
        """Access delay for Starlink (550km) should be in realistic range."""
        # Overhead: 550/300000 ≈ 1.83 ms
        # Low elevation (~10°): slant ~1800 km → ~6 ms
        # With 25° min elevation: slant ~1000 km → ~3.3 ms
        positions = np.array([
            [0.0, 0.0, 550.0],  # overhead
            [0.0, 5.0, 550.0],  # moderate elevation
        ])
        links = model.compute_access(0.0, 0.0, 0.0, positions, 25.0)
        for link in links:
            delay_ms = link.delay
            assert 1.5 < delay_ms < 5.0, (
                f"Unrealistic access delay: {delay_ms:.2f} ms "
                f"(elev={link.elevation_deg:.1f}°)"
            )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAccessModelProtocol:
    """Verify SphericalAccessModel satisfies AccessModel protocol."""

    def test_spherical_satisfies_protocol(self) -> None:
        """SphericalAccessModel must be structurally compatible with AccessModel."""
        model: AccessModel = SphericalAccessModel()
        positions = np.array([[0.0, 0.0, 550.0]])
        links = model.compute_access(0.0, 0.0, 0.0, positions, 0.0)
        assert len(links) == 1
        nearest = model.nearest_satellite(0.0, 0.0, 0.0, positions, 0.0)
        assert nearest is not None
