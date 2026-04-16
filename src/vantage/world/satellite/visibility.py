"""Satellite access link computation.

Computes slant range, elevation angle, and propagation delay between
ground points and satellites. Supports both user-satellite uplinks
and satellite-ground station downlinks.

Uses spherical Earth model (R=6371 km) consistent with topology.py.
"""

from __future__ import annotations

from math import asin, cos, degrees, radians, sin, sqrt
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vantage.common.constants import C_VACUUM_KM_S, EARTH_RADIUS_KM
from vantage.domain import AccessLink

__all__ = [
    "AccessModel",
    "SphericalAccessModel",
]


class AccessModel(Protocol):
    """Protocol for satellite access link computation."""

    def compute_access(
        self,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_km: float,
        sat_positions: NDArray[np.float64],
        min_elevation_deg: float,
    ) -> tuple[AccessLink, ...]:
        """Compute access links to all visible satellites.

        Args:
            ground_lat_deg: Ground point latitude (degrees).
            ground_lon_deg: Ground point longitude (degrees).
            ground_alt_km: Ground point altitude (km, 0 for sea level).
            sat_positions: Array of shape (n_sats, 3) with [lat_deg, lon_deg, alt_km].
            min_elevation_deg: Minimum elevation angle for visibility.

        Returns:
            Tuple of AccessLink for each visible satellite, sorted by
            elevation angle (highest first).
        """
        ...

    def nearest_satellite(
        self,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_km: float,
        sat_positions: NDArray[np.float64],
        min_elevation_deg: float,
    ) -> AccessLink | None:
        """Find the satellite with the highest elevation angle.

        Returns:
            AccessLink for the best satellite, or None if no satellite is visible.
        """
        ...


def _to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> tuple[float, float, float]:
    """Convert geodetic coordinates to ECEF using spherical Earth model.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        alt_km: Altitude above sea level in km.

    Returns:
        (x, y, z) in km.
    """
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    return x, y, z


def _elevation_and_range(
    ground_lat_deg: float,
    ground_lon_deg: float,
    gx: float, gy: float, gz: float,
    sx: float, sy: float, sz: float,
) -> tuple[float, float]:
    """Compute elevation angle and slant range in a single pass.

    Returns:
        (elevation_deg, slant_range_km). Elevation is negative when
        the satellite is below the local horizon.
    """
    dx = sx - gx
    dy = sy - gy
    dz = sz - gz
    dist = sqrt(dx * dx + dy * dy + dz * dz)

    if dist < 1e-10:
        return 90.0, 0.0

    lat = radians(ground_lat_deg)
    lon = radians(ground_lon_deg)
    ux = cos(lat) * cos(lon)
    uy = cos(lat) * sin(lon)
    uz = sin(lat)

    sin_elev = (dx * ux + dy * uy + dz * uz) / dist
    sin_elev = max(-1.0, min(1.0, sin_elev))
    return degrees(asin(sin_elev)), dist


class SphericalAccessModel:
    """Access model using spherical Earth approximation.

    Consistent with topology.py's haversine distance calculation.
    Uses speed of light in vacuum for propagation delay.
    """

    def compute_access(
        self,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_km: float,
        sat_positions: NDArray[np.float64],
        min_elevation_deg: float,
    ) -> tuple[AccessLink, ...]:
        """Compute access links to all visible satellites (vectorized)."""
        if sat_positions.ndim != 2 or sat_positions.shape[1] != 3:
            raise ValueError(
                f"sat_positions must have shape (n_sats, 3), got {sat_positions.shape}"
            )

        # Ground point ECEF
        g_lat = np.radians(ground_lat_deg)
        g_lon = np.radians(ground_lon_deg)
        g_r = EARTH_RADIUS_KM + ground_alt_km
        gx = g_r * np.cos(g_lat) * np.cos(g_lon)
        gy = g_r * np.cos(g_lat) * np.sin(g_lon)
        gz = g_r * np.sin(g_lat)

        # All satellites ECEF (vectorized)
        s_lat = np.radians(sat_positions[:, 0])
        s_lon = np.radians(sat_positions[:, 1])
        s_r = EARTH_RADIUS_KM + sat_positions[:, 2]
        sx = s_r * np.cos(s_lat) * np.cos(s_lon)
        sy = s_r * np.cos(s_lat) * np.sin(s_lon)
        sz = s_r * np.sin(s_lat)

        # Slant range
        dx, dy, dz = sx - gx, sy - gy, sz - gz
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Elevation angle
        ux = np.cos(g_lat) * np.cos(g_lon)
        uy = np.cos(g_lat) * np.sin(g_lon)
        uz = np.sin(g_lat)
        sin_elev = np.clip((dx * ux + dy * uy + dz * uz) / np.maximum(dist, 1e-10), -1.0, 1.0)
        elev = np.degrees(np.arcsin(sin_elev))

        # Filter visible satellites
        mask = elev >= min_elevation_deg
        visible_ids = np.where(mask)[0]
        visible_elev = elev[mask]
        visible_slant = dist[mask]

        # Sort by elevation descending
        order = np.argsort(-visible_elev)
        visible_ids = visible_ids[order]
        visible_elev = visible_elev[order]
        visible_slant = visible_slant[order]
        visible_delay = visible_slant / C_VACUUM_KM_S * 1000  # ms

        return tuple(
            AccessLink(
                sat_id=int(visible_ids[i]),
                elevation_deg=float(visible_elev[i]),
                slant_range_km=float(visible_slant[i]),
                delay=float(visible_delay[i]),
            )
            for i in range(len(visible_ids))
        )

    def nearest_satellite(
        self,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_km: float,
        sat_positions: NDArray[np.float64],
        min_elevation_deg: float,
    ) -> AccessLink | None:
        """Find the satellite with the highest elevation angle."""
        visible = self.compute_access(
            ground_lat_deg, ground_lon_deg, ground_alt_km,
            sat_positions, min_elevation_deg,
        )
        return visible[0] if visible else None

    def compute_access_pair(
        self,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_km: float,
        sat_lat_deg: float,
        sat_lon_deg: float,
        sat_alt_km: float,
    ) -> AccessLink:
        """Compute access link for a specific ground-satellite pair.

        Does not check elevation constraints. Use for known-good pairs.

        Args:
            ground_lat_deg: Ground point latitude.
            ground_lon_deg: Ground point longitude.
            ground_alt_km: Ground point altitude.
            sat_lat_deg: Satellite latitude.
            sat_lon_deg: Satellite longitude.
            sat_alt_km: Satellite altitude.

        Returns:
            AccessLink with computed slant range, elevation, and delay.
        """
        gx, gy, gz = _to_ecef(ground_lat_deg, ground_lon_deg, ground_alt_km)
        sx, sy, sz = _to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_km)

        elev, slant = _elevation_and_range(
            ground_lat_deg, ground_lon_deg, gx, gy, gz, sx, sy, sz
        )
        delay = slant / C_VACUUM_KM_S * 1000  # ms

        return AccessLink(
            sat_id=-1,
            elevation_deg=elev,
            slant_range_km=slant,
            delay=delay,
        )
