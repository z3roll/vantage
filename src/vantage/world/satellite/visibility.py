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

from vantage.domain import AccessLink

from vantage.common.constants import C_VACUUM_KM_S, EARTH_RADIUS_KM

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
        """Compute access links to all visible satellites."""
        if sat_positions.ndim != 2 or sat_positions.shape[1] != 3:
            raise ValueError(
                f"sat_positions must have shape (n_sats, 3), got {sat_positions.shape}"
            )

        gx, gy, gz = _to_ecef(ground_lat_deg, ground_lon_deg, ground_alt_km)
        links: list[AccessLink] = []

        for sat_id in range(sat_positions.shape[0]):
            sat_lat = float(sat_positions[sat_id, 0])
            sat_lon = float(sat_positions[sat_id, 1])
            sat_alt = float(sat_positions[sat_id, 2])

            sx, sy, sz = _to_ecef(sat_lat, sat_lon, sat_alt)
            elev, slant = _elevation_and_range(
                ground_lat_deg, ground_lon_deg, gx, gy, gz, sx, sy, sz
            )

            if elev >= min_elevation_deg:
                delay = slant / C_VACUUM_KM_S * 1000  # ms
                links.append(
                    AccessLink(
                        sat_id=sat_id,
                        elevation_deg=elev,
                        slant_range_km=slant,
                        delay=delay,
                    )
                )

        # Sort by elevation angle descending (highest first)
        links.sort(key=lambda link: link.elevation_deg, reverse=True)
        return tuple(links)

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
