"""Geographic and propagation delay utilities.

All delay values in milliseconds (ms).
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from vantage.common.constants import C_VACUUM_KM_S, EARTH_RADIUS_KM


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * asin(sqrt(a))


def access_delay(
    ground_lat: float,
    ground_lon: float,
    sat_lat: float,
    sat_lon: float,
    sat_alt: float,
) -> float:
    """One-way propagation delay between a ground point and a satellite, in ms.

    Uses ECEF distance / speed of light. Ground altitude assumed 0 (sea level).
    """
    g_lat = radians(ground_lat)
    g_lon = radians(ground_lon)
    g_r = EARTH_RADIUS_KM

    s_lat = radians(sat_lat)
    s_lon = radians(sat_lon)
    s_r = EARTH_RADIUS_KM + sat_alt

    gx = g_r * cos(g_lat) * cos(g_lon)
    gy = g_r * cos(g_lat) * sin(g_lon)
    gz = g_r * sin(g_lat)

    sx = s_r * cos(s_lat) * cos(s_lon)
    sy = s_r * cos(s_lat) * sin(s_lon)
    sz = s_r * sin(s_lat)

    dist = sqrt((sx - gx) ** 2 + (sy - gy) ** 2 + (sz - gz) ** 2)
    return dist / C_VACUUM_KM_S * 1000
