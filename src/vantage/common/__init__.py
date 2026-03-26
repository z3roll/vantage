"""Common utilities: constants, geographic functions, propagation delays."""

from vantage.common.constants import (
    C_FIBER_KM_S,
    C_VACUUM_KM_S,
    DEFAULT_DETOUR_FACTOR,
    DEFAULT_MIN_ELEVATION_DEG,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
)
from vantage.common.geo import access_delay, haversine_km
from vantage.common.time import resolve_local_time

__all__ = [
    "C_FIBER_KM_S",
    "C_VACUUM_KM_S",
    "DEFAULT_DETOUR_FACTOR",
    "DEFAULT_MIN_ELEVATION_DEG",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_M",
    "access_delay",
    "haversine_km",
    "resolve_local_time",
]
