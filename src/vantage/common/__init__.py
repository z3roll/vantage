"""Common utilities: constants, geographic functions, propagation delays, link models."""

from vantage.common.constants import (
    C_FIBER_KM_S,
    C_VACUUM_KM_S,
    DEFAULT_DETOUR_FACTOR,
    DEFAULT_MIN_ELEVATION_DEG,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
)
from vantage.common.geo import access_delay, haversine_km
from vantage.common.link_model import (
    LinkPerformance,
    bottleneck_capacity,
    link_performance,
    path_loss,
    pftk_throughput,
)

__all__ = [
    "C_FIBER_KM_S",
    "C_VACUUM_KM_S",
    "DEFAULT_DETOUR_FACTOR",
    "DEFAULT_MIN_ELEVATION_DEG",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_M",
    "LinkPerformance",
    "access_delay",
    "bottleneck_capacity",
    "haversine_km",
    "link_performance",
    "path_loss",
    "pftk_throughput",
]
