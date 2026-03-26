"""Shared utilities for controller policy implementations.

Common helpers:
- find_nearest_pop: geographic PoP selection
- find_ingress_satellite: user→satellite uplink resolution
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vantage.common import DEFAULT_MIN_ELEVATION_DEG, haversine_km
from vantage.domain import AccessLink, Endpoint, PoP
from vantage.world.satellite.visibility import SphericalAccessModel

# Module-level singleton — SphericalAccessModel is stateless.
_ACCESS_MODEL = SphericalAccessModel()


def find_nearest_pop(
    lat: float, lon: float, pops: tuple[PoP, ...]
) -> PoP | None:
    """Find the PoP geographically closest to (lat, lon).

    Returns None if *pops* is empty.
    """
    best_dist = float("inf")
    best: PoP | None = None
    for pop in pops:
        d = haversine_km(lat, lon, pop.lat_deg, pop.lon_deg)
        if d < best_dist:
            best_dist = d
            best = pop
    return best


def find_ingress_satellite(
    src: Endpoint,
    sat_positions: NDArray[np.float64],
) -> AccessLink | None:
    """Find the best uplink satellite for a source endpoint.

    Returns the AccessLink with highest elevation, or None if no
    satellite is visible above the minimum elevation threshold.
    """
    return _ACCESS_MODEL.nearest_satellite(
        src.lat_deg, src.lon_deg, 0.0, sat_positions, DEFAULT_MIN_ELEVATION_DEG
    )
