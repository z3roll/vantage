"""Shared utilities for controller policy implementations.

Common helpers:
- find_nearest_pop: geographic PoP selection
- find_ingress_satellite: user→satellite uplink resolution
"""

from __future__ import annotations

import random as _random

import numpy as np
from numpy.typing import NDArray

from vantage.common import DEFAULT_MIN_ELEVATION_DEG, haversine_km
from vantage.domain import AccessLink, Endpoint, PoP
from vantage.world.satellite.visibility import SphericalAccessModel

_ACCESS_MODEL = SphericalAccessModel()
_RNG = _random.Random(0)


def find_nearest_pop(
    lat: float, lon: float, pops: tuple[PoP, ...]
) -> PoP | None:
    """Find the PoP geographically closest to (lat, lon)."""
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
    *,
    top_prob: float = 0.8,
    _visible: list[AccessLink] | None = None,
) -> AccessLink | None:
    """Pick an uplink satellite for a source endpoint.

    80% of the time picks the highest-elevation satellite (best signal,
    most likely to bent-pipe). 20% picks a random visible satellite
    (simulates obstructions, terminal diversity, etc.).

    Pass ``_visible`` to reuse a pre-computed visibility list (avoids
    re-scanning all satellites for the same source location).
    """
    if _visible is None:
        _visible = _ACCESS_MODEL.compute_access(
            src.lat_deg, src.lon_deg, 0.0, sat_positions, DEFAULT_MIN_ELEVATION_DEG,
        )
    if not _visible:
        return None
    if _RNG.random() < top_prob:
        return _visible[0]
    return _RNG.choice(_visible)
