"""Shared utilities for controller policy implementations.

Common helpers:
- find_ingress_satellite: user→satellite uplink resolution
"""

from __future__ import annotations

import random as _random

import numpy as np
from numpy.typing import NDArray

from vantage.common import DEFAULT_MIN_ELEVATION_DEG
from vantage.domain import AccessLink, Endpoint
from vantage.world.satellite.visibility import SphericalAccessModel

_ACCESS_MODEL = SphericalAccessModel()
_RNG = _random.Random(0)


def find_ingress_satellite(
    src: Endpoint,
    sat_positions: NDArray[np.float64],
    *,
    top_prob: float = 0.8,
    rng: _random.Random | None = None,
    _visible: list[AccessLink] | None = None,
) -> AccessLink | None:
    """Pick an uplink satellite for a source endpoint.

    With probability ``top_prob`` the highest-elevation visible
    satellite is returned (best signal, most likely to bent-pipe);
    with probability ``1 - top_prob`` a uniform-random visible sat
    is returned (simulates obstructions, terminal diversity, etc.).

    Two independent fast paths to avoid RNG consumption:

    * ``top_prob >= 1.0`` short-circuits to the top-elevation sat
      with **no RNG access at all** — a deterministic caller (e.g.
      :func:`compute_cell_sat_cost`) does not perturb the RNG state
      that another module may be relying on.
    * Empty visibility list returns ``None`` before any RNG read.

    Otherwise the function consults *rng* if provided, falling back
    to a process-wide module-level RNG seeded at import time.
    Stochastic callers that care about reproducibility should pass
    their own :class:`random.Random` instance — the module RNG is
    shared by every caller and ordering effects across modules
    silently couple their decision sequences.

    Pass ``_visible`` to reuse a pre-computed visibility list and
    skip the access-model recomputation.
    """
    if _visible is None:
        _visible = _ACCESS_MODEL.compute_access(
            src.lat_deg, src.lon_deg, 0.0, sat_positions, DEFAULT_MIN_ELEVATION_DEG,
        )
    if not _visible:
        return None
    if top_prob >= 1.0:
        return _visible[0]
    rng_eff = rng if rng is not None else _RNG
    if rng_eff.random() < top_prob:
        return _visible[0]
    return rng_eff.choice(_visible)
