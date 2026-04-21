"""Ground delay estimation: geographic (distance-based) model.

Single concrete implementation of the :class:`GroundDelay` protocol:

    * :class:`GeographicGroundDelay` — estimates RTT from each PoP to
      the nearest service node (loaded from
      ``config/service_prefixes.json``). Never raises — always has an
      estimate, falling back to a configurable default for unknown
      services.

All values are **one-way** RTT in milliseconds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from vantage.common import C_FIBER_KM_S, haversine_km

__all__ = [
    "GeographicGroundDelay",
    "GroundDelay",
]


class GroundDelay(Protocol):
    """Protocol for ground segment delay lookup (PoP → destination).

    Returns a one-way ground RTT in ms.  Raises :class:`KeyError` if
    no estimate is available for the pair.
    """

    def estimate(self, pop_code: str, dest_name: str) -> float: ...


# ---------------------------------------------------------------------------
# GeographicGroundDelay (service location-based)
# ---------------------------------------------------------------------------


class GeographicGroundDelay:
    """Estimate ground delay from PoP to nearest service node.

    For each ``(pop, dest)`` pair, finds the closest node of that
    service and returns ``haversine_distance × detour_factor / c_fiber``.

    Data loaded from ``config/service_prefixes.json`` (service →
    locations) and PoP coordinates from :class:`GroundInfrastructure`.

    Falls back to a configurable default RTT for unknown services.
    """

    def __init__(
        self,
        pop_coords: Mapping[str, tuple[float, float]],
        service_locations: Mapping[str, list[dict]],
        *,
        detour_factor: float = 1.4,
        base_ms: float = 5.0,
        jitter_sigma: float = 0.3,
        default_one_way_ms: float = 20.0,
        seed: int = 42,
    ) -> None:
        self._pop_coords = pop_coords
        self._service_locs = service_locations
        self._detour = detour_factor
        self._base = base_ms
        self._sigma = jitter_sigma
        self._default = default_one_way_ms
        self._rng = __import__("random").Random(seed)
        # Pre-compute (pop, dest) → (mu, sigma) for LogNormal sampling
        self._distributions: dict[tuple[str, str], tuple[float, float]] = {}
        self._precompute()

    def _precompute(self) -> None:
        """Build LogNormal(μ, σ) for every (pop, dest) pair.

        median = base_ms + distance_delay  (base models routing/processing)
        σ = jitter_sigma                    (models real-world variance)
        """
        import math
        for pop_code, (pop_lat, pop_lon) in self._pop_coords.items():
            for dest_name, locs in self._service_locs.items():
                min_dist = min(
                    haversine_km(pop_lat, pop_lon, loc["lat"], loc["lon"])
                    for loc in locs
                )
                distance_ms = min_dist * self._detour / C_FIBER_KM_S * 1000.0
                median = self._base + distance_ms
                mu = math.log(max(0.1, median))
                self._distributions[(pop_code, dest_name)] = (mu, self._sigma)

    def estimate(self, pop_code: str, dest_name: str) -> float:
        """Sample one-way ground delay (ms) from LogNormal distribution."""
        import math
        params = self._distributions.get((pop_code, dest_name))
        if params is None:
            return self._default
        mu, sigma = params
        return math.exp(self._rng.gauss(mu, sigma))

    def has(self, pop_code: str, dest_name: str) -> bool:
        return pop_code in self._pop_coords and dest_name in self._service_locs

    def pops(self) -> frozenset[str]:
        return frozenset(self._pop_coords.keys())

    def destinations(self) -> frozenset[str]:
        return frozenset(self._service_locs.keys())

    def __len__(self) -> int:
        return len(self._pop_coords) * len(self._service_locs)

    @classmethod
    def from_config(
        cls,
        config_dir: str | Path,
        pop_coords: Mapping[str, tuple[float, float]],
        *,
        detour_factor: float = 1.4,
        default_one_way_ms: float = 20.0,
    ) -> GeographicGroundDelay:
        """Load from ``config/service_prefixes.json``.

        Args:
            config_dir: Directory containing ``service_prefixes.json``.
            pop_coords: ``{pop_code: (lat, lon)}`` mapping from
                :class:`GroundInfrastructure`.
        """
        path = Path(config_dir) / "service_prefixes.json"
        with path.open() as f:
            raw = json.load(f)

        service_locs: dict[str, list[dict]] = {}
        for svc_name, svc_data in raw.items():
            locs = svc_data.get("locations", [])
            if locs:
                service_locs[svc_name] = locs

        return cls(
            pop_coords=pop_coords,
            service_locations=service_locs,
            detour_factor=detour_factor,
            default_one_way_ms=default_one_way_ms,
        )
