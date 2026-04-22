"""Ground delay estimation — deterministic geographic prior.

Two concepts live here:

    * :class:`GroundDelay` — Protocol for a one-way RTT lookup. The
      contract is now "deterministic, stateless, no per-call jitter";
      anything that wants realistic epoch-to-epoch variation goes
      through :class:`vantage.world.ground.truth.GroundTruth` instead.
    * :class:`GeographicGroundDelay` — the single concrete prior.
      Returns ``base_ms + haversine_km(pop → nearest service node) ×
      detour_factor / c_fiber`` as a **one-way RTT (ms)**. Falls
      back to a configurable default for unknown services.

Values do NOT change across runs or across calls. Reproducibility is
free: the same ``(pop, dest)`` gives the same number forever.
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
    """Protocol for a deterministic one-way RTT lookup (PoP → destination).

    Returns a one-way ground RTT in ms. Raises :class:`KeyError` if
    no estimate is available for the pair. Implementations must be
    pure — the same input always returns the same output so the
    planner can treat this as a fixed prior.
    """

    def estimate(self, pop_code: str, dest_name: str) -> float: ...


class GeographicGroundDelay:
    """Deterministic distance-based one-way RTT prior.

    ``one_way_ms(pop, dest) = base_ms + min_distance_km · detour / c_fiber``

    Previously this class sampled a LogNormal per ``(pop, dest)`` so
    it doubled as both the cold-start prior AND the run-level truth.
    That coupling made it impossible for the planner to learn from
    observations without looking at the same RNG that produced the
    "truth" it was trying to predict. The refactor pulls truth out
    into :class:`vantage.world.ground.truth.GroundTruth`; this class
    is now just a flat distance model.
    """

    def __init__(
        self,
        pop_coords: Mapping[str, tuple[float, float]],
        service_locations: Mapping[str, list[dict]],
        *,
        detour_factor: float = 1.4,
        base_ms: float = 5.0,
        default_one_way_ms: float = 20.0,
    ) -> None:
        self._pop_coords = pop_coords
        self._service_locs = service_locations
        self._detour = detour_factor
        self._base = base_ms
        self._default = default_one_way_ms
        # Pre-compute one-way RTT for every ``(pop, dest)`` pair.
        # Everything is a fixed function of the static inputs; no RNG.
        self._one_way_ms: dict[tuple[str, str], float] = {}
        self._precompute()

    def _precompute(self) -> None:
        for pop_code, (pop_lat, pop_lon) in self._pop_coords.items():
            for dest_name, locs in self._service_locs.items():
                if not locs:
                    continue
                min_dist_km = min(
                    haversine_km(pop_lat, pop_lon, loc["lat"], loc["lon"])
                    for loc in locs
                )
                distance_ms = min_dist_km * self._detour / C_FIBER_KM_S * 1000.0
                self._one_way_ms[(pop_code, dest_name)] = self._base + distance_ms

    def estimate(self, pop_code: str, dest_name: str) -> float:
        """Return the deterministic one-way RTT (ms) for ``(pop, dest)``.

        Unknown services fall back to ``default_one_way_ms`` so callers
        always get a finite number — consistent with the pre-refactor
        contract.
        """
        return self._one_way_ms.get((pop_code, dest_name), self._default)

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
        """Load from ``config/service_prefixes.json``."""
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
