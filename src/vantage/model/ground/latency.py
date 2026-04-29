"""Ground delay estimation — deterministic geographic prior.

Two concepts live here:

    * :class:`GroundDelay` — Protocol for a one-way RTT lookup. The
      contract is now "deterministic, stateless, no per-call jitter";
      anything that wants realistic epoch-to-epoch variation goes
      through :class:`GroundTruth` instead.
    * :class:`GeographicGroundDelay` — the single concrete prior.
      Returns ``base_ms + haversine_km(pop → nearest service node) ×
      detour_factor / c_fiber`` as a **one-way RTT (ms)**. Falls
      back to a configurable default for unknown services.

Values do NOT change across runs or across calls. Reproducibility is
free: the same ``(pop, dest)`` gives the same number forever.
"""

from __future__ import annotations

import json
import math
import random as _random
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from vantage.common import C_FIBER_KM_S, haversine_km
from vantage.common.seed import mix_seed

__all__ = [
    "GeographicGroundDelay",
    "GroundDelay",
    "GroundPrior",
    "GroundTruth",
]


class GroundDelay(Protocol):
    """Protocol for a deterministic one-way RTT lookup (PoP → destination).

    Returns a one-way ground RTT in ms. Raises :class:`KeyError` if
    no estimate is available for the pair. Implementations must be
    pure — the same input always returns the same output so the
    planner can treat this as a fixed prior.
    """

    def estimate(self, pop_code: str, dest_name: str) -> float: ...


class GroundPrior(Protocol):
    """Deterministic one-way RTT (ms) from a PoP to a destination."""

    def estimate(self, pop_code: str, dest_name: str) -> float: ...


class GeographicGroundDelay:
    """Deterministic distance-based one-way RTT prior.

    ``one_way_ms(pop, dest) = base_ms + min_distance_km · detour / c_fiber``

    Previously this class sampled a LogNormal per ``(pop, dest)`` so
    it doubled as both the cold-start prior AND the run-level truth.
    That coupling made it impossible for the planner to learn from
    observations without looking at the same RNG that produced the
    "truth" it was trying to predict. Truth now lives in
    :class:`GroundTruth`; this class is just a flat distance model.
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


class GroundTruth:
    """Per-flow, per-epoch ground-truth RTT sampler."""

    __slots__ = ("_prior", "_seed_base", "_sigma", "_cur_epoch", "_samples")

    def __init__(
        self,
        prior: GroundPrior,
        seed_base: int,
        *,
        sigma: float = 0.3,
    ) -> None:
        self._prior = prior
        self._seed_base = int(seed_base)
        self._sigma = float(sigma)
        self._cur_epoch: int = -1
        self._samples: dict[tuple[str, str, str], float] = {}

    @property
    def seed_base(self) -> int:
        return self._seed_base

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self, pop_code: str, dest: str, epoch: int, flow_id: str) -> float:
        if epoch != self._cur_epoch:
            self._samples.clear()
            self._cur_epoch = epoch
        key = (pop_code, dest, flow_id)
        cached = self._samples.get(key)
        if cached is not None:
            return cached
        one_way_median = float(self._prior.estimate(pop_code, dest))
        if not math.isfinite(one_way_median) or one_way_median <= 0:
            raise ValueError(
                f"GroundTruth: prior returned non-positive median "
                f"{one_way_median} for (pop={pop_code!r}, dest={dest!r})"
            )
        median_rtt = 2.0 * one_way_median
        rng = _random.Random(mix_seed(self._seed_base, epoch, pop_code, dest, flow_id))
        value = math.exp(rng.gauss(math.log(median_rtt), self._sigma))
        self._samples[key] = value
        return value
