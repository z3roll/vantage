"""Ground delay estimation: measurement-based and geographic.

Two concrete implementations of :class:`GroundDelay`:

    * :class:`MeasuredGroundDelay` — backed by traceroute measurements
      under ``data/probe_trace/traceroute/``. Strict: raises
      :class:`KeyError` on unknown pairs.
    * :class:`GeographicGroundDelay` — estimates RTT from the PoP to
      the nearest service node (loaded from
      ``config/service_prefixes.json``). Never raises — always has an
      estimate.

All values are **one-way** RTT in milliseconds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from vantage.common import C_FIBER_KM_S, haversine_km

__all__ = [
    "GeographicGroundDelay",
    "GroundDelay",
    "MeasuredGroundDelay",
]


class GroundDelay(Protocol):
    """Protocol for ground segment delay lookup (PoP → destination).

    Returns a one-way ground RTT in ms.  Raises :class:`KeyError` if
    no estimate is available for the pair.
    """

    def estimate(self, pop_code: str, dest_name: str) -> float: ...


# ---------------------------------------------------------------------------
# MeasuredGroundDelay (traceroute-backed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeasuredGroundDelay:
    """Ground delay backed by a fixed ``(pop, destination) → RTT`` table.

    Construct via :meth:`from_traceroute_dir` for real experiments, or
    via the direct constructor for unit tests with hand-injected data.
    All values are one-way RTT in ms.
    """

    one_way_rtt_ms: Mapping[tuple[str, str], float]

    def __post_init__(self) -> None:
        if not isinstance(self.one_way_rtt_ms, MappingProxyType):
            object.__setattr__(
                self, "one_way_rtt_ms", MappingProxyType(dict(self.one_way_rtt_ms))
            )

    def estimate(self, pop_code: str, dest_name: str) -> float:
        """Return one-way ground delay (ms) or raise :class:`KeyError`."""
        try:
            return self.one_way_rtt_ms[(pop_code, dest_name)]
        except KeyError:
            raise KeyError(
                f"no measured ground RTT for (pop={pop_code!r}, dest={dest_name!r})"
            ) from None

    def has(self, pop_code: str, dest_name: str) -> bool:
        return (pop_code, dest_name) in self.one_way_rtt_ms

    def pops(self) -> frozenset[str]:
        return frozenset(pop for (pop, _) in self.one_way_rtt_ms)

    def destinations(self) -> frozenset[str]:
        return frozenset(dst for (_, dst) in self.one_way_rtt_ms)

    def __len__(self) -> int:
        return len(self.one_way_rtt_ms)

    @classmethod
    def empty(cls) -> MeasuredGroundDelay:
        return cls(one_way_rtt_ms=MappingProxyType({}))

    @classmethod
    def from_traceroute_dir(
        cls,
        traceroute_dir: str | Path,
        services: tuple[str, ...] = ("google", "facebook", "wikipedia"),
        *,
        require_all_services: bool = True,
    ) -> MeasuredGroundDelay:
        """Build from per-service traceroute summary JSON files."""
        traceroute_dir = Path(traceroute_dir)
        if not traceroute_dir.exists():
            raise FileNotFoundError(f"traceroute directory not found: {traceroute_dir}")

        raw: dict[tuple[str, str], list[float]] = {}
        missing: list[str] = []

        for service in services:
            summary_path = traceroute_dir / f"{service}_summary.json"
            if not summary_path.exists():
                missing.append(service)
                continue
            with summary_path.open() as f:
                data = json.load(f)
            for entry in data.values():
                if not entry or not entry.get("pop") or not entry.get("summary"):
                    continue
                avg_vals = entry["summary"].get("average_values")
                if not avg_vals:
                    continue
                pop_code = entry["pop"].get("code")
                if not pop_code:
                    continue
                rt_ms = avg_vals.get("avg_ground_segment")
                if rt_ms is None or rt_ms <= 0:
                    continue
                raw.setdefault((pop_code, service), []).append(float(rt_ms))

        if missing == list(services):
            raise FileNotFoundError(
                f"no traceroute summaries in {traceroute_dir}: "
                f"looked for {', '.join(f'{s}_summary.json' for s in services)}"
            )
        if require_all_services and missing:
            raise FileNotFoundError(
                f"missing service files in {traceroute_dir}: "
                f"{', '.join(f'{s}_summary.json' for s in missing)}"
            )

        one_way = {
            key: sum(samples) / len(samples) / 2.0
            for key, samples in raw.items()
        }
        if not one_way:
            raise ValueError(
                f"traceroute directory {traceroute_dir} produced no valid samples"
            )
        return cls(one_way_rtt_ms=MappingProxyType(one_way))


# ---------------------------------------------------------------------------
# GeographicGroundDelay (service location-based)
# ---------------------------------------------------------------------------


class GeographicGroundDelay:
    """Estimate ground delay from PoP to nearest service node.

    For each ``(pop, dest)`` pair, finds the closest node of that
    service and returns ``haversine_distance × detour_factor / c_fiber``.

    Data loaded from ``config/service_prefixes.json`` (service →
    locations) and PoP coordinates from :class:`GroundInfrastructure`.

    Implements :class:`GroundDelay` — same ``estimate(pop, dest)``
    interface as :class:`MeasuredGroundDelay`, but never raises
    :class:`KeyError`. Falls back to a configurable default RTT for
    unknown services.
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


# ---------------------------------------------------------------------------
# TracerouteReplayDelay (time-based real measurements)
# ---------------------------------------------------------------------------


class TracerouteReplayDelay:
    """Ground delay from real traceroute measurements, indexed by time-of-day.

    Loads per-probe detailed_report from summary JSON files. Groups
    measurements by (PoP, service, hour-of-day). At runtime, returns
    a random measurement from the matching hour bucket.

    Implements :class:`GroundDelay` — call ``set_time(epoch_s)`` each
    epoch to set the simulation clock before calling ``estimate()``.
    """

    def __init__(
        self,
        hourly_data: dict[tuple[str, str, int], list[float]],
        fallback: GroundDelay | None = None,
        seed: int = 42,
    ) -> None:
        self._hourly = hourly_data  # (pop, dest, hour) → [one_way_ms, ...]
        self._fallback = fallback
        self._rng = __import__("random").Random(seed)
        self._current_hour: int = 0

    def set_time(self, epoch_s: float) -> None:
        """Set simulation UTC time (seconds). Called each epoch."""
        self._current_hour = int(epoch_s / 3600) % 24

    def estimate(self, pop_code: str, dest_name: str) -> float:
        """One-way ground delay (ms) from the matching hour bucket."""
        bucket = self._hourly.get((pop_code, dest_name, self._current_hour))
        if bucket:
            return self._rng.choice(bucket)
        # Try any hour for this (pop, dest)
        for h in range(24):
            bucket = self._hourly.get((pop_code, dest_name, h))
            if bucket:
                return self._rng.choice(bucket)
        # Fallback to geographic estimator
        if self._fallback is not None:
            return self._fallback.estimate(pop_code, dest_name)
        return 20.0

    def has(self, pop_code: str, dest_name: str) -> bool:
        return any(
            (pop_code, dest_name, h) in self._hourly for h in range(24)
        )

    def pops(self) -> frozenset[str]:
        return frozenset(p for p, _, _ in self._hourly)

    def destinations(self) -> frozenset[str]:
        return frozenset(d for _, d, _ in self._hourly)

    @classmethod
    def from_traceroute_dir(
        cls,
        traceroute_dir: str | Path,
        services: tuple[str, ...] = ("google", "facebook", "wikipedia"),
        fallback: GroundDelay | None = None,
    ) -> TracerouteReplayDelay:
        """Build from raw traceroute JSON files (with timestamps).

        Reads ``{traceroute_dir}/{service}/{probe_id}_{msm_id}.json``
        and groups ground_segment RTT by (pop_code, service, hour_of_day).
        """
        from datetime import datetime, timezone

        traceroute_dir = Path(traceroute_dir)
        hourly: dict[tuple[str, str, int], list[float]] = {}

        for service in services:
            # Load summary to get probe → pop mapping
            summary_path = traceroute_dir / f"{service}_summary.json"
            if not summary_path.exists():
                continue
            with summary_path.open() as f:
                summary = json.load(f)

            probe_pop: dict[str, str] = {}
            for probe_id, entry in summary.items():
                if not entry:
                    continue
                pop = (entry.get("pop") or {}).get("code")
                if pop:
                    probe_pop[probe_id] = pop

            # Load raw measurements with timestamps
            raw_dir = traceroute_dir / service
            if not raw_dir.is_dir():
                continue
            for fname in raw_dir.iterdir():
                if not fname.suffix == ".json":
                    continue
                probe_id = fname.stem.split("_")[0]
                pop_code = probe_pop.get(probe_id)
                if not pop_code:
                    continue

                with fname.open() as f:
                    measurements = json.load(f)

                for m in measurements:
                    ts = m.get("timestamp")
                    if ts is None:
                        continue
                    # Extract ground segment from the detailed report
                    # Raw files have full traceroute; need to match with summary
                    # For now use the summary's detailed_report which has ground_segment
                    pass

            # Use detailed_report from summary (already has ground_segment per measurement)
            for probe_id, entry in summary.items():
                if not entry:
                    continue
                pop_code = (entry.get("pop") or {}).get("code")
                if not pop_code:
                    continue
                reports = entry.get("detailed_report", [])

                # We don't have timestamps in detailed_report, but raw files do.
                # Load raw file to get timestamps, pair with detailed_report indices.
                raw_files = list((traceroute_dir / service).glob(f"{probe_id}_*.json"))
                if not raw_files:
                    # No timestamp data — spread measurements evenly across 24h
                    n = len(reports)
                    for i, r in enumerate(reports):
                        gs = r.get("ground_segment", 0)
                        if gs <= 0:
                            continue
                        hour = int(i / n * 24) % 24
                        key = (pop_code, service, hour)
                        hourly.setdefault(key, []).append(gs / 2)  # RT → one-way
                    continue

                with raw_files[0].open() as f:
                    raw_measurements = json.load(f)

                # Pair by index (both lists ordered by time)
                for i, r in enumerate(reports):
                    gs = r.get("ground_segment", 0)
                    if gs <= 0 or i >= len(raw_measurements):
                        continue
                    ts = raw_measurements[i].get("timestamp", 0)
                    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
                    key = (pop_code, service, hour)
                    hourly.setdefault(key, []).append(gs / 2)

        return cls(hourly_data=hourly, fallback=fallback)
