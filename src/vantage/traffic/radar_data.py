"""Radar data loaders: hourly demand weights and service-class mix schedules."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class PopHourlyDemand:
    """Hourly demand weights per PoP from Cloudflare Radar.

    Indexed by (pop_code, day_type, local_hour) -> weight in [0, 1].
    Also stores PoP timezone mapping.
    """

    weights: Mapping[tuple[str, str, int], float]
    timezones: Mapping[str, str]  # pop_code -> IANA timezone

    def weight(self, pop_code: str, day_type: str, local_hour: int) -> float:
        """Get demand weight. Returns 0.5 (mid-level) if not found."""
        return self.weights.get((pop_code, day_type, local_hour), 0.5)

    def timezone(self, pop_code: str) -> str:
        """Get PoP timezone. Returns UTC if not found."""
        return self.timezones.get(pop_code, "UTC")

    @staticmethod
    def from_csv(path: str | Path) -> PopHourlyDemand:
        """Load from pop_hourly_demand_radar.csv.

        Args:
            path: Path to CSV with columns: pop_code, timezone,
                  day_type, local_hour, demand_weight.
        """
        weights: dict[tuple[str, str, int], float] = {}
        timezones: dict[str, str] = {}
        with Path(path).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                pop_code = row["pop_code"]
                day_type = row["day_type"]
                local_hour = int(row["local_hour"])
                demand_weight = float(row["demand_weight"])
                weights[(pop_code, day_type, local_hour)] = demand_weight
                if pop_code not in timezones:
                    timezones[pop_code] = row["timezone"]
        return PopHourlyDemand(
            weights=MappingProxyType(weights),
            timezones=MappingProxyType(timezones),
        )


@dataclass(frozen=True, slots=True)
class ServiceMixSchedule:
    """Hourly service-class traffic mix baseline.

    Indexed by (day_type, local_hour) -> {service_class -> weight}.
    """

    mix: Mapping[tuple[str, int], Mapping[str, float]]

    def get_mix(self, day_type: str, local_hour: int) -> Mapping[str, float]:
        """Get service mix for a given time. Returns empty mapping if not found."""
        return self.mix.get((day_type, local_hour), MappingProxyType({}))

    @staticmethod
    def from_csv(path: str | Path) -> ServiceMixSchedule:
        """Load from service_class_hourly_mix_baseline.csv.

        Args:
            path: Path to CSV with columns: day_type, local_hour,
                  service_class, traffic_weight.
        """
        raw: dict[tuple[str, int], dict[str, float]] = {}
        with Path(path).open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                day_type = row["day_type"]
                local_hour = int(row["local_hour"])
                service_class = row["service_class"]
                weight = float(row["traffic_weight"])
                key = (day_type, local_hour)
                if key not in raw:
                    raw[key] = {}
                raw[key][service_class] = weight
        frozen = {k: MappingProxyType(v) for k, v in raw.items()}
        return ServiceMixSchedule(mix=MappingProxyType(frozen))
