"""Time-varying service-class traffic generator driven by Radar data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from vantage.domain import FlowKey, TrafficDemand

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vantage.traffic.radar_data import PopHourlyDemand, ServiceMixSchedule
    from vantage.traffic.service_population import ServiceClassPopulation


class TimeVaryingServiceMixGenerator:
    """Generates terminal->service_class flows with time-varying intensity.

    Demand for each (terminal, service_class) at each epoch:
        demand = base_demand_gbps
                 * demand_weight(pop, day_type, hour)
                 * service_mix(day_type, hour, class)

    Where:
    - demand_weight: diurnal profile from Cloudflare Radar (0~1 per PoP per hour)
    - service_mix: hourly service-class traffic share (sums to ~1 per hour)
    """

    def __init__(
        self,
        population: ServiceClassPopulation,
        pop_hourly_demand: PopHourlyDemand,
        service_mix: ServiceMixSchedule,
        terminal_pop_mapping: Mapping[str, str],
        base_demand_gbps: float = 0.01,
        simulation_start_utc: datetime | None = None,
        epoch_interval_s: float = 3600.0,
    ) -> None:
        self._population = population
        self._pop_demand = pop_hourly_demand
        self._service_mix = service_mix
        self._terminal_pop = terminal_pop_mapping
        self._base_demand = base_demand_gbps
        self._start_utc = simulation_start_utc or datetime(
            2026, 3, 23, 0, 0, 0, tzinfo=UTC
        )  # default: a Monday 00:00 UTC
        self._interval_s = epoch_interval_s

    def _local_time(self, epoch: int, tz_str: str) -> tuple[int, str]:
        """Convert epoch number to (local_hour, day_type) for a timezone.

        Args:
            epoch: Simulation epoch index (0-based).
            tz_str: IANA timezone string, e.g. "Pacific/Auckland".

        Returns:
            Tuple of (local_hour 0-23, day_type "weekday"|"weekend").
        """
        utc_time = self._start_utc + timedelta(seconds=epoch * self._interval_s)
        local_time = utc_time.astimezone(ZoneInfo(tz_str))
        local_hour = local_time.hour
        day_type = "weekday" if local_time.weekday() < 5 else "weekend"
        return local_hour, day_type

    def generate(self, epoch: int) -> TrafficDemand:
        """Generate traffic demand for a given epoch.

        For each source terminal, looks up its PoP, converts epoch to
        local time, then computes demand per service class using diurnal
        weight and service mix.

        Args:
            epoch: Simulation epoch index (0-based).

        Returns:
            TrafficDemand with flows keyed by (terminal_name, service_class).
        """
        flows: dict[FlowKey, float] = {}

        for src in self._population.sources:
            pop_code = self._terminal_pop.get(src.name)
            if pop_code is None:
                continue

            tz_str = self._pop_demand.timezone(pop_code)
            local_hour, day_type = self._local_time(epoch, tz_str)

            demand_weight = self._pop_demand.weight(pop_code, day_type, local_hour)
            service_mix = self._service_mix.get_mix(day_type, local_hour)

            for svc, svc_weight in service_mix.items():
                flow_demand = self._base_demand * demand_weight * svc_weight
                if flow_demand > 0:
                    flows[FlowKey(src=src.name, dst=svc)] = flow_demand

        return TrafficDemand(epoch=epoch, flows=MappingProxyType(flows))
