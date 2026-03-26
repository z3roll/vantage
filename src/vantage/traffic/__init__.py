"""Traffic subsystem: demand generation and endpoint population.

Backward-compatible re-exports — existing ``from vantage.traffic import X``
statements continue to work.
"""

from __future__ import annotations

from typing import Protocol

from vantage.domain import TrafficDemand
from vantage.traffic.generators import GravityGenerator, RealisticGenerator, UniformGenerator
from vantage.traffic.population import (
    DEFAULT_DESTINATIONS,
    EndpointPopulation,
)
from vantage.traffic.radar_data import PopHourlyDemand, ServiceMixSchedule
from vantage.traffic.service_population import ServiceClassPopulation
from vantage.traffic.time_varying import TimeVaryingServiceMixGenerator


class TrafficGenerator(Protocol):
    """Protocol for traffic demand generation."""

    def generate(self, epoch: int) -> TrafficDemand:
        """Generate traffic demand for a given epoch."""
        ...


__all__ = [
    "DEFAULT_DESTINATIONS",
    "EndpointPopulation",
    "GravityGenerator",
    "PopHourlyDemand",
    "RealisticGenerator",
    "ServiceClassPopulation",
    "ServiceMixSchedule",
    "TimeVaryingServiceMixGenerator",
    "TrafficGenerator",
    "UniformGenerator",
]
