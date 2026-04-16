"""Traffic subsystem: demand generation and endpoint population."""

from __future__ import annotations

from typing import Protocol

from vantage.domain import TrafficDemand
from vantage.traffic.generators import (
    FlowLevelGenerator,
    GravityGenerator,
    RealisticGenerator,
    UniformGenerator,
)
from vantage.traffic.population import CityGroup, EndpointPopulation


class TrafficGenerator(Protocol):
    """Protocol for traffic demand generation."""

    def generate(self, epoch: int) -> TrafficDemand: ...


__all__ = [
    "CityGroup",
    "EndpointPopulation",
    "FlowLevelGenerator",
    "GravityGenerator",
    "RealisticGenerator",
    "TrafficGenerator",
    "UniformGenerator",
]
