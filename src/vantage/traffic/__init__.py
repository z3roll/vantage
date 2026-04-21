"""Traffic subsystem: demand generation and endpoint population."""

from __future__ import annotations

from vantage.traffic.generators import FlowLevelGenerator
from vantage.traffic.population import CityGroup, EndpointPopulation

__all__ = [
    "CityGroup",
    "EndpointPopulation",
    "FlowLevelGenerator",
]
