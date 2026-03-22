"""Traffic subsystem: demand generation and endpoint population.

Backward-compatible re-exports — existing ``from vantage.traffic import X``
statements continue to work.
"""

from __future__ import annotations

from typing import Protocol

from vantage.domain import TrafficDemand
from vantage.traffic.generators import GravityGenerator, UniformGenerator
from vantage.traffic.population import (
    DEFAULT_DESTINATIONS,
    EndpointPopulation,
)


class TrafficGenerator(Protocol):
    """Protocol for traffic demand generation."""

    def generate(self, epoch: int) -> TrafficDemand:
        """Generate traffic demand for a given epoch."""
        ...


__all__ = [
    "DEFAULT_DESTINATIONS",
    "EndpointPopulation",
    "GravityGenerator",
    "TrafficGenerator",
    "UniformGenerator",
]
