"""Traffic domain types.

All delay values in ms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A traffic source or destination."""

    name: str
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True)
class FlowKey:
    """Identifies a traffic aggregate: source endpoint → destination endpoint."""

    src: str  # endpoint name
    dst: str  # endpoint name


@dataclass(frozen=True, slots=True)
class TrafficDemand:
    """Traffic demand for one epoch: mapping of FlowKey → demand in Gbps."""

    epoch: int
    flows: Mapping[FlowKey, float]  # FlowKey → demand_gbps
