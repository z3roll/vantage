"""Forwarding result domain types.

All delay/RTT values in ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vantage.domain.traffic import FlowKey


@dataclass(frozen=True, slots=True)
class PathAllocation:
    """Fully resolved single path for a flow."""

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int


@dataclass(frozen=True, slots=True)
class RoutingIntent:
    """Controller output: one fully resolved path per flow per epoch."""

    epoch: int
    allocations: Mapping[FlowKey, PathAllocation]


@dataclass(frozen=True, slots=True)
class FlowOutcome:
    """Realized outcome for a single flow. All RTT values in ms.

    satellite_rtt: Terminal→PoP (uplink + ISL + downlink + backhaul), calibrated.
    ground_rtt: PoP→Destination.
    """

    flow_key: FlowKey
    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int
    satellite_rtt: float   # terminal→PoP, calibrated (ms)
    ground_rtt: float      # PoP→destination (ms)
    total_rtt: float       # satellite + ground (ms)
    demand_gbps: float


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Complete forwarding result for one epoch."""

    epoch: int
    flow_outcomes: tuple[FlowOutcome, ...]
    total_demand_gbps: float
    routed_demand_gbps: float
    unrouted_demand_gbps: float
