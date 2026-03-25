"""Forwarding result and cost table domain types.

All delay/RTT values in ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vantage.domain.traffic import FlowKey


@dataclass(frozen=True, slots=True)
class PathAllocation:
    """Fully resolved single path for a flow (internal use by forward)."""

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int


@dataclass(frozen=True, slots=True)
class CostTables:
    """Controller output: precomputed cost tables for terminal-side PoP selection.

    sat_cost: (ingress_sat, pop_code) → min satellite segment RTT (ISL + downlink + backhaul).
    ground_cost: (pop_code, dest) → ground segment RTT.

    Terminals select best PoP via: argmin over pop: sat_cost[ingress, pop] + ground_cost[pop, dest]
    """

    epoch: int
    sat_cost: Mapping[tuple[int, str], float]  # (ingress_sat, pop_code) → RTT ms
    ground_cost: Mapping[tuple[str, str], float]  # (pop_code, dest) → RTT ms


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
