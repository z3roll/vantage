"""Forwarding result domain types.

All delay/RTT values in ms. Bandwidth values in Gbps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from vantage.traffic.types import FlowKey


@dataclass(frozen=True, slots=True)
class ResolvedFlow:
    """Outcome of resolving a single flow through a forward strategy."""

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int
    satellite_rtt: float
    ground_rtt: float
    propagation_rtt: float = 0.0
    queuing_rtt: float = 0.0
    transmission_rtt: float = 0.0
    loss_probability: float = 0.0
    bottleneck_gbps: float = 0.0


@dataclass(frozen=True, slots=True)
class FlowOutcome:
    """Realized outcome for a single flow. All RTT values in ms.

    satellite_rtt: Terminal→PoP total (propagation + queuing + transmission).
    ground_rtt: PoP→Destination.
    """

    flow_key: FlowKey
    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int
    satellite_rtt: float   # terminal→PoP, total including queuing (ms)
    ground_rtt: float      # PoP→destination (ms)
    total_rtt: float       # satellite + ground (ms)
    demand_gbps: float
    # --- link performance fields (defaults preserve backward compat) ---
    propagation_rtt: float = 0.0     # pure propagation portion of satellite_rtt (ms)
    queuing_rtt: float = 0.0         # total queuing delay along sat path, RTT (ms)
    transmission_rtt: float = 0.0    # total serialization delay along sat path, RTT (ms)
    loss_probability: float = 0.0    # end-to-end packet loss probability [0,1]
    bottleneck_gbps: float = 0.0     # min link capacity along the sat path (Gbps)
    effective_throughput_gbps: float = 0.0  # demand × (1−loss), capped by bottleneck


_EMPTY_TIMING: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Complete forwarding result for one epoch."""

    epoch: int
    flow_outcomes: tuple[FlowOutcome, ...]
    total_demand_gbps: float
    routed_demand_gbps: float
    unrouted_demand_gbps: float
    # Per-phase wall-clock timings (ms) for the forwarding pipeline of
    # this epoch. Always carries ``total_ms`` + the four realize()
    # phases ``ingress_ms`` / ``decide_ms`` / ``charge_ms`` /
    # ``measure_ms``; empty when a caller constructs an
    # :class:`EpochResult` outside :func:`vantage.forward.realize`.
    forward_timing_ms: Mapping[str, float] = field(
        default_factory=lambda: _EMPTY_TIMING,
    )
