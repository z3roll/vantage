"""Forwarding result domain types.

All delay/RTT values in ms. Bandwidth values in Gbps.
"""

from __future__ import annotations

from dataclasses import dataclass

from vantage.domain.traffic import FlowKey


@dataclass(frozen=True, slots=True)
class PathAllocation:
    """Fully resolved single path for a flow (internal use by forward)."""

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int


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


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Complete forwarding result for one epoch."""

    epoch: int
    flow_outcomes: tuple[FlowOutcome, ...]
    total_demand_gbps: float
    routed_demand_gbps: float
    unrouted_demand_gbps: float


@dataclass(frozen=True, slots=True)
class SLAViolation:
    """One flow's SLA was not met: its served rate fell below its CIR.

    SLA semantics: a flow meets its SLA when the post-throttle served
    rate is ≥ its committed information rate (CIR). This type records
    one violation with enough context to compute severity in
    ``[0.0, 1.0]`` — ``0`` means "just barely below CIR",
    ``1`` means "entirely blocked".
    """

    flow_key: FlowKey
    demand_gbps: float
    served_gbps: float
    cir_gbps: float

    @property
    def shortfall_gbps(self) -> float:
        """How far below CIR the flow landed."""
        return max(0.0, self.cir_gbps - self.served_gbps)

    @property
    def severity(self) -> float:
        """Fraction of the CIR missed, in ``[0.0, 1.0]``."""
        if self.cir_gbps <= 0:
            return 0.0
        return min(1.0, self.shortfall_gbps / self.cir_gbps)
