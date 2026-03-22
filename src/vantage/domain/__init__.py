"""Shared immutable domain objects for Vantage TE system."""

from vantage.domain.ground import GroundStation, GSPoPEdge, PoP
from vantage.domain.result import (
    EpochResult,
    FlowOutcome,
    PathAllocation,
    RoutingIntent,
)
from vantage.domain.satellite import (
    AccessLink,
    ConstellationConfig,
    GatewayAttachments,
    ISLEdge,
    ISLGraph,
    SatelliteState,
    ShellConfig,
)
from vantage.domain.snapshot import (
    InfrastructureView,
    NetworkSnapshot,
)
from vantage.domain.traffic import Endpoint, FlowKey, TrafficDemand

__all__ = [
    "AccessLink",
    "ConstellationConfig",
    "Endpoint",
    "EpochResult",
    "FlowKey",
    "FlowOutcome",
    "GSPoPEdge",
    "GatewayAttachments",
    "GroundStation",
    "ISLEdge",
    "ISLGraph",
    "InfrastructureView",
    "NetworkSnapshot",
    "PathAllocation",
    "PoP",
    "RoutingIntent",
    "SatelliteState",
    "ShellConfig",
    "TrafficDemand",
]
