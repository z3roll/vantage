"""Shared immutable domain objects for Vantage TE system."""

from vantage.domain.ground import GroundStation, GSPoPEdge, PoP
from vantage.domain.result import (
    CostTables,
    EpochResult,
    FlowOutcome,
    PathAllocation,
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
from vantage.domain.service import SERVICE_CLASSES
from vantage.domain.snapshot import (
    InfrastructureView,
    NetworkSnapshot,
)
from vantage.domain.traffic import Endpoint, FlowKey, TrafficDemand

__all__ = [
    "AccessLink",
    "ConstellationConfig",
    "CostTables",
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
    "SERVICE_CLASSES",
    "SatelliteState",
    "ShellConfig",
    "TrafficDemand",
]
