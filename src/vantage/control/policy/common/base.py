"""Base controller using the candidate + scorer framework.

Subclasses only need to implement _select_pop() or override _select_alloc()
to define their PoP selection strategy. Path enumeration and selection
are handled by the base class.
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.candidate import (
    enumerate_all_candidates,
    enumerate_pop_candidates,
)
from vantage.control.policy.common.scoring import (
    CandidateScorer,
    SatelliteCostScorer,
    select_best,
)
from vantage.control.policy.common.utils import find_ingress_satellite
from vantage.domain import (
    Endpoint,
    FlowKey,
    NetworkSnapshot,
    PathAllocation,
    RoutingIntent,
    TrafficDemand,
)


class CandidateBasedController:
    """Base controller: enumerate candidates → score → select best.

    Subclasses provide the scorer and optionally override candidate
    filtering (e.g., restrict to a single PoP).
    """

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        scorer: CandidateScorer | None = None,
    ) -> None:
        self._endpoints = endpoints or {}
        self._scorer = scorer or SatelliteCostScorer()

    def optimize(
        self, snapshot: NetworkSnapshot, demand: TrafficDemand
    ) -> RoutingIntent:
        allocations: dict[FlowKey, PathAllocation] = {}

        for flow_key in demand.flows:
            src = self._endpoints.get(flow_key.src)
            if src is None:
                continue
            uplink = find_ingress_satellite(src, snapshot.satellite.positions)
            if uplink is None:
                continue

            alloc = self._select_alloc(
                flow_key, src, uplink.sat_id, snapshot
            )
            if alloc is not None:
                allocations[flow_key] = alloc

        return RoutingIntent(
            epoch=demand.epoch, allocations=MappingProxyType(allocations)
        )

    def _select_alloc(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        user_sat: int,
        snapshot: NetworkSnapshot,
    ) -> PathAllocation | None:
        """Select the best allocation for a flow. Override for custom logic."""
        candidates = enumerate_all_candidates(user_sat, src_ep, snapshot)
        best = select_best(candidates, self._scorer)
        return best.to_allocation() if best is not None else None
