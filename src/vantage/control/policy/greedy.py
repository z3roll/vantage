"""VantageGreedy controller: ground-aware joint E2E optimization.

Fills sat_cost with real values, ground_cost with L1 cache + L2 fallback.
Terminal picks PoP with lowest total (sat + ground) E2E delay.
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, Endpoint, NetworkSnapshot
from vantage.world.ground import GroundKnowledge


class VantageGreedyController:
    """Joint E2E optimization: both sat_cost and ground_cost are populated."""

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        ground_knowledge: GroundKnowledge | None = None,
    ) -> None:
        self._endpoints = endpoints or {}
        self._gk = ground_knowledge or GroundKnowledge()

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        sat_cost = precompute_sat_cost(snapshot)

        # Ground cost: L1 cache hit or L2 estimator fallback
        # All (PoP, dest) pairs populated → no exploration gap
        ground_cost: dict[tuple[str, str], float] = {}
        destinations = [
            ep for ep in self._endpoints.values()
            if not ep.name.startswith("terminal_")
        ]
        for pop in snapshot.infra.pops:
            for dst in destinations:
                ground_cost[(pop.code, dst.name)] = self._gk.get_or_estimate(
                    pop.code, dst.name,
                    pop.lat_deg, pop.lon_deg,
                    dst.lat_deg, dst.lon_deg,
                )

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
