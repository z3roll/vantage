"""VantageGreedy controller: ground-aware joint E2E optimization.

Fills sat_cost with real values, ground_cost from GroundKnowledge cache.
Only uses data that has been measured (no oracle estimation).
Terminals pick PoP with lowest total (sat + ground) E2E delay.
Missing ground_cost entries default to 0 in forward → those PoPs
compete on satellite cost alone (optimistic for unmeasured paths).
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, Endpoint, NetworkSnapshot
from vantage.world.ground import GroundKnowledge


class VantageGreedyController:
    """Joint E2E optimization using only measured ground delay data."""

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

        # Ground cost: only from GroundKnowledge cache (measured data).
        # No oracle estimation — unmeasured (pop, dest) pairs are simply
        # absent, and forward.py defaults them to 0.
        ground_cost = self._gk.all_entries()

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
