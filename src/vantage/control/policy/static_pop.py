"""StaticPoP controller: best static PoP per destination from cache.

Uses real sat_cost + ground_cost from cache only (no estimation).
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, NetworkSnapshot
from vantage.world.ground import GroundKnowledge


class StaticPoPController:
    """Baseline: static per-destination PoP from ground knowledge cache."""

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        sat_cost = precompute_sat_cost(snapshot)

        # Ground cost from cache only (no estimation fallback)
        ground_cost: dict[tuple[str, str], float] = {}
        for (pop_code, dest), delay in self._gk._cache.items():
            ground_cost[(pop_code, dest)] = delay

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
