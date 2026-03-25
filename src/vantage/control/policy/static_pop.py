"""StaticPoP controller: best static PoP per destination from cache.

Fills sat_cost with zeros, ground_cost from GroundKnowledge cache.
Terminal picks PoP with lowest cached ground delay.
"""

from __future__ import annotations

from types import MappingProxyType

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
        sat_cost: dict[tuple[int, str], float] = {}

        # Ground cost from cache only (no estimation)
        ground_cost: dict[tuple[str, str], float] = {}
        for pop in snapshot.infra.pops:
            for (pop_code, dest), delay in self._gk._cache.items():
                if pop_code == pop.code:
                    ground_cost[(pop_code, dest)] = delay

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
