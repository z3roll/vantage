"""VantageGreedy controller: ground-aware joint E2E optimization.

The primary algorithm. Enumerates all candidate paths across all PoPs,
scores each by total E2E delay (satellite + ground), picks the best.
Falls back to NearestPoP for unknown destinations.
"""

from __future__ import annotations

from vantage.control.policy.common.base import CandidateBasedController
from vantage.control.policy.common.candidate import (
    enumerate_all_candidates,
    enumerate_pop_candidates,
)
from vantage.control.policy.common.scoring import (
    E2EScorer,
    SatelliteCostScorer,
    select_best,
)
from vantage.control.policy.common.utils import find_nearest_pop
from vantage.domain import Endpoint, FlowKey, NetworkSnapshot, PathAllocation
from vantage.world.ground import GroundKnowledge


class VantageGreedyController(CandidateBasedController):
    """Joint E2E delay optimization via greedy PoP selection.

    Fallback behavior:
    - Unknown destination (no knowledge entry): fall back to nearest PoP.
    - Nearest PoP has no ground delay data: fall back to nearest PoP.
    - Greedy search finds a better PoP: use it.
    """

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        ground_knowledge: GroundKnowledge | None = None,
    ) -> None:
        super().__init__(endpoints=endpoints, scorer=E2EScorer())
        self._gk = ground_knowledge or GroundKnowledge()

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    def _select_alloc(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        user_sat: int,
        snapshot: NetworkSnapshot,
    ) -> PathAllocation | None:
        nearest = find_nearest_pop(
            src_ep.lat_deg, src_ep.lon_deg, snapshot.infra.pops
        )
        if nearest is None:
            return None

        # Fallback 1: destination unknown
        if not self._gk.has(flow_key.dst):
            return self._fallback_nearest(nearest.code, user_sat, src_ep, snapshot)

        # Fallback 2: nearest PoP has no ground delay for this destination
        if self._gk.get(nearest.code, flow_key.dst) is None:
            return self._fallback_nearest(nearest.code, user_sat, src_ep, snapshot)

        # Greedy: enumerate all candidates with ground_rtt, score by E2E
        candidates = enumerate_all_candidates(
            user_sat, src_ep, snapshot,
            ground_knowledge=self._gk, dest=flow_key.dst,
        )
        best = select_best(candidates, self._scorer)
        return best.to_allocation() if best is not None else None

    @staticmethod
    def _fallback_nearest(
        pop_code: str,
        user_sat: int,
        src_ep: Endpoint,
        snapshot: NetworkSnapshot,
    ) -> PathAllocation | None:
        candidates = enumerate_pop_candidates(
            pop_code, user_sat, src_ep, snapshot
        )
        best = select_best(candidates, SatelliteCostScorer())
        return best.to_allocation() if best is not None else None
