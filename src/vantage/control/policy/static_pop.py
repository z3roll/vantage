"""StaticPoP controller: best static PoP per destination.

Same PoP for all users to the same destination, based on ground
knowledge lookup.
"""

from __future__ import annotations

from vantage.control.policy.common.base import CandidateBasedController
from vantage.control.policy.common.candidate import enumerate_pop_candidates
from vantage.control.policy.common.scoring import SatelliteCostScorer, select_best
from vantage.domain import Endpoint, FlowKey, NetworkSnapshot, PathAllocation
from vantage.world.ground import GroundKnowledge


class StaticPoPController(CandidateBasedController):
    """Baseline: static per-destination PoP from ground knowledge."""

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        ground_knowledge: GroundKnowledge | None = None,
    ) -> None:
        super().__init__(endpoints=endpoints, scorer=SatelliteCostScorer())
        self._gk = ground_knowledge or GroundKnowledge()

    def _select_alloc(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        user_sat: int,
        snapshot: NetworkSnapshot,
    ) -> PathAllocation | None:
        best = self._gk.best_pop_for(flow_key.dst)
        if best is None:
            return None
        candidates = enumerate_pop_candidates(
            best[0], user_sat, src_ep, snapshot
        )
        selected = select_best(candidates, self._scorer)
        return selected.to_allocation() if selected is not None else None
