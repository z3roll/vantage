"""NearestPoP controller: hot-potato baseline.

Routes each flow to the PoP geographically closest to the source.
"""

from __future__ import annotations

from vantage.control.policy.common.base import CandidateBasedController
from vantage.control.policy.common.candidate import enumerate_pop_candidates
from vantage.control.policy.common.scoring import SatelliteCostScorer, select_best
from vantage.control.policy.common.utils import find_nearest_pop
from vantage.domain import Endpoint, FlowKey, NetworkSnapshot, PathAllocation


class NearestPoPController(CandidateBasedController):
    """Baseline: route each flow to the PoP nearest to its source."""

    def __init__(self, endpoints: dict[str, Endpoint] | None = None) -> None:
        super().__init__(endpoints=endpoints, scorer=SatelliteCostScorer())

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
        candidates = enumerate_pop_candidates(
            nearest.code, user_sat, src_ep, snapshot
        )
        best = select_best(candidates, self._scorer)
        return best.to_allocation() if best is not None else None
