"""GroundOnly controller: ground-delay oracle baseline.

Picks the PoP with lowest ground delay to the destination,
ignoring satellite segment cost.
"""

from __future__ import annotations

from vantage.control.policy.common.base import CandidateBasedController
from vantage.control.policy.common.candidate import enumerate_pop_candidates
from vantage.control.policy.common.scoring import SatelliteCostScorer, select_best
from vantage.domain import Endpoint, FlowKey, NetworkSnapshot, PathAllocation
from vantage.world.ground import GroundDelay, HaversineDelay


class GroundOnlyController(CandidateBasedController):
    """Baseline: minimize ground delay, ignore satellite segment."""

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        ground_delay: GroundDelay | None = None,
    ) -> None:
        super().__init__(endpoints=endpoints, scorer=SatelliteCostScorer())
        self._ground_delay: GroundDelay = ground_delay or HaversineDelay()

    def _select_alloc(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        user_sat: int,
        snapshot: NetworkSnapshot,
    ) -> PathAllocation | None:
        dst = self._endpoints.get(flow_key.dst)
        if dst is None:
            return None

        # Stage 1: select PoP with lowest ground delay
        best_pop: str | None = None
        best_delay = float("inf")
        for pop in snapshot.infra.pops:
            if not snapshot.infra.pop_gs_edges(pop.code):
                continue
            delay = self._ground_delay.estimate(
                pop.lat_deg, pop.lon_deg, dst.lat_deg, dst.lon_deg
            )
            if delay < best_delay:
                best_delay = delay
                best_pop = pop.code

        if best_pop is None:
            return None

        # Stage 2: select best satellite path within that PoP
        candidates = enumerate_pop_candidates(
            best_pop, user_sat, src_ep, snapshot
        )
        best = select_best(candidates, self._scorer)
        return best.to_allocation() if best is not None else None
