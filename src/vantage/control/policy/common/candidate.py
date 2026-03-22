"""Path candidate enumeration for controller policies.

Enumerates all valid (PoP, GS, egress_sat) combinations with pre-computed
delay components. Candidates are strategy-neutral — scorers decide ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vantage.domain import Endpoint, NetworkSnapshot, PathAllocation

if TYPE_CHECKING:
    from vantage.world.ground import GroundKnowledge


@dataclass(frozen=True, slots=True)
class PathCandidate:
    """A fully resolved candidate path with pre-computed delay components.

    All delay values are RTT in ms. ground_rtt is None when no ground
    delay information is available for this (pop, dest) pair.
    """

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int
    propagation_rtt: float  # uplink + ISL + downlink RTT
    backhaul_rtt: float  # GS ↔ PoP RTT
    ground_rtt: float | None = None  # PoP → dest RTT (None if unknown)

    @property
    def satellite_rtt(self) -> float:
        """Total satellite segment RTT (user → PoP)."""
        return self.propagation_rtt + self.backhaul_rtt

    @property
    def total_known_rtt(self) -> float | None:
        """Total E2E RTT if ground delay is known, else None."""
        if self.ground_rtt is None:
            return None
        return self.satellite_rtt + self.ground_rtt

    def to_allocation(self) -> PathAllocation:
        """Convert to PathAllocation for RoutingIntent."""
        return PathAllocation(
            pop_code=self.pop_code,
            gs_id=self.gs_id,
            user_sat=self.user_sat,
            egress_sat=self.egress_sat,
        )


def enumerate_pop_candidates(
    pop_code: str,
    user_sat: int,
    src_ep: Endpoint,
    snapshot: NetworkSnapshot,
    ground_knowledge: GroundKnowledge | None = None,
    dest: str | None = None,
) -> tuple[PathCandidate, ...]:
    """Enumerate all (GS, egress_sat) candidates for a given PoP.

    If ground_knowledge and dest are provided, each candidate's
    ground_rtt is filled from the knowledge service.
    """
    sat = snapshot.satellite
    candidates: list[PathCandidate] = []

    # Pre-fetch ground_rtt for this PoP if available
    grtt: float | None = None
    if ground_knowledge is not None and dest is not None:
        grtt = ground_knowledge.get(pop_code, dest)

    for gs_id, backhaul in snapshot.infra.pop_gs_edges(pop_code):
        gs = snapshot.infra.gs_by_id(gs_id)
        if gs is None:
            continue
        gs_links = sat.gateway_attachments.attachments.get(gs_id)
        if not gs_links:
            continue
        for downlink in gs_links:
            propagation_rtt = sat.compute_satellite_rtt(
                user_sat, downlink.sat_id,
                src_ep.lat_deg, src_ep.lon_deg,
                gs.lat_deg, gs.lon_deg,
            )
            candidates.append(PathCandidate(
                pop_code=pop_code,
                gs_id=gs_id,
                user_sat=user_sat,
                egress_sat=downlink.sat_id,
                propagation_rtt=propagation_rtt,
                backhaul_rtt=backhaul * 2,
                ground_rtt=grtt,
            ))

    return tuple(candidates)


def enumerate_all_candidates(
    user_sat: int,
    src_ep: Endpoint,
    snapshot: NetworkSnapshot,
    ground_knowledge: GroundKnowledge | None = None,
    dest: str | None = None,
) -> tuple[PathCandidate, ...]:
    """Enumerate candidates across all PoPs.

    Strategy-neutral: produces the full candidate set. Scorers decide
    which candidates are viable and how to rank them.
    """
    all_candidates: list[PathCandidate] = []
    for pop in snapshot.infra.pops:
        all_candidates.extend(
            enumerate_pop_candidates(
                pop.code, user_sat, src_ep, snapshot,
                ground_knowledge, dest,
            )
        )
    return tuple(all_candidates)
