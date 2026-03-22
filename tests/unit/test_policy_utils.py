"""Tests for shared policy utilities, candidates, and scoring."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from vantage.control.policy.common.candidate import (
    PathCandidate,
    enumerate_all_candidates,
    enumerate_pop_candidates,
)
from vantage.control.policy.common.scoring import (
    E2EScorer,
    SatelliteCostScorer,
    satellite_cost_scorer,
    select_best,
)
from vantage.control.policy.common.utils import find_ingress_satellite, find_nearest_pop
from vantage.domain import (
    AccessLink,
    Endpoint,
    GSPoPEdge,
    GatewayAttachments,
    GroundStation,
    ISLEdge,
    ISLGraph,
    InfrastructureView,
    NetworkSnapshot,
    PoP,
    SatelliteState,
)
from vantage.world.ground import GroundKnowledge


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_pop_snapshot() -> NetworkSnapshot:
    """2-sat, 2-PoP network for candidate enumeration tests."""
    graph = ISLGraph(
        shell_id=1, timeslot=0, num_sats=2,
        edges=(ISLEdge(0, 1, 1.0, 300.0, "intra_orbit"),),
    )
    positions = np.array([[0.0, 0.0, 550.0], [1.0, 1.0, 550.0]])
    delay_matrix = np.array([[0.0, 5.0], [5.0, 0.0]])
    pred_matrix = np.array([[0, 0], [1, 1]], dtype=np.int32)

    gw = GatewayAttachments(attachments=MappingProxyType({
        "gs1": (AccessLink(sat_id=1, elevation_deg=80.0, slant_range_km=560.0, delay=1.87),),
        "gs2": (AccessLink(sat_id=0, elevation_deg=70.0, slant_range_km=580.0, delay=1.93),),
    }))
    sat = SatelliteState(
        positions=positions, graph=graph,
        delay_matrix=delay_matrix, predecessor_matrix=pred_matrix,
        gateway_attachments=gw,
    )
    gs1 = GroundStation("gs1", 0.5, 0.5, "XX", "T1", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False)
    gs2 = GroundStation("gs2", -0.5, -0.5, "XX", "T2", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False)
    pop_a = PoP("pop_a", "a", "PopA", 0.5, 0.5)
    pop_b = PoP("pop_b", "b", "PopB", -0.5, -0.5)
    edges = (
        GSPoPEdge("gs1", "a", 10.0, 0.05, 100.0),
        GSPoPEdge("gs2", "b", 15.0, 0.07, 100.0),
    )
    infra = InfrastructureView(
        pops=(pop_a, pop_b), ground_stations=(gs1, gs2), gs_pop_edges=edges,
    )

    return NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)


@pytest.fixture
def src_ep() -> Endpoint:
    return Endpoint("user", 0.0, 0.0)


# ---------------------------------------------------------------------------
# find_nearest_pop
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindNearestPop:

    def test_returns_nearest(self) -> None:
        pops = (
            PoP("far", "far", "Far", 50.0, 0.0),
            PoP("near", "near", "Near", 0.1, 0.1),
        )
        result = find_nearest_pop(0.0, 0.0, pops)
        assert result is not None
        assert result.code == "near"

    def test_single_pop(self) -> None:
        pops = (PoP("only", "only", "Only", 10.0, 20.0),)
        result = find_nearest_pop(0.0, 0.0, pops)
        assert result is not None
        assert result.code == "only"

    def test_empty_pops_returns_none(self) -> None:
        result = find_nearest_pop(0.0, 0.0, ())
        assert result is None


# ---------------------------------------------------------------------------
# find_ingress_satellite
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindIngressSatellite:

    def test_returns_visible_satellite(self) -> None:
        positions = np.array([[0.0, 0.0, 550.0]])
        src = Endpoint("user", 0.0, 0.0)
        result = find_ingress_satellite(src, positions)
        assert result is not None
        assert result.sat_id == 0

    def test_no_visible_returns_none(self) -> None:
        positions = np.array([[0.0, 180.0, 550.0]])
        src = Endpoint("user", 0.0, 0.0)
        result = find_ingress_satellite(src, positions)
        assert result is None


# ---------------------------------------------------------------------------
# PathCandidate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPathCandidate:

    def test_satellite_rtt(self) -> None:
        c = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0)
        assert c.satellite_rtt == 15.0

    def test_to_allocation(self) -> None:
        c = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0)
        alloc = c.to_allocation()
        assert alloc.pop_code == "p"
        assert alloc.gs_id == "gs"
        assert alloc.user_sat == 0
        assert alloc.egress_sat == 1

    def test_frozen(self) -> None:
        c = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0)
        with pytest.raises(AttributeError):
            c.pop_code = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# enumerate_pop_candidates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnumeratePopCandidates:

    def test_enumerates_valid_candidates(
        self, two_pop_snapshot: NetworkSnapshot, src_ep: Endpoint
    ) -> None:
        candidates = enumerate_pop_candidates("a", 0, src_ep, two_pop_snapshot)
        assert len(candidates) >= 1
        for c in candidates:
            assert c.pop_code == "a"
            assert c.propagation_rtt > 0
            assert c.backhaul_rtt > 0

    def test_unknown_pop_returns_empty(
        self, two_pop_snapshot: NetworkSnapshot, src_ep: Endpoint
    ) -> None:
        candidates = enumerate_pop_candidates("nonexistent", 0, src_ep, two_pop_snapshot)
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# enumerate_all_candidates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnumerateAllCandidates:

    def test_covers_both_pops(
        self, two_pop_snapshot: NetworkSnapshot, src_ep: Endpoint
    ) -> None:
        candidates = enumerate_all_candidates(0, src_ep, two_pop_snapshot)
        pop_codes = {c.pop_code for c in candidates}
        assert "a" in pop_codes
        assert "b" in pop_codes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSatelliteCostScorer:

    def test_scores_by_satellite_rtt(self) -> None:
        c = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0)
        assert satellite_cost_scorer.score(c) == 15.0


@pytest.mark.unit
class TestE2EScorer:

    def test_scores_with_ground_rtt(self) -> None:
        scorer = E2EScorer()
        c = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0, ground_rtt=8.0)
        assert scorer.score(c) == 23.0  # 10 + 5 + 8

    def test_returns_none_without_ground_rtt(self) -> None:
        scorer = E2EScorer()
        c = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0)
        assert scorer.score(c) is None


@pytest.mark.unit
class TestSelectBest:

    def test_picks_lowest_score(self) -> None:
        candidates = [
            PathCandidate("a", "gs1", 0, 1, propagation_rtt=20.0, backhaul_rtt=5.0),
            PathCandidate("b", "gs2", 0, 1, propagation_rtt=10.0, backhaul_rtt=3.0),
        ]
        best = select_best(candidates, satellite_cost_scorer)
        assert best is not None
        assert best.pop_code == "b"

    def test_skips_none_scores(self) -> None:
        scorer = E2EScorer()
        candidates = [
            PathCandidate("a", "gs1", 0, 1, propagation_rtt=10.0, backhaul_rtt=3.0),  # no ground
            PathCandidate("b", "gs2", 0, 1, propagation_rtt=20.0, backhaul_rtt=5.0, ground_rtt=5.0),
        ]
        best = select_best(candidates, scorer)
        assert best is not None
        assert best.pop_code == "b"

    def test_empty_returns_none(self) -> None:
        best = select_best([], satellite_cost_scorer)
        assert best is None

    def test_total_known_rtt(self) -> None:
        c_with = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0, ground_rtt=8.0)
        assert c_with.total_known_rtt == 23.0
        c_without = PathCandidate("p", "gs", 0, 1, propagation_rtt=10.0, backhaul_rtt=5.0)
        assert c_without.total_known_rtt is None
