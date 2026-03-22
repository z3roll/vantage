"""Tests for TEController Protocol and baseline controllers."""

from __future__ import annotations

import pytest

from vantage.domain import (
    Endpoint,
    FlowKey,
    ISLEdge,
    TrafficDemand,
)
from vantage.control.controller import (
    SupportsGroundFeedback,
    create_controller,
)
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.control.policy.ground_only import GroundOnlyController
from vantage.control.policy.greedy import VantageGreedyController
from vantage.world.ground import GroundKnowledge


@pytest.fixture
def endpoints() -> dict[str, Endpoint]:
    return {
        "user_ny": Endpoint("user_ny", 40.7, -74.0),
        "user_london": Endpoint("user_london", 51.5, -0.1),
        "google": Endpoint("google", 37.4, -122.1),
        "facebook": Endpoint("facebook", 37.5, -122.1),
    }


@pytest.fixture
def demand() -> TrafficDemand:
    from types import MappingProxyType
    flows = {
        FlowKey("user_ny", "google"): 1.0,
        FlowKey("user_london", "google"): 1.0,
        FlowKey("user_ny", "facebook"): 0.5,
    }
    return TrafficDemand(epoch=0, flows=MappingProxyType(flows))


@pytest.mark.unit
class TestNearestPoPController:

    def test_selects_nearest_pop(
        self, endpoints: dict[str, Endpoint], demand: TrafficDemand
    ) -> None:
        from vantage.domain import (
            PoP, GroundStation, GSPoPEdge, AccessLink,
            InfrastructureView, GatewayAttachments,
            NetworkSnapshot, SatelliteState, ISLGraph,
        )
        import numpy as np
        from types import MappingProxyType

        pops = (
            PoP("sttlwax1", "sea", "Seattle", 47.6, -122.3),
            PoP("lonlon1", "lon", "London", 51.5, -0.1),
        )
        gs_list = (
            GroundStation("gs_sea", 47.2, -122.0, "US", "WA", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False),
            GroundStation("gs_lon", 51.4, -0.2, "GB", "London", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False),
        )
        edges = (
            GSPoPEdge("gs_sea", "sea", 50.0, 0.25, 100.0),
            GSPoPEdge("gs_lon", "lon", 30.0, 0.15, 100.0),
        )
        infra = InfrastructureView(pops=pops, ground_stations=gs_list, gs_pop_edges=edges)
        positions = np.array([[40.7, -74.0, 550.0], [51.5, -0.1, 550.0]])
        graph = ISLGraph(shell_id=1, timeslot=0, num_sats=2,
                         edges=(ISLEdge(0, 1, 20.0, 6000.0, "inter_orbit"),))
        gw = GatewayAttachments(attachments=MappingProxyType({
            "gs_sea": (AccessLink(0, 85.0, 555.0, 1.85),),
            "gs_lon": (AccessLink(1, 85.0, 555.0, 1.85),),
        }))
        sat = SatelliteState(
            positions=positions, graph=graph,
            delay_matrix=np.array([[0.0, 20.0], [20.0, 0.0]]),
            predecessor_matrix=np.array([[0, 0], [1, 1]], dtype=np.int32),
            gateway_attachments=gw,
        )
        snapshot = NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)

        ctrl = NearestPoPController(endpoints=endpoints)
        intent = ctrl.optimize(snapshot, demand)

        assert intent.allocations[FlowKey("user_ny", "google")].pop_code == "sea"
        assert intent.allocations[FlowKey("user_london", "google")].pop_code == "lon"

    def test_all_flows_allocated(
        self, endpoints: dict[str, Endpoint], demand: TrafficDemand
    ) -> None:
        from vantage.domain import (
            PoP, GroundStation, GSPoPEdge, AccessLink,
            InfrastructureView, GatewayAttachments,
            NetworkSnapshot, SatelliteState, ISLGraph,
        )
        import numpy as np
        from types import MappingProxyType

        pops = (PoP("sttlwax1", "sea", "Seattle", 47.6, -122.3),)
        gs_list = (GroundStation("gs1", 47.2, -122.0, "US", "WA", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False),)
        edges = (GSPoPEdge("gs1", "sea", 50.0, 0.25, 100.0),)
        infra = InfrastructureView(pops=pops, ground_stations=gs_list, gs_pop_edges=edges)
        positions = np.array([[40.7, -74.0, 550.0], [51.5, -0.1, 550.0]])
        graph = ISLGraph(shell_id=1, timeslot=0, num_sats=2,
                         edges=(ISLEdge(0, 1, 20.0, 6000.0, "inter_orbit"),))
        gw = GatewayAttachments(attachments=MappingProxyType({
            "gs1": (AccessLink(0, 85.0, 555.0, 1.85),
                    AccessLink(1, 30.0, 1000.0, 3.3)),
        }))
        sat = SatelliteState(
            positions=positions, graph=graph,
            delay_matrix=np.array([[0.0, 20.0], [20.0, 0.0]]),
            predecessor_matrix=np.array([[0, 0], [1, 1]], dtype=np.int32),
            gateway_attachments=gw,
        )
        snapshot = NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)

        ctrl = NearestPoPController(endpoints=endpoints)
        intent = ctrl.optimize(snapshot, demand)
        assert len(intent.allocations) == 3


@pytest.mark.unit
class TestGroundOnlyController:

    def test_selects_pop_nearest_to_destination(
        self, endpoints: dict[str, Endpoint], demand: TrafficDemand
    ) -> None:
        from vantage.domain import (
            PoP, GroundStation, GSPoPEdge, AccessLink,
            InfrastructureView, GatewayAttachments,
            NetworkSnapshot, SatelliteState, ISLGraph,
        )
        import numpy as np
        from types import MappingProxyType

        pops = (
            PoP("sttlwax1", "sea", "Seattle", 47.6, -122.3),
            PoP("sjcsjc1", "sjc", "San Jose", 37.3, -121.9),
        )
        gs_list = (
            GroundStation("gs1", 47.2, -122.0, "US", "WA", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False),
            GroundStation("gs2", 37.2, -121.8, "US", "CA", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False),
        )
        edges = (
            GSPoPEdge("gs1", "sea", 100.0, 0.5, 100.0),
            GSPoPEdge("gs2", "sjc", 50.0, 0.25, 100.0),
        )
        infra = InfrastructureView(pops=pops, ground_stations=gs_list, gs_pop_edges=edges)
        positions = np.array([[40.7, -74.0, 550.0], [51.5, -0.1, 550.0]])
        graph = ISLGraph(shell_id=1, timeslot=0, num_sats=2,
                         edges=(ISLEdge(0, 1, 20.0, 6000.0, "inter_orbit"),))
        gw = GatewayAttachments(attachments=MappingProxyType({
            "gs1": (AccessLink(0, 30.0, 1000.0, 3.3),),
            "gs2": (AccessLink(1, 35.0, 900.0, 3.0),),
        }))
        sat = SatelliteState(
            positions=positions, graph=graph,
            delay_matrix=np.array([[0.0, 20.0], [20.0, 0.0]]),
            predecessor_matrix=np.array([[0, 0], [1, 1]], dtype=np.int32),
            gateway_attachments=gw,
        )
        snapshot = NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)

        ctrl = GroundOnlyController(endpoints=endpoints)
        intent = ctrl.optimize(snapshot, demand)

        assert intent.allocations[FlowKey("user_ny", "google")].pop_code == "sjc"
        assert intent.allocations[FlowKey("user_london", "google")].pop_code == "sjc"


@pytest.mark.unit
class TestControllerFactory:

    def test_create_nearest_pop(self) -> None:
        ctrl = create_controller("nearest_pop")
        assert isinstance(ctrl, NearestPoPController)

    def test_create_ground_only(self) -> None:
        ctrl = create_controller("ground_only")
        assert isinstance(ctrl, GroundOnlyController)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown controller"):
            create_controller("nonexistent")

    def test_greedy_supports_feedback_protocol(self) -> None:
        ctrl = VantageGreedyController()
        assert isinstance(ctrl, SupportsGroundFeedback)

    def test_nearest_pop_not_feedback(self) -> None:
        ctrl = NearestPoPController()
        assert not isinstance(ctrl, SupportsGroundFeedback)

    def test_ground_only_not_feedback(self) -> None:
        ctrl = GroundOnlyController()
        assert not isinstance(ctrl, SupportsGroundFeedback)

    def test_greedy_shares_knowledge(self) -> None:
        gk = GroundKnowledge()
        ctrl = VantageGreedyController(ground_knowledge=gk)
        assert ctrl.ground_knowledge is gk

    def test_intent_is_frozen(
        self, endpoints: dict[str, Endpoint], demand: TrafficDemand
    ) -> None:
        from vantage.domain import (
            PoP, GroundStation, GSPoPEdge, AccessLink,
            InfrastructureView, GatewayAttachments,
            NetworkSnapshot, SatelliteState, ISLGraph,
        )
        import numpy as np
        from types import MappingProxyType

        pops = (PoP("x", "x", "X", 0.0, 0.0),)
        gs_list = (GroundStation("gs1", 0.0, 0.0, "XX", "T", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False),)
        edges = (GSPoPEdge("gs1", "x", 10.0, 0.05, 100.0),)
        infra = InfrastructureView(pops=pops, ground_stations=gs_list, gs_pop_edges=edges)
        positions = np.array([[0.0, 0.0, 550.0]])
        graph = ISLGraph(shell_id=1, timeslot=0, num_sats=1, edges=())
        gw = GatewayAttachments(attachments=MappingProxyType({
            "gs1": (AccessLink(0, 85.0, 555.0, 1.85),),
        }))
        sat = SatelliteState(
            positions=positions, graph=graph,
            delay_matrix=np.zeros((1, 1)),
            predecessor_matrix=np.array([[0]], dtype=np.int32),
            gateway_attachments=gw,
        )
        snapshot = NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)

        ctrl = NearestPoPController(endpoints=endpoints)
        intent = ctrl.optimize(snapshot, demand)
        with pytest.raises(AttributeError):
            intent.epoch = 99  # type: ignore[misc]
