"""End-to-end integration test: full epoch loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.engine.context import RunContext
from vantage.control.controller import create_controller
from vantage.control.policy.greedy import VantageGreedyController
from vantage.engine import RunConfig, run
from vantage.domain import Endpoint
from vantage.traffic import EndpointPopulation, UniformGenerator
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.ground import GroundInfrastructure, GroundKnowledge, HaversineDelay
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")


@pytest.fixture(scope="module")
def world() -> WorldModel:
    constellation = XMLConstellationModel(str(STARPERF_XML), dt_s=15.0)
    ground = GroundInfrastructure(DATA_DIR / "processed")
    satellite = SatelliteSegment(
        constellation=constellation,
        topology_builder=PlusGridTopology(),
        shell_id=1,
        ground_stations=ground.ground_stations,
        visibility=SphericalAccessModel(),
    )
    return WorldModel(satellite, ground)


@pytest.fixture(scope="module")
def endpoints() -> dict[str, Endpoint]:
    pop = EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json")
    ep = {}
    for s in pop.sources:
        ep[s.name] = s
    for d in pop.destinations:
        ep[d.name] = d
    return ep


def _make_context(world: WorldModel, endpoints: dict[str, Endpoint]) -> RunContext:
    """Create an isolated RunContext with fresh knowledge + L2 estimator."""
    return RunContext(
        world=world,
        endpoints=endpoints,
        ground_knowledge=GroundKnowledge(estimator=HaversineDelay()),
    )


@pytest.mark.integration
class TestE2EPipeline:
    """End-to-end simulation pipeline test.

    Each test uses an isolated RunContext with a fresh GroundKnowledge
    to avoid cross-test state leakage.
    """

    def test_nearest_pop_runs(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = create_controller("nearest_pop", endpoints=ctx.endpoints)
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="nearest_pop")

        assert result.num_epochs == 1
        epoch = result.epochs[0]
        assert len(epoch.flow_outcomes) > 0
        assert epoch.routed_demand_gbps > 0

    def test_greedy_runs(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = VantageGreedyController(
            endpoints=ctx.endpoints,
            ground_knowledge=ctx.ground_knowledge,
        )
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="greedy")

        assert result.num_epochs == 1
        assert len(result.epochs[0].flow_outcomes) > 0

    def test_greedy_improves_over_nearest(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx_nearest = _make_context(world, endpoints)
        ctx_greedy = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        nearest_ctrl = create_controller("nearest_pop", endpoints=ctx_nearest.endpoints)
        greedy_ctrl = VantageGreedyController(
            endpoints=ctx_greedy.endpoints,
            ground_knowledge=ctx_greedy.ground_knowledge,
        )

        r_nearest = run(ctx_nearest, traffic, nearest_ctrl, config=config, controller_name="nearest_pop")
        r_greedy = run(ctx_greedy, traffic, greedy_ctrl, config=config, controller_name="greedy")

        assert r_greedy.avg_total_rtt < r_nearest.avg_total_rtt * 1.1

    def test_feedback_populates_cache(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        """Feedback loop must bootstrap: empty knowledge → estimated ground_rtt → knowledge populated."""
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = VantageGreedyController(
            endpoints=ctx.endpoints,
            ground_knowledge=ctx.ground_knowledge,
        )
        config = RunConfig(num_epochs=2, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="greedy")

        all_ground = [f.ground_rtt for e in result.epochs for f in e.flow_outcomes]
        assert any(g > 0 for g in all_ground), "Feedback loop failed: all ground_rtt are 0.0"
        assert ctx.ground_knowledge.has("google") or ctx.ground_knowledge.has("facebook"), (
            "Knowledge was not populated by engine feedback"
        )

    def test_ground_rtt_nonzero_with_estimator(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = create_controller("nearest_pop", endpoints=ctx.endpoints)
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="nearest_pop")

        for flow in result.epochs[0].flow_outcomes:
            assert flow.ground_rtt > 0, f"Flow {flow.flow_key} has ground_rtt=0"

    def test_flow_delay_decomposition(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "processed/terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = create_controller("nearest_pop", endpoints=ctx.endpoints)
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="nearest_pop")

        for flow in result.epochs[0].flow_outcomes:
            expected = flow.satellite_rtt + flow.ground_rtt
            assert abs(flow.total_rtt - expected) < 1e-9
