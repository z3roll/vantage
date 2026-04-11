"""End-to-end integration test: full epoch loop with cost-table controllers."""

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
from vantage.world.ground import (
    GroundInfrastructure,
    GroundKnowledge,
    MeasuredGroundDelay,
)
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "vantage" / "config"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")
TRACEROUTE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "probe_trace" / "traceroute"
)


@pytest.fixture(scope="module")
def world() -> WorldModel:
    constellation = XMLConstellationModel(str(STARPERF_XML), dt_s=15.0)
    ground = GroundInfrastructure(DATA_DIR)
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
    population = EndpointPopulation.from_terminal_registry(DATA_DIR / "terminals.json")
    ep = {}
    for s in population.sources:
        ep[s.name] = s
    for d in population.destinations:
        ep[d.name] = d
    return ep


def _make_context(world: WorldModel, endpoints: dict[str, Endpoint]) -> RunContext:
    ground_truth = MeasuredGroundDelay.from_traceroute_dir(TRACEROUTE_DIR)
    return RunContext(
        world=world,
        endpoints=endpoints,
        ground_knowledge=GroundKnowledge(estimator=ground_truth),
    )


@pytest.mark.integration
class TestE2EPipeline:

    def test_nearest_pop_runs(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = create_controller("nearest_pop")
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="nearest_pop")

        assert result.num_epochs == 1
        assert len(result.epochs[0].flow_outcomes) > 0
        assert result.epochs[0].routed_demand_gbps > 0

    def test_greedy_runs(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "terminals.json"),
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

    def test_feedback_populates_knowledge(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = VantageGreedyController(
            endpoints=ctx.endpoints,
            ground_knowledge=ctx.ground_knowledge,
        )
        config = RunConfig(num_epochs=2, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="greedy")

        all_ground = [f.ground_rtt for e in result.epochs for f in e.flow_outcomes]
        assert any(g > 0 for g in all_ground), "Feedback loop failed"

    def test_flow_delay_decomposition(self, world: WorldModel, endpoints: dict[str, Endpoint]) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(
            EndpointPopulation.from_terminal_registry(DATA_DIR / "terminals.json"),
            demand_per_flow_gbps=0.01,
        )
        controller = create_controller("nearest_pop")
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run(ctx, traffic, controller, config=config, controller_name="nearest_pop")

        for flow in result.epochs[0].flow_outcomes:
            expected = flow.satellite_rtt + flow.ground_rtt
            assert abs(flow.total_rtt - expected) < 1e-9
