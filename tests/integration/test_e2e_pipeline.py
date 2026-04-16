"""End-to-end integration test: full epoch loop with RoutingPlane controllers."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.engine.context import RunContext
from vantage.control.controller import create_controller
from vantage.control.policy.greedy import ProgressiveController
from vantage.domain import CellGrid, Endpoint
from vantage.engine import RunConfig, run_routing
from vantage.traffic import EndpointPopulation, UniformGenerator
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.ground import (
    GeographicGroundDelay,
    GroundInfrastructure,
    GroundKnowledge,
)
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "vantage" / "config"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")
LAND_GEOJSON = Path(__file__).resolve().parents[2] / "dashboard" / "ne_countries.geojson"
CELL_CACHE = Path(__file__).resolve().parents[2] / "data" / "processed" / "land_cells_res5.json"


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
def population() -> EndpointPopulation:
    """Hand-built tiny endpoint set for the integration tests.

    Pre-2026-04-17 used ``EndpointPopulation.from_terminal_registry``
    against ``src/vantage/config/terminals.json``; that JSON was
    deleted in the data-pipeline consolidation and ``from_terminal_registry``
    now requires explicit ``destinations``. Building the population
    inline keeps this test file self-contained — no hidden dependency
    on a generated config artifact, and the destination set matches
    the ``data/probe_trace/traceroute/{google,facebook}/`` directories
    so the ``MeasuredGroundDelay`` estimator has data for every
    (PoP, dest) pair the controllers consult.
    """
    sources = (
        Endpoint(name="term_nyc", lat_deg=40.7, lon_deg=-74.0),
        Endpoint(name="term_lon", lat_deg=51.5, lon_deg=-0.1),
        Endpoint(name="term_tok", lat_deg=35.7, lon_deg=139.7),
        Endpoint(name="term_syd", lat_deg=-33.9, lon_deg=151.2),
        Endpoint(name="term_sao", lat_deg=-23.5, lon_deg=-46.6),
    )
    destinations = (
        Endpoint(name="google", lat_deg=37.4, lon_deg=-122.1),
        Endpoint(name="facebook", lat_deg=37.5, lon_deg=-122.2),
    )
    return EndpointPopulation(sources=sources, destinations=destinations)


@pytest.fixture(scope="module")
def endpoints(population: EndpointPopulation) -> dict[str, Endpoint]:
    ep: dict[str, Endpoint] = {s.name: s for s in population.sources}
    ep.update({d.name: d for d in population.destinations})
    return ep


@pytest.fixture(scope="module")
def cell_grid(endpoints: dict[str, Endpoint]) -> CellGrid:
    return CellGrid.from_polygon_coverage(
        LAND_GEOJSON,
        endpoints=[(e.name, e.lat_deg, e.lon_deg) for e in endpoints.values()],
        cache_path=CELL_CACHE,
    )


def _make_context(world: WorldModel, endpoints: dict[str, Endpoint]) -> RunContext:
    """Build a RunContext with a GeographicGroundDelay estimator.

    The 2026-04-17 progressive fix made `_ground_cost` raise on
    missing measurements; the older `MeasuredGroundDelay`
    (traceroute-backed, ~29 PoPs covered) doesn't have data for
    every PoP in the snapshot (~49) and would now KeyError. Using
    the haversine-based GeographicGroundDelay keeps the integration
    test self-contained and guarantees an estimate for every
    (PoP, destination) pair that any controller might consult.
    """
    infra = world.snapshot_at(0, 0.0).infra
    pop_coords = {p.code: (p.lat_deg, p.lon_deg) for p in infra.pops}
    service_locations = {
        ep.name: [{"lat": ep.lat_deg, "lon": ep.lon_deg}]
        for ep in endpoints.values()
        if not ep.name.startswith("term_")
    }
    estimator = GeographicGroundDelay(
        pop_coords=pop_coords,
        service_locations=service_locations,
    )
    return RunContext(
        world=world,
        endpoints=endpoints,
        ground_knowledge=GroundKnowledge(estimator=estimator),
    )


@pytest.mark.integration
class TestE2EPipeline:

    def test_nearest_pop_runs(
        self, world: WorldModel, endpoints: dict[str, Endpoint],
        cell_grid: CellGrid, population: EndpointPopulation,
    ) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(population, demand_per_flow_gbps=0.01)
        controller = create_controller("nearest_pop")
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run_routing(
            context=ctx, cell_grid=cell_grid, traffic=traffic,
            controller=controller, config=config, controller_name="nearest_pop",
        )

        assert result.num_epochs == 1
        assert len(result.epochs[0].flow_outcomes) > 0
        assert result.epochs[0].routed_demand_gbps > 0

    def test_greedy_runs(
        self, world: WorldModel, endpoints: dict[str, Endpoint],
        cell_grid: CellGrid, population: EndpointPopulation,
    ) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(population, demand_per_flow_gbps=0.01)
        controller = ProgressiveController(
            ground_knowledge=ctx.ground_knowledge,
        )
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run_routing(
            context=ctx, cell_grid=cell_grid, traffic=traffic,
            controller=controller, config=config, controller_name="greedy",
        )

        assert result.num_epochs == 1
        assert len(result.epochs[0].flow_outcomes) > 0

    def test_feedback_populates_knowledge(
        self, world: WorldModel, endpoints: dict[str, Endpoint],
        cell_grid: CellGrid, population: EndpointPopulation,
    ) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(population, demand_per_flow_gbps=0.01)
        controller = ProgressiveController(
            ground_knowledge=ctx.ground_knowledge,
        )
        config = RunConfig(num_epochs=2, epoch_interval_s=300.0)

        result = run_routing(
            context=ctx, cell_grid=cell_grid, traffic=traffic,
            controller=controller, config=config, controller_name="greedy",
        )

        all_ground = [f.ground_rtt for e in result.epochs for f in e.flow_outcomes]
        assert any(g > 0 for g in all_ground), "Feedback loop failed"

    def test_flow_delay_decomposition(
        self, world: WorldModel, endpoints: dict[str, Endpoint],
        cell_grid: CellGrid, population: EndpointPopulation,
    ) -> None:
        ctx = _make_context(world, endpoints)
        traffic = UniformGenerator(population, demand_per_flow_gbps=0.01)
        controller = create_controller("nearest_pop")
        config = RunConfig(num_epochs=1, epoch_interval_s=300.0)

        result = run_routing(
            context=ctx, cell_grid=cell_grid, traffic=traffic,
            controller=controller, config=config, controller_name="nearest_pop",
        )

        for flow in result.epochs[0].flow_outcomes:
            expected = flow.satellite_rtt + flow.ground_rtt
            assert abs(flow.total_rtt - expected) < 1e-9
