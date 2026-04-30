"""Construction helpers for the simulation runtime objects."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vantage.model import CellGrid
from vantage.model.ground import (
    GeographicGroundDelay,
    GroundInfrastructure,
    GroundStation,
    GroundTruth,
)
from vantage.model.network import WorldModel
from vantage.model.satellite import (
    PlusGridTopology,
    SatelliteSegment,
    SphericalAccessModel,
    XMLConstellationModel,
)
from vantage.sim.config import CELL_CACHE, DATA_DIR, EPOCH_S, LAND_GEOJSON, XML, SimConfig
from vantage.traffic import Endpoint, EndpointPopulation, FlowLevelGenerator

__all__ = ["SimulationRuntime", "build_runtime"]


@dataclass(slots=True)
class SimulationRuntime:
    world: WorldModel
    population: EndpointPopulation
    endpoints: dict[str, Endpoint]
    cell_grid: CellGrid
    geo_delay: GeographicGroundDelay
    ground_truth: GroundTruth
    gs_by_id: dict[str, GroundStation]
    pop_gs_list: dict[str, list[str]]
    pop_list: list[str]
    svc_names: list[str]
    dst_locs: dict[str, Any]
    dst_weights: dict[str, float]
    traffic: FlowLevelGenerator
    prune_summary: str | None = None


def build_runtime(config: SimConfig) -> SimulationRuntime:
    ground = GroundInfrastructure.from_config(DATA_DIR)
    prune_summary = None
    if config.max_gs_per_pop > 0:
        by_pop: dict[str, list] = defaultdict(list)
        for edge in ground.gs_pop_edges:
            by_pop[edge.pop_code].append(edge)
        kept: list = []
        for edges in by_pop.values():
            edges.sort(key=lambda edge: edge.backhaul_delay)
            kept.extend(edges[: config.max_gs_per_pop])
        ground = ground.with_gs_pop_edges(tuple(kept))
        kept_per_pop: dict[str, int] = {}
        for edge in ground.gs_pop_edges:
            kept_per_pop[edge.pop_code] = kept_per_pop.get(edge.pop_code, 0) + 1
        prune_summary = (
            f"Pruned GS attachments: max {config.max_gs_per_pop} GS/PoP -> "
            f"{len(ground.gs_pop_edges)} edges; "
            f"PoPs w/ GSs: {len(kept_per_pop)}/{len(ground.pops)}"
        )

    satellite = SatelliteSegment(
        constellation=XMLConstellationModel(str(XML), dt_s=15.0),
        topology_builder=PlusGridTopology(),
        shell_id=1,
        ground_stations=ground.ground_stations,
        visibility=SphericalAccessModel(),
    )
    world = WorldModel(satellite, ground)

    with open(DATA_DIR / "service_prefixes.json") as file:
        svc_data = json.load(file)
    dst_locs = {
        name: data["locations"]
        for name, data in svc_data.items()
        if data.get("locations")
    }
    dst_weights = {
        name: data.get("traffic_weight", 1.0)
        for name, data in svc_data.items()
        if data.get("locations")
    }
    svc_names = sorted(dst_locs.keys())
    dests = [
        Endpoint(name, locations[0]["lat"], locations[0]["lon"])
        for name, locations in dst_locs.items()
    ]

    population = EndpointPopulation.from_starlink_users(
        DATA_DIR / "starlink_users.json",
        DATA_DIR / "world_cities.json",
        destinations=tuple(dests),
        user_scale=config.user_scale,
    )
    endpoints = {source.name: source for source in population.sources}
    endpoints.update({dest.name: dest for dest in population.destinations})
    cell_grid = CellGrid.from_polygon_coverage(
        LAND_GEOJSON,
        endpoints=[
            (endpoint.name, endpoint.lat_deg, endpoint.lon_deg)
            for endpoint in endpoints.values()
        ],
        cache_path=CELL_CACHE,
    )
    geo_delay = GeographicGroundDelay(
        pop_coords={pop.code: (pop.lat_deg, pop.lon_deg) for pop in ground.pops},
        service_locations=dst_locs,
    )
    ground_truth = GroundTruth(
        prior=geo_delay,
        seed_base=config.seeds.ground_seed,
    )
    gs_by_id = {gs.gs_id: gs for gs in world.ground_stations}

    snap0 = world.snapshot_at(0, 0.0)
    pop_gs_list: dict[str, list[str]] = {}
    for edge in snap0.infra.gs_pop_edges:
        pop_gs_list.setdefault(edge.pop_code, []).append(edge.gs_id)
    pop_list = [pop.code for pop in ground.pops]

    traffic = FlowLevelGenerator(
        population,
        config_dir=DATA_DIR,
        epoch_interval_s=EPOCH_S,
        dst_weights=dst_weights,
        dst_locations=dst_locs,
        seed=config.seeds.traffic_seed,
    )

    return SimulationRuntime(
        world=world,
        population=population,
        endpoints=endpoints,
        cell_grid=cell_grid,
        geo_delay=geo_delay,
        ground_truth=ground_truth,
        gs_by_id=gs_by_id,
        pop_gs_list=pop_gs_list,
        pop_list=pop_list,
        svc_names=svc_names,
        dst_locs=dst_locs,
        dst_weights=dst_weights,
        traffic=traffic,
        prune_summary=prune_summary,
    )
