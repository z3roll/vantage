"""Parallel experiment runner for the routing-plane execution path.

Mirrors :mod:`vantage.main`'s ``setup`` step but drives the new
:func:`vantage.engine.run_routing.run_routing` loop instead of the
legacy ``CostTables``-based ``run``. Prints a text summary that can
be eyeballed against ``main.py``'s baseline output for sanity
checking before we flip the production pipeline over.

Run it with::

    uv run python -m vantage.main_routing

Scope:
    * Only exercises the ``nearest_pop`` controller — that is the
      only one with a ``compute_routing_plane`` implementation today.
    * No dashboard JSON export; ``main.py`` remains the production
      entry point until this path is vetted.
    * No max-min fair-share throttling; every flow that has a route
      is served at full demand. Capacity utilization is *measured*
      but not enforced.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import CellGrid
from vantage.engine import RunConfig, RunContext, run_routing
from vantage.traffic import EndpointPopulation, RealisticGenerator
from vantage.world.ground import (
    GroundInfrastructure,
    GroundKnowledge,
    MeasuredGroundDelay,
)
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parent / "config"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACEROUTE_DIR = PROJECT_ROOT / "data" / "probe_trace" / "traceroute"

# Constellation XML location. Environment override wins so CI and
# non-Mac workstations don't have to patch the module. TODO: the
# legacy main.py still hardcodes the same path; fold these into a
# shared config module when the routing-plane path replaces main.py.
_DEFAULT_STARPERF_XML = Path(
    "/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml"
)
STARPERF_XML = Path(
    os.environ.get("ARGUS_STARLINK_XML", str(_DEFAULT_STARPERF_XML))
)

NUM_EPOCHS = 30
EPOCH_INTERVAL = 300.0


def _setup():
    """Build the world / endpoints / traffic / cell grid / ground truth once.

    The ground-truth measurement table is loaded here (not in ``main``)
    because it acts as a *filter* on the destination set: traffic is
    only generated for destinations that have a real measurement on
    at least one PoP. This keeps the baseline's "flows/epoch" number
    honest — we never generate flows we have no way of scoring.
    """
    if not STARPERF_XML.exists():
        raise FileNotFoundError(
            f"Starlink constellation XML not found at {STARPERF_XML!s}. "
            "Set ARGUS_STARLINK_XML to override the default location."
        )
    ground = GroundInfrastructure(DATA_DIR)
    satellite = SatelliteSegment(
        constellation=XMLConstellationModel(str(STARPERF_XML), dt_s=15.0),
        topology_builder=PlusGridTopology(),
        shell_id=1,
        ground_stations=ground.ground_stations,
        visibility=SphericalAccessModel(),
    )
    world = WorldModel(satellite, ground)

    # Ground-truth measurement table. Its destination set defines the
    # *only* destinations the traffic generator is allowed to produce.
    ground_truth = MeasuredGroundDelay.from_traceroute_dir(TRACEROUTE_DIR)
    measured_dests = ground_truth.destinations()

    # Load the raw population, then drop any destination without a
    # measurement. Sources (user terminals) are untouched.
    raw_population = EndpointPopulation.from_terminal_registry(
        DATA_DIR / "terminals.json"
    )
    filtered_destinations = tuple(
        d for d in raw_population.destinations if d.name in measured_dests
    )
    dropped_count = len(raw_population.destinations) - len(filtered_destinations)
    population = EndpointPopulation(
        sources=raw_population.sources,
        destinations=filtered_destinations,
    )
    print(
        f"  destination filter: {len(raw_population.destinations)} → "
        f"{len(filtered_destinations)} "
        f"(dropped {dropped_count} unmeasured)"
    )

    # Endpoints dict only contains sources + measured destinations.
    # No unmeasured destination ever appears in a FlowKey.
    endpoints: dict[str, object] = {s.name: s for s in population.sources}
    endpoints.update({d.name: d for d in population.destinations})

    # CellGrid is built once from every endpoint name we might see as a
    # flow source; the legacy forward never needed this because PoP
    # assignment was per-flow.
    cell_grid = CellGrid.from_endpoints(
        [(e.name, e.lat_deg, e.lon_deg) for e in endpoints.values()]
    )

    traffic = RealisticGenerator(
        population,
        demand_per_flow_gbps=0.01,
        initial_dests_per_terminal=3,
        new_dest_probability=0.15,
    )
    config = RunConfig(num_epochs=NUM_EPOCHS, epoch_interval_s=EPOCH_INTERVAL)

    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as pool:
        snapshots = list(
            pool.map(
                lambda e: world.snapshot_at(e, e * EPOCH_INTERVAL),
                range(NUM_EPOCHS),
            )
        )
    print(f"{time.perf_counter() - t0:.1f}s")

    return (
        world, ground, endpoints, cell_grid, traffic, config, snapshots,
        ground_truth,
    )


def main() -> None:
    (
        world, ground, endpoints, cell_grid, traffic, config, snapshots,
        ground_truth,
    ) = _setup()

    pops = [p.code for p in ground.pops]
    dests = [e.name for e in endpoints.values() if not e.name.startswith("terminal_")]
    n_terminals = sum(1 for e in endpoints.values() if e.name.startswith("terminal_"))
    n_flows = n_terminals * len(dests)
    print(
        f"\n{len(pops)} PoPs, {len(dests)} dests, {len(cell_grid)} cells, "
        f"{n_flows} flows/epoch, {NUM_EPOCHS} epochs"
    )
    print(
        f"  measured pairs: {len(ground_truth)} "
        f"({len(ground_truth.pops())} PoPs × "
        f"{len(ground_truth.destinations())} services)\n"
    )

    gk = GroundKnowledge(estimator=ground_truth)
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)
    ctrl = NearestPoPController()

    print("Running nearest_pop via RoutingPlane...")
    result = run_routing(
        context=ctx,
        cell_grid=cell_grid,
        traffic=traffic,
        controller=ctrl,
        config=config,
    )

    for epoch_result, caps in zip(result.epochs, result.capacity, strict=True):
        flows = epoch_result.flow_outcomes
        n = max(len(flows), 1)
        avg_rtt = sum(f.total_rtt for f in flows) / n
        avg_sat = sum(f.satellite_rtt for f in flows) / n
        avg_gnd = sum(f.ground_rtt for f in flows) / n
        print(
            f"  epoch {caps.epoch:2d}  rtt={avg_rtt:5.1f}ms  "
            f"sat={avg_sat:5.1f}ms  gnd={avg_gnd:5.1f}ms  "
            f"flows={len(flows):4d}  "
            f"isl_max={caps.max_isl_utilization:5.3f}  "
            f"sat_max={caps.max_sat_feeder_utilization:5.3f}  "
            f"gs_max={caps.max_gs_feeder_utilization:5.3f}  "
            f"sat_isl={caps.saturated_isl_count:3d}"
        )

    print(
        f"\nWall: {result.wall_time_s:.1f}s  "
        f"avg_total_rtt={result.avg_total_rtt:.1f}ms"
    )


if __name__ == "__main__":
    main()
