"""Argus experiment runner — exports results for dashboard visualization."""

import json
import time
from pathlib import Path

from vantage.control.controller import create_controller
from vantage.control.policy.greedy import VantageGreedyController
from vantage.engine import RunConfig, RunContext
from vantage.forward import realize
from vantage.probe import ProbeManager, TrafficDrivenPolicy
from vantage.traffic import EndpointPopulation, RealisticGenerator
from vantage.world.ground import GroundInfrastructure, GroundKnowledge, HaversineDelay
from vantage.world.ground.knowledge import LRUEviction
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parent / "config"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")
DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"

NUM_EPOCHS = 30
EPOCH_INTERVAL = 300.0


def setup():
    ground = GroundInfrastructure(DATA_DIR)
    satellite = SatelliteSegment(
        constellation=XMLConstellationModel(str(STARPERF_XML), dt_s=15.0),
        topology_builder=PlusGridTopology(),
        shell_id=1,
        ground_stations=ground.ground_stations,
        visibility=SphericalAccessModel(),
    )
    world = WorldModel(satellite, ground)

    population = EndpointPopulation.from_terminal_registry(DATA_DIR / "terminals.json")
    endpoints = {}
    for s in population.sources:
        endpoints[s.name] = s
    for d in population.destinations:
        endpoints[d.name] = d

    traffic = RealisticGenerator(population, demand_per_flow_gbps=0.01,
                                 initial_dests_per_terminal=3, new_dest_probability=0.15)
    config = RunConfig(num_epochs=NUM_EPOCHS, epoch_interval_s=EPOCH_INTERVAL)

    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    snapshots = [world.snapshot_at(e, e * EPOCH_INTERVAL) for e in range(NUM_EPOCHS)]
    print(f"{time.perf_counter() - t0:.1f}s")

    return world, ground, endpoints, population, traffic, config, snapshots


def run_nearest_pop(world, ground, endpoints, traffic, config, snapshots):
    ctrl = create_controller("nearest_pop")
    gk = GroundKnowledge(estimator=HaversineDelay())
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)

    epoch_data = []
    for epoch in range(config.num_epochs):
        tables = ctrl.compute_tables(snapshots[epoch])
        demand = traffic.generate(epoch)
        result = realize(tables, snapshots[epoch], demand, ctx)

        flows = result.flow_outcomes
        n = max(len(flows), 1)
        epoch_data.append({
            "avg_rtt": round(sum(f.total_rtt for f in flows) / n, 2),
            "avg_sat": round(sum(f.satellite_rtt for f in flows) / n, 2),
            "avg_gnd": round(sum(f.ground_rtt for f in flows) / n, 2),
            "gk_size": 0,
            "coverage": {},
            "pops_used": len({f.pop_code for f in flows}),
            "n_flows": len(flows),
        })
        print(f"  nearest_pop  epoch {epoch:2d}  rtt={epoch_data[-1]['avg_rtt']:.1f}ms  flows={len(flows)}")

    return {"epochs": epoch_data}


def run_greedy(world, ground, endpoints, traffic, config, snapshots,
               probe_budget: int, key: str):
    pop_names = [p.code for p in ground.pops]
    dest_names = [e.name for e in endpoints.values() if not e.name.startswith("terminal_")]

    gk = GroundKnowledge(estimator=HaversineDelay(), pop_capacity=100, eviction=LRUEviction())
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)
    ctrl = VantageGreedyController(endpoints=endpoints, ground_knowledge=gk)

    target_policy = TrafficDrivenPolicy()
    probe_mgr = ProbeManager(
        ground_truth=HaversineDelay(),
        knowledge=gk,
        pops=ground.pops,
        endpoints=endpoints,
        target_policy=target_policy,
        probe_budget_per_pop=probe_budget,
        passive_sample_rate=0.3,  # 30% sampling — gradual learning
        probe_interval_s=EPOCH_INTERVAL,
    )

    epoch_data = []
    coverage_matrix = []  # per epoch: list of [pop_idx, dest_idx] pairs

    for epoch in range(config.num_epochs):
        t = epoch * EPOCH_INTERVAL
        snapshot = snapshots[epoch]

        demand = traffic.generate(epoch)
        probe_mgr.collect("active_probe", current_time_s=t)
        tables = ctrl.compute_tables(snapshot)
        result = realize(tables, snapshot, demand, ctx)
        probe_mgr.collect("passive_sample", epoch_result=result)
        target_policy.update_stats(probe_mgr.get_stats())

        flows = result.flow_outcomes
        n = max(len(flows), 1)

        # Per-dest coverage count
        coverage = {}
        for d in dest_names:
            coverage[d] = sum(1 for p in pop_names if gk.get(p, d) is not None)

        # Coverage matrix for heatmap
        cached_pairs = []
        for pi, p in enumerate(pop_names):
            for di, d in enumerate(dest_names):
                if gk.get(p, d) is not None:
                    cached_pairs.append([pi, di])

        coverage_matrix.append(cached_pairs)

        epoch_data.append({
            "avg_rtt": round(sum(f.total_rtt for f in flows) / n, 2),
            "avg_sat": round(sum(f.satellite_rtt for f in flows) / n, 2),
            "avg_gnd": round(sum(f.ground_rtt for f in flows) / n, 2),
            "gk_size": gk.total_size(),
            "coverage": coverage,
            "pops_used": len({f.pop_code for f in flows}),
            "n_flows": len(flows),
        })

        print(f"  {key:12s}  epoch {epoch:2d}  rtt={epoch_data[-1]['avg_rtt']:.1f}ms  "
              f"gk={gk.total_size()}  cover={len(cached_pairs)}")

    return {"epochs": epoch_data}, coverage_matrix


def main() -> None:
    world, ground, endpoints, population, traffic, config, snapshots = setup()

    pop_names = [p.code for p in ground.pops]
    dest_names = [e.name for e in endpoints.values() if not e.name.startswith("terminal_")]
    n_flows = len(population.sources) * len(population.destinations)
    print(f"\n{len(pop_names)} PoPs, {len(dest_names)} dests, {n_flows} flows/epoch, {NUM_EPOCHS} epochs\n")

    # Run experiments
    print("Running nearest_pop...")
    nearest = run_nearest_pop(world, ground, endpoints, traffic, config, snapshots)

    print("\nRunning greedy (budget=2)...")
    greedy2, cov_matrix_2 = run_greedy(world, ground, endpoints, traffic, config, snapshots,
                                        probe_budget=2, key="greedy_2")

    print("\nRunning greedy (budget=10)...")
    greedy10, cov_matrix_10 = run_greedy(world, ground, endpoints, traffic, config, snapshots,
                                          probe_budget=10, key="greedy_10")

    # Export JSON for dashboard
    data = {
        "config": {
            "num_epochs": NUM_EPOCHS,
            "n_pops": len(pop_names),
            "n_dests": len(dest_names),
            "n_flows": n_flows,
        },
        "strategies": {
            "nearest_pop": nearest,
            "greedy_2": greedy2,
            "greedy_10": greedy10,
        },
        "pop_names": pop_names,
        "dest_names": dest_names,
        "coverage_matrix": {
            "greedy_2": cov_matrix_2,
            "greedy_10": cov_matrix_10,
        },
    }

    out_path = DASHBOARD_DIR / "data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"\nExported dashboard data → {out_path}")
    print(f"Open dashboard: open {DASHBOARD_DIR / 'index.html'}")

    # Print summary
    base_avg = sum(e["avg_rtt"] for e in nearest["epochs"]) / NUM_EPOCHS
    g2_avg = sum(e["avg_rtt"] for e in greedy2["epochs"][1:]) / (NUM_EPOCHS - 1)
    g10_avg = sum(e["avg_rtt"] for e in greedy10["epochs"][1:]) / (NUM_EPOCHS - 1)
    print(f"\nBaseline: {base_avg:.1f}ms")
    print(f"Greedy-2 (steady): {g2_avg:.1f}ms  ({base_avg - g2_avg:+.1f}ms)")
    print(f"Greedy-10 (steady): {g10_avg:.1f}ms  ({base_avg - g10_avg:+.1f}ms)")


if __name__ == "__main__":
    main()
