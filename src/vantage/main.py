"""Argus experiment runner — probe-based learning comparison."""

import time
from pathlib import Path

from vantage.control.controller import create_controller
from vantage.control.policy.greedy import VantageGreedyController
from vantage.engine import RunConfig, RunContext
from vantage.forward import realize
from vantage.probe import ProbeManager, TrafficDrivenPolicy
from vantage.traffic import EndpointPopulation, UniformGenerator
from vantage.world.ground import GroundInfrastructure, GroundKnowledge, HaversineDelay
from vantage.world.ground.knowledge import LRUEviction
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parent / "config"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")

NUM_EPOCHS = 15
EPOCH_INTERVAL = 300.0


def setup():
    """Build world, endpoints, traffic (shared across experiments)."""
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

    traffic = UniformGenerator(population, demand_per_flow_gbps=0.01)
    config = RunConfig(num_epochs=NUM_EPOCHS, epoch_interval_s=EPOCH_INTERVAL)

    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    snapshots = [world.snapshot_at(e, e * EPOCH_INTERVAL) for e in range(NUM_EPOCHS)]
    print(f"{time.perf_counter() - t0:.1f}s")

    return world, ground, endpoints, population, traffic, config, snapshots


def run_nearest_pop(world, ground, endpoints, traffic, config, snapshots):
    """Baseline: no ground knowledge, nearest PoP always."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Nearest PoP (baseline, no ground knowledge)")
    print("=" * 70)

    ctrl = create_controller("nearest_pop")
    # Estimator needed by forward for actual delay computation (ground truth)
    gk = GroundKnowledge(estimator=HaversineDelay())
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)

    avgs = []
    for epoch in range(config.num_epochs):
        tables = ctrl.compute_tables(snapshots[epoch])
        demand = traffic.generate(epoch)
        result = realize(tables, snapshots[epoch], demand, ctx)

        rtts = [f.total_rtt for f in result.flow_outcomes]
        avg = sum(rtts) / len(rtts) if rtts else 0.0
        avgs.append(avg)

    print(f"{'Ep':>3s}  {'AvgRTT':>8s}")
    print("-" * 14)
    for i, avg in enumerate(avgs):
        print(f"{i:3d}  {avg:7.1f}ms")
    return avgs


def run_greedy(world, ground, endpoints, traffic, config, snapshots,
               probe_budget: int, label: str):
    """Greedy with probe-based learning."""
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {label} (probe budget={probe_budget}/PoP/cycle)")
    print(f"{'=' * 70}")

    # Estimator = ground truth for forward's delay computation (always available)
    # Cache = what controller knows (starts empty, fills via probing)
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
        passive_sample_rate=1.0,
        probe_interval_s=EPOCH_INTERVAL,
    )

    n_pops = len(ground.pops)
    n_dests = len([e for e in endpoints.values() if not e.name.startswith("terminal_")])

    print(f"{'Ep':>3s}  {'AvgRTT':>8s}  {'Sat':>7s}  {'Gnd':>7s}  "
          f"{'GK':>6s}  {'New':>5s}  {'Cover':>8s}  {'PoPs':>4s}")
    print("-" * 62)

    avgs = []
    for epoch in range(config.num_epochs):
        t = epoch * EPOCH_INTERVAL
        snapshot = snapshots[epoch]
        demand = traffic.generate(epoch)

        # Active probe
        n_new = probe_mgr.collect("active_probe", current_time_s=t)

        # Controller
        tables = ctrl.compute_tables(snapshot)

        # Forward
        result = realize(tables, snapshot, demand, ctx)

        # Passive sample + update policy
        n_passive = probe_mgr.collect("passive_sample", epoch_result=result)
        n_new += n_passive
        target_policy.update_stats(probe_mgr.get_stats())

        # Metrics
        rtts = [f.total_rtt for f in result.flow_outcomes]
        avg = sum(rtts) / len(rtts) if rtts else 0.0
        avg_sat = sum(f.satellite_rtt for f in result.flow_outcomes) / max(len(result.flow_outcomes), 1)
        avg_gnd = sum(f.ground_rtt for f in result.flow_outcomes) / max(len(result.flow_outcomes), 1)
        pops_used = len({f.pop_code for f in result.flow_outcomes})
        gk_size = gk.total_size()

        # Coverage: how many (pop, dest) pairs have data
        coverage = gk_size
        total_pairs = n_pops * n_dests

        avgs.append(avg)
        print(f"{epoch:3d}  {avg:7.1f}ms  {avg_sat:6.1f}ms  {avg_gnd:6.1f}ms  "
              f"{gk_size:6d}  +{n_new:<4d}  {coverage}/{total_pairs:<4d}  {pops_used:4d}")

    # Coverage summary
    print(f"\nPer-dest coverage:")
    for name in [e.name for e in endpoints.values() if not e.name.startswith("terminal_")]:
        n = sum(1 for p in ground.pops if gk.get(p.code, name) is not None)
        print(f"  {name:15s}  {n}/{n_pops}")

    return avgs


def main() -> None:
    world, ground, endpoints, population, traffic, config, snapshots = setup()

    n_flows = len(population.sources) * len(population.destinations)
    n_pops = len(ground.pops)
    n_dests = len(population.destinations)
    print(f"\n{n_pops} PoPs, {n_dests} dests, {n_flows} flows/epoch, {NUM_EPOCHS} epochs")

    # Run experiments
    baseline = run_nearest_pop(world, ground, endpoints, traffic, config, snapshots)
    greedy_slow = run_greedy(world, ground, endpoints, traffic, config, snapshots,
                             probe_budget=1, label="Greedy (slow probe)")
    greedy_fast = run_greedy(world, ground, endpoints, traffic, config, snapshots,
                             probe_budget=5, label="Greedy (fast probe)")

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Ep':>3s}  {'Nearest':>9s}  {'Greedy-1':>9s}  {'Greedy-5':>9s}  "
          f"{'Δ1':>7s}  {'Δ5':>7s}")
    print("-" * 50)
    for i in range(NUM_EPOCHS):
        d1 = baseline[i] - greedy_slow[i]
        d5 = baseline[i] - greedy_fast[i]
        print(f"{i:3d}  {baseline[i]:8.1f}ms  {greedy_slow[i]:8.1f}ms  "
              f"{greedy_fast[i]:8.1f}ms  {d1:+6.1f}ms  {d5:+6.1f}ms")

    print(f"\nAvg improvement (Greedy-1): {sum(baseline)/len(baseline) - sum(greedy_slow)/len(greedy_slow):+.1f}ms")
    print(f"Avg improvement (Greedy-5): {sum(baseline)/len(baseline) - sum(greedy_fast)/len(greedy_fast):+.1f}ms")


if __name__ == "__main__":
    main()
