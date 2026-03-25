"""Argus experiment runner — probe-based ground delay learning."""

import time
from pathlib import Path

from vantage.control.controller import create_controller
from vantage.control.policy.greedy import VantageGreedyController
from vantage.engine import RunConfig, RunContext
from vantage.engine_feedback import GroundDelayFeedback
from vantage.forward import realize
from vantage.probe import (
    HotListPolicy,
    LRUEviction,
    ProbeManager,
    TrafficDrivenPolicy,
)
from vantage.traffic import EndpointPopulation, UniformGenerator
from vantage.world.ground import GroundInfrastructure, GroundKnowledge, HaversineDelay
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parent / "config"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")


def main() -> None:
    # ── Setup ───────────────────────────────────────────────
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
    config = RunConfig(num_epochs=15, epoch_interval_s=300.0)
    n_pops = len(ground.pops)
    n_dests = len(population.destinations)

    # Precompute snapshots
    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    snapshots = [world.snapshot_at(e, e * config.epoch_interval_s) for e in range(config.num_epochs)]
    print(f"{time.perf_counter() - t0:.1f}s")

    # ── GroundKnowledge: starts EMPTY (no oracle) ───────────
    gk = GroundKnowledge()  # no estimator — cold start
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)
    ctrl = VantageGreedyController(endpoints=endpoints, ground_knowledge=gk)

    # ── Probe Manager ───────────────────────────────────────
    # Traffic-driven: after first epoch, Controller knows global hot dests
    target_policy = TrafficDrivenPolicy()
    eviction = LRUEviction()
    probe_mgr = ProbeManager(
        ground_truth=HaversineDelay(),
        knowledge=gk,
        pops=ground.pops,
        endpoints=endpoints,
        target_policy=target_policy,
        eviction_policy=eviction,
        pop_cache_capacity=100,      # each PoP caches up to 100 destinations
        probe_budget_per_pop=1,      # each PoP probes 1 dest per cycle
        passive_sample_rate=1.0,     # sample all flows (simulation)
        probe_interval_s=300.0,      # probe every epoch (= epoch_interval)
    )

    # ── Run ─────────────────────────────────────────────────
    print(f"\nGreedy with probe-based learning ({config.num_epochs} epochs, "
          f"{n_pops} PoPs, {n_dests} dests)")
    print(f"Cold start: no ground delay data. Probe budget: 5/PoP/cycle.\n")

    print(f"{'Ep':>3s}  {'AvgRTT':>8s}  {'Sat':>7s}  {'Gnd':>7s}  "
          f"{'GK':>5s}  {'Passive':>7s}  {'Active':>6s}  {'PoPs':>4s}")
    print("-" * 62)

    for epoch in range(config.num_epochs):
        t = epoch * config.epoch_interval_s
        snapshot = snapshots[epoch]
        demand = traffic.generate(epoch)

        # 1. Active probe (PoPs probe destinations per Controller instruction)
        n_active = probe_mgr.collect("active_probe", current_time_s=t)

        # 2. Sync PoP caches → centralized GroundKnowledge
        probe_mgr.sync_to_knowledge()

        # 3. Controller reads current data, builds cost tables
        tables = ctrl.compute_tables(snapshot)

        # 4. User traffic routed (terminal-side PoP selection)
        result = realize(tables, snapshot, demand, ctx)

        # 5. Passive sampling from user traffic
        n_passive = probe_mgr.collect("passive_sample", epoch_result=result)

        # 6. Sync passive results too
        probe_mgr.sync_to_knowledge()

        # 7. Update traffic-driven policy with global stats
        target_policy.update_stats(probe_mgr.get_stats())

        # ── Metrics ─────────────────────────────────────────
        rtts = [f.total_rtt for f in result.flow_outcomes]
        avg_rtt = sum(rtts) / len(rtts) if rtts else 0.0
        avg_sat = sum(f.satellite_rtt for f in result.flow_outcomes) / max(len(result.flow_outcomes), 1)
        avg_gnd = sum(f.ground_rtt for f in result.flow_outcomes) / max(len(result.flow_outcomes), 1)
        pops_used = len({f.pop_code for f in result.flow_outcomes})
        gk_size = len(gk._cache)

        print(f"{epoch:3d}  {avg_rtt:7.1f}ms  {avg_sat:6.1f}ms  {avg_gnd:6.1f}ms  "
              f"{gk_size:5d}  +{n_passive:<6d}  +{n_active:<5d}  {pops_used:4d}")

    # ── Summary ─────────────────────────────────────────────
    print(f"\nGroundKnowledge: {len(gk._cache)} entries")
    cache_summary = probe_mgr.get_cache_summary()
    pops_with_data = sum(1 for v in cache_summary.values() if v > 0)
    print(f"PoPs with data: {pops_with_data}/{n_pops}")
    print(f"\nPer-dest coverage:")
    for d in population.destinations:
        n = sum(1 for p in ground.pops if gk.get(p.code, d.name) is not None)
        print(f"  {d.name:15s}  {n}/{n_pops} PoPs")


if __name__ == "__main__":
    main()
