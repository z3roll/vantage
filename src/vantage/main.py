"""Vantage TE experiment runner — Greedy learning analysis."""

import time
from pathlib import Path

from vantage.control.policy.greedy import VantageGreedyController
from vantage.domain import EpochResult
from vantage.engine import RunConfig, RunContext
from vantage.engine_feedback import GroundDelayFeedback
from vantage.forward import realize
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
    config = RunConfig(num_epochs=10, epoch_interval_s=300.0)
    n_flows = len(population.sources) * len(population.destinations)
    n_pops = len(ground.pops)
    n_dests = len(population.destinations)

    # Precompute snapshots
    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    snapshots = [world.snapshot_at(e, e * config.epoch_interval_s) for e in range(config.num_epochs)]
    print(f"{time.perf_counter() - t0:.1f}s")

    # ── Greedy with learning ────────────────────────────────
    gk = GroundKnowledge(estimator=HaversineDelay())
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)
    ctrl = VantageGreedyController(endpoints=endpoints, ground_knowledge=gk)
    feedback = GroundDelayFeedback(gk)

    print(f"\nGreedy learning over {config.num_epochs} epochs "
          f"({n_flows} flows/epoch, {n_pops} PoPs, {n_dests} destinations)")
    print(f"Cache capacity: {n_pops} x {n_dests} = {n_pops * n_dests} entries")
    print()

    print(f"{'Epoch':>5s}  {'AvgRTT':>8s}  {'Sat':>7s}  {'Gnd':>7s}  "
          f"{'Cache':>7s}  {'Fill%':>6s}  {'Fallback':>8s}  {'Greedy':>7s}  {'Time':>7s}")
    print("-" * 78)

    for epoch in range(config.num_epochs):
        snapshot = snapshots[epoch]
        demand = traffic.generate(epoch)

        # Count cache entries before this epoch
        cache_before = len(gk._cache)

        t0 = time.perf_counter()
        intent = ctrl.optimize(snapshot, demand)
        result = realize(intent, snapshot, demand, ctx)
        feedback.observe(result)
        dt = time.perf_counter() - t0

        # Cache entries after
        cache_after = len(gk._cache)
        fill_pct = cache_after / (n_pops * n_dests) * 100

        # Count flows that went through fallback vs greedy search
        # Fallback = flow routed to nearest PoP (no cache data available)
        # We can detect this by checking: did the flow's PoP match nearest PoP?
        rtts = [f.total_rtt for f in result.flow_outcomes]
        sat_rtts = [f.satellite_rtt for f in result.flow_outcomes]
        gnd_rtts = [f.ground_rtt for f in result.flow_outcomes]
        avg_rtt = sum(rtts) / len(rtts) if rtts else 0.0
        avg_sat = sum(sat_rtts) / len(sat_rtts) if sat_rtts else 0.0
        avg_gnd = sum(gnd_rtts) / len(gnd_rtts) if gnd_rtts else 0.0

        # Count unique PoPs used (proxy for exploration)
        pops_used = len({f.pop_code for f in result.flow_outcomes})

        # New cache entries this epoch
        new_entries = cache_after - cache_before

        print(f"{epoch:5d}  {avg_rtt:7.1f}ms  {avg_sat:6.1f}ms  {avg_gnd:6.1f}ms  "
              f"{cache_after:4d}/{n_pops * n_dests:<3d}  {fill_pct:5.1f}%  "
              f"+{new_entries:<7d}  {pops_used:4d}pop  {dt * 1000:5.0f}ms")

    # ── Cache analysis ──────────────────────────────────────
    print(f"\nCache final state: {len(gk._cache)} entries")
    print(f"\nPer-destination coverage:")
    for d in population.destinations:
        pops_with_data = sum(1 for p in ground.pops if gk.get(p.code, d.name) is not None)
        print(f"  {d.name:15s}  {pops_with_data}/{n_pops} PoPs cached")

    # ── Epoch-over-epoch improvement ────────────────────────
    print(f"\nLearning curve:")
    # Show which PoPs are NOT in cache (exploration gap)
    cached_pops = {pop_code for (pop_code, _) in gk._cache}
    all_pops = {p.code for p in ground.pops}
    missing = all_pops - cached_pops
    if missing:
        print(f"\nPoPs never explored ({len(missing)}/{n_pops}):")
        for code in sorted(missing):
            p = next(p for p in ground.pops if p.code == code)
            print(f"  {code:10s}  ({p.name}, {p.lat_deg:.1f}, {p.lon_deg:.1f})")


if __name__ == "__main__":
    main()
