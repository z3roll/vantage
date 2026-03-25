"""Argus experiment runner — Greedy learning analysis."""

import time
from pathlib import Path

from vantage.control.policy.greedy import VantageGreedyController
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
    print()

    print(f"{'Epoch':>5s}  {'AvgRTT':>8s}  {'Sat':>7s}  {'Gnd':>7s}  "
          f"{'Cache':>7s}  {'CtrlTime':>9s}  {'FwdTime':>8s}  {'PoPs':>5s}")
    print("-" * 72)

    for epoch in range(config.num_epochs):
        snapshot = snapshots[epoch]
        demand = traffic.generate(epoch)

        cache_before = len(gk._cache)

        t0 = time.perf_counter()
        tables = ctrl.compute_tables(snapshot)
        dt_ctrl = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = realize(tables, snapshot, demand, ctx)
        dt_fwd = time.perf_counter() - t0

        feedback.observe(result)

        cache_after = len(gk._cache)

        rtts = [f.total_rtt for f in result.flow_outcomes]
        avg_rtt = sum(rtts) / len(rtts) if rtts else 0.0
        avg_sat = sum(f.satellite_rtt for f in result.flow_outcomes) / len(result.flow_outcomes)
        avg_gnd = sum(f.ground_rtt for f in result.flow_outcomes) / len(result.flow_outcomes)
        pops_used = len({f.pop_code for f in result.flow_outcomes})

        print(f"{epoch:5d}  {avg_rtt:7.1f}ms  {avg_sat:6.1f}ms  {avg_gnd:6.1f}ms  "
              f"{cache_after:4d}/{n_pops * n_dests:<3d}  "
              f"{dt_ctrl * 1000:7.1f}ms  {dt_fwd * 1000:6.1f}ms  {pops_used:5d}")

    print(f"\nCache final: {len(gk._cache)} entries")
    print(f"Per-destination coverage:")
    for d in population.destinations:
        pops_with_data = sum(1 for p in ground.pops if gk.get(p.code, d.name) is not None)
        print(f"  {d.name:15s}  {pops_with_data}/{n_pops} PoPs cached")


if __name__ == "__main__":
    main()
