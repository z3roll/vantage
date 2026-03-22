"""Vantage TE experiment runner."""

import time
from pathlib import Path

from vantage.analysis import compare_controllers, compute_latency_stats, compute_segment_breakdown
from vantage.control.controller import create_controller
from vantage.control.policy.greedy import VantageGreedyController
from vantage.domain import EpochResult
from vantage.engine import RunConfig, RunContext, RunResult
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
    strategies = ["nearest_pop", "ground_only", "greedy"]

    # ── Precompute snapshots ──────────────────────────────────
    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    snapshots = []
    for epoch in range(config.num_epochs):
        t_s = epoch * config.epoch_interval_s
        snapshots.append(world.snapshot_at(epoch, t_s))
    dt_pre = time.perf_counter() - t0
    print(f"{dt_pre:.1f}s ({config.num_epochs} timeslots, {dt_pre/config.num_epochs*1000:.0f}ms each)")

    # ── Prepare controllers ─────────────────────────────────
    controllers = {}
    feedbacks = {}
    contexts = {}
    epoch_results: dict[str, list[EpochResult]] = {}

    for name in strategies:
        gk = GroundKnowledge(estimator=HaversineDelay())
        contexts[name] = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)
        if name == "greedy":
            controllers[name] = VantageGreedyController(endpoints=endpoints, ground_knowledge=gk)
            feedbacks[name] = GroundDelayFeedback(gk)
        else:
            controllers[name] = create_controller(name, endpoints=endpoints)
            feedbacks[name] = None
        epoch_results[name] = []

    # ── Epoch loop: all strategies side by side ─────────────
    header = "  ".join(f"{n:>15s}" for n in strategies)
    print(f"{'Epoch':>5s}  {header}")
    print("-" * (8 + 17 * len(strategies)))

    for epoch in range(config.num_epochs):
        parts = []
        snapshot = snapshots[epoch]
        demand = traffic.generate(epoch)
        for name in strategies:
            ctx = contexts[name]

            t0 = time.perf_counter()
            intent = controllers[name].optimize(snapshot, demand)
            result = realize(intent, snapshot, demand, ctx)
            if feedbacks[name] is not None:
                feedbacks[name].observe(result)
            dt = time.perf_counter() - t0

            epoch_results[name].append(result)

            rtts = [f.total_rtt for f in result.flow_outcomes]
            avg = sum(rtts) / len(rtts) if rtts else 0.0
            parts.append(f"{avg:6.1f}ms {dt*1000:5.0f}ms")

        line = "  ".join(f"{p:>15s}" for p in parts)
        print(f"{epoch:5d}  {line}")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'Strategy':15s} {'Mean':>7s} {'P50':>7s} {'P95':>7s} {'Sat':>7s} {'Gnd':>7s}")
    print("-" * 52)
    for name in strategies:
        stats = compute_latency_stats(epoch_results[name])
        seg = compute_segment_breakdown(epoch_results[name])
        print(f"{name:15s} {stats.mean:7.1f} {stats.p50:7.1f} "
              f"{stats.p95:7.1f} {seg.satellite:7.1f} {seg.ground:7.1f}")

    print(f"\n{'vs nearest_pop':15s} {'Improv':>8s} {'%':>6s} {'Better':>7s} {'Worse':>6s}")
    print("-" * 46)
    base = epoch_results["nearest_pop"]
    for name in strategies:
        if name == "nearest_pop":
            continue
        cmp = compare_controllers(base, epoch_results[name], "nearest_pop", name)
        print(f"{name:15s} {cmp.improvement:7.1f}ms {cmp.improvement_pct:5.1f}% "
              f"{cmp.flows_improved:7d} {cmp.flows_worsened:6d}")


if __name__ == "__main__":
    main()
