"""Radar-driven experiment: service-class traffic + profiled ground delay.

Uses Cloudflare Radar hourly demand data and RIPE Atlas anchor measurements
to model time-varying service-class traffic with profile-based ground RTT.

Run:
    uv run python -m vantage.main_service
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from vantage.common import haversine_km
from vantage.control.controller import create_controller
from vantage.domain import Endpoint
from vantage.domain.service import SERVICE_CLASSES
from vantage.engine import RunConfig, RunContext
from vantage.forward import realize
from vantage.traffic import PopHourlyDemand, ServiceMixSchedule, TimeVaryingServiceMixGenerator
from vantage.traffic.service_population import ServiceClassPopulation
from vantage.world.ground import GroundInfrastructure, GroundKnowledge
from vantage.world.ground.profiled_delay import create_profiled_delay
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parent / "config"
RADAR_DIR = Path(__file__).resolve().parents[2] / "data" / "model_inputs" / "radar"
STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")
DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"

NUM_EPOCHS = 48  # 2 days → full diurnal cycle
EPOCH_INTERVAL = 3600.0  # 1 hour


def _build_terminal_pop_mapping(
    terminals: tuple[Endpoint, ...],
    pops: tuple,
) -> dict[str, str]:
    """Map each terminal to its nearest PoP by Haversine distance."""
    mapping: dict[str, str] = {}
    for t in terminals:
        best_pop = min(
            pops,
            key=lambda p: haversine_km(t.lat_deg, t.lon_deg, p.lat_deg, p.lon_deg),
        )
        mapping[t.name] = best_pop.code
    return mapping


def setup():
    """Build world, traffic generator, and profiled delay model."""
    ground = GroundInfrastructure(DATA_DIR)
    satellite = SatelliteSegment(
        constellation=XMLConstellationModel(str(STARPERF_XML), dt_s=15.0),
        topology_builder=PlusGridTopology(),
        shell_id=1,
        ground_stations=ground.ground_stations,
        visibility=SphericalAccessModel(),
    )
    world = WorldModel(satellite, ground)

    # Load Radar data
    pop_hourly_demand = PopHourlyDemand.from_csv(RADAR_DIR / "pop_hourly_demand_radar.csv")
    service_mix = ServiceMixSchedule.from_csv(RADAR_DIR / "service_class_hourly_mix_baseline.csv")

    # Build population + mapping
    population = ServiceClassPopulation.from_terminal_registry(DATA_DIR / "terminals.json")
    terminal_pop_mapping = _build_terminal_pop_mapping(population.sources, ground.pops)

    # Endpoints: terminals only (service classes have no lat/lon)
    endpoints: dict[str, Endpoint] = {s.name: s for s in population.sources}

    # Traffic generator
    simulation_start_utc = datetime(2026, 3, 23, 0, 0, 0, tzinfo=UTC)
    traffic = TimeVaryingServiceMixGenerator(
        population=population,
        pop_hourly_demand=pop_hourly_demand,
        service_mix=service_mix,
        terminal_pop_mapping=terminal_pop_mapping,
        base_demand_gbps=0.01,
        simulation_start_utc=simulation_start_utc,
        epoch_interval_s=EPOCH_INTERVAL,
    )

    # Profiled ground delay model
    profiled_delay = create_profiled_delay()

    config = RunConfig(num_epochs=NUM_EPOCHS, epoch_interval_s=EPOCH_INTERVAL)

    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    snapshots = [world.snapshot_at(e, e * EPOCH_INTERVAL) for e in range(NUM_EPOCHS)]
    print(f"{time.perf_counter() - t0:.1f}s")

    return (
        world, ground, endpoints, population, traffic, config, snapshots,
        profiled_delay, simulation_start_utc,
    )


def run_nearest_pop_service(
    world, ground, endpoints, traffic, config, snapshots,
    profiled_delay, simulation_start_utc,
):
    """Nearest-PoP baseline with service-class traffic and profiled ground delay."""
    gk = GroundKnowledge()
    ctrl = create_controller(
        "service_aware",
        service_classes=SERVICE_CLASSES,
        service_ground_delay=profiled_delay,
        ground_knowledge=gk,
        simulation_start_utc=simulation_start_utc,
        epoch_interval_s=config.epoch_interval_s,
    )
    ctx = RunContext(
        world=world,
        endpoints=endpoints,
        ground_knowledge=gk,
        service_ground_delay=profiled_delay,
        simulation_start_utc=simulation_start_utc,
    )

    epoch_data = []
    for epoch in range(config.num_epochs):
        tables = ctrl.compute_tables(snapshots[epoch])
        demand = traffic.generate(epoch)
        result = realize(
            tables, snapshots[epoch], demand, ctx,
            epoch_interval_s=config.epoch_interval_s,
            simulation_start_utc=simulation_start_utc,
        )

        flows = result.flow_outcomes
        n = max(len(flows), 1)

        # Per-service-class stats
        by_service: dict[str, list[float]] = {}
        for f in flows:
            by_service.setdefault(f.flow_key.dst, []).append(f.total_rtt)

        service_stats = {
            svc: {"avg_rtt": round(sum(rtts) / len(rtts), 2), "n_flows": len(rtts)}
            for svc, rtts in by_service.items()
        }

        epoch_data.append({
            "epoch": epoch,
            "avg_rtt": round(sum(f.total_rtt for f in flows) / n, 2),
            "avg_sat": round(sum(f.satellite_rtt for f in flows) / n, 2),
            "avg_gnd": round(sum(f.ground_rtt for f in flows) / n, 2),
            "n_flows": len(flows),
            "pops_used": len({f.pop_code for f in flows}),
            "service_stats": service_stats,
        })

        print(
            f"  epoch {epoch:2d}  "
            f"rtt={epoch_data[-1]['avg_rtt']:.1f}ms  "
            f"sat={epoch_data[-1]['avg_sat']:.1f}ms  "
            f"gnd={epoch_data[-1]['avg_gnd']:.1f}ms  "
            f"flows={len(flows)}"
        )

    return {"epochs": epoch_data}


def main() -> None:
    (
        world, ground, endpoints, population, traffic, config, snapshots,
        profiled_delay, simulation_start_utc,
    ) = setup()

    n_terminals = len(population.sources)
    n_services = len(population.service_classes)
    print(
        f"\nRadar experiment: {n_terminals} terminals × {n_services} services, "
        f"{NUM_EPOCHS} epochs ({EPOCH_INTERVAL}s interval)"
    )
    print(f"Simulation start: {simulation_start_utc}")
    print(f"Service classes: {', '.join(SERVICE_CLASSES)}\n")

    print("Running service_aware (service-class)...")
    nearest = run_nearest_pop_service(
        world, ground, endpoints, traffic, config, snapshots,
        profiled_delay, simulation_start_utc,
    )

    # Summary
    epochs = nearest["epochs"]
    avg_total = sum(e["avg_rtt"] for e in epochs) / len(epochs)
    avg_sat = sum(e["avg_sat"] for e in epochs) / len(epochs)
    avg_gnd = sum(e["avg_gnd"] for e in epochs) / len(epochs)
    print(f"\nOverall: total={avg_total:.1f}ms  sat={avg_sat:.1f}ms  ground={avg_gnd:.1f}ms")

    # Per-service-class aggregation
    all_service_rtts: dict[str, list[float]] = {}
    for e in epochs:
        for svc, stats in e["service_stats"].items():
            all_service_rtts.setdefault(svc, []).append(stats["avg_rtt"])

    print("\nPer-service-class average RTT:")
    for svc in sorted(all_service_rtts):
        rtts = all_service_rtts[svc]
        print(f"  {svc:25s}  {sum(rtts) / len(rtts):.1f}ms")

    # Export results
    out_path = DASHBOARD_DIR / "service_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "num_epochs": NUM_EPOCHS,
                "epoch_interval_s": EPOCH_INTERVAL,
                "n_terminals": n_terminals,
                "n_services": n_services,
                "service_classes": list(SERVICE_CLASSES),
            },
            "service_aware": nearest,
        }, f)
    print(f"\nExported → {out_path}")


if __name__ == "__main__":
    main()
