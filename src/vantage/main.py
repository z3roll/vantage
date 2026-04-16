"""Argus experiment runner — RoutingPlane baseline with dashboard export.

Usage:
    uv run python -m vantage.main                    # 60 epochs, 10 Gbps
    uv run python -m vantage.main --epochs=300       # 5 minutes
    uv run python -m vantage.main --load=50          # 50 Gbps target load

Data pipeline:
    terminals  ← starlink_users.json × world_cities.json (Gravity distribution)
    destinations ← service_prefixes.json (12 services, PeeringDB locations)
    traffic    ← FlowLevelGenerator (Bounded Pareto + Poisson + diurnal)
    ground RTT ← GeographicGroundDelay (haversine to nearest service node)
    sat path   ← RoutingPlane + FIB walk + M/M/1/K queuing
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import CellGrid, Endpoint
from vantage.engine import RunConfig, RunContext, run_routing
from vantage.traffic import EndpointPopulation, FlowLevelGenerator
from vantage.world.ground import (
    GeographicGroundDelay,
    GroundInfrastructure,
    GroundKnowledge,
)
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

DATA_DIR = Path(__file__).resolve().parent / "config"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

_DEFAULT_STARPERF_XML = Path(
    "/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml"
)
STARPERF_XML = Path(
    os.environ.get("ARGUS_STARLINK_XML", str(_DEFAULT_STARPERF_XML))
)

LAND_GEOJSON = PROJECT_ROOT / "dashboard" / "ne_countries.geojson"
CELL_CACHE = PROJECT_ROOT / "data" / "processed" / "land_cells_res5.json"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_world() -> tuple[WorldModel, GroundInfrastructure]:
    """Build the physical world model (satellite + ground)."""
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
    return WorldModel(satellite, ground), ground


def _precompute_snapshots(world: WorldModel, num_epochs: int, interval: float):
    """Pre-compute all network snapshots in parallel."""
    print("Precomputing snapshots...", end=" ", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as pool:
        snapshots = list(pool.map(
            lambda e: world.snapshot_at(e, e * interval),
            range(num_epochs),
        ))
    print(f"{time.perf_counter() - t0:.1f}s")
    return snapshots


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(data) - 1)
    return data[f] * (c - k) + data[c] * (k - f)


def _parse_cli_float(flag: str, default: float) -> float:
    for arg in sys.argv:
        if arg.startswith(f"--{flag}="):
            return float(arg.split("=")[1])
    return default


def _parse_cli_int(flag: str, default: int) -> int:
    for arg in sys.argv:
        if arg.startswith(f"--{flag}="):
            return int(arg.split("=")[1])
    return default


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run RoutingPlane baseline experiment with dashboard export."""
    num_epochs = _parse_cli_int("epochs", 60)
    epoch_interval = 1.0

    world, ground = _build_world()

    # ── Destinations from service_prefixes.json ──
    with open(DATA_DIR / "service_prefixes.json") as f:
        service_data = json.load(f)

    service_destinations = []
    dst_weights: dict[str, float] = {}
    dst_locations: dict[str, list[dict]] = {}
    for svc_name, svc_info in service_data.items():
        locs = svc_info.get("locations", [])
        if not locs:
            continue
        service_destinations.append(
            Endpoint(svc_name, locs[0]["lat"], locs[0]["lon"])
        )
        dst_weights[svc_name] = svc_info.get("traffic_weight", 1.0)
        dst_locations[svc_name] = locs

    # ── Cities: one endpoint per city, user counts scaled from Starlink data ──
    user_scale = _parse_cli_float("scale", 1.0)  # 0.1 = 10% users, 2.0 = 2x users
    population = EndpointPopulation.from_starlink_users(
        DATA_DIR / "starlink_users.json",
        DATA_DIR / "world_cities.json",
        destinations=tuple(service_destinations),
        user_scale=user_scale,
    )
    total_users = sum(cg.user_count for cg in population.city_groups)
    print(f"  cities: {len(population.sources)} ({total_users:,} users, scale={user_scale}x)")
    print(f"  destinations: {len(service_destinations)} services")

    endpoints: dict = {s.name: s for s in population.sources}
    endpoints.update({d.name: d for d in population.destinations})

    # ── CellGrid ──
    if not LAND_GEOJSON.exists():
        raise FileNotFoundError(f"Land mask not found at {LAND_GEOJSON}")
    cell_grid = CellGrid.from_polygon_coverage(
        LAND_GEOJSON,
        endpoints=[(e.name, e.lat_deg, e.lon_deg) for e in endpoints.values()],
        cache_path=CELL_CACHE,
    )

    # ── Traffic (per-city Poisson, user-type mixture, timezone-aware) ──
    traffic = FlowLevelGenerator(
        population,
        config_dir=DATA_DIR,
        epoch_interval_s=epoch_interval,
        dst_weights=dst_weights,
        dst_locations=dst_locations,
    )

    config = RunConfig(num_epochs=num_epochs, epoch_interval_s=epoch_interval)
    snapshots = _precompute_snapshots(world, num_epochs, epoch_interval)

    # ── GeographicGroundDelay ──
    pop_coords = {p.code: (p.lat_deg, p.lon_deg) for p in ground.pops}
    geo_delay = GeographicGroundDelay(
        pop_coords=pop_coords,
        service_locations=dst_locations,
    )

    n_dests = len(service_destinations)
    print(f"\n{len(ground.pops)} PoPs, {n_dests} service dests, {len(cell_grid)} cells")
    print(f"  {num_epochs} epochs x {epoch_interval}s = {num_epochs * epoch_interval:.0f}s simulation\n")

    gk = GroundKnowledge(estimator=geo_delay)
    ctx = RunContext(world=world, endpoints=endpoints, ground_knowledge=gk)
    ctrl = NearestPoPController()

    print("Running RoutingPlane baseline (nearest_pop)...")
    result = run_routing(
        context=ctx, cell_grid=cell_grid, traffic=traffic,
        controller=ctrl, config=config,
    )

    # ── Collect per-epoch metrics ──
    epoch_data = []
    for i, (er, cap) in enumerate(zip(result.epochs, result.capacity, strict=True)):
        flows = er.flow_outcomes
        n = max(len(flows), 1)

        total_rtts = [f.total_rtt for f in flows]
        sat_rtts = [f.satellite_rtt for f in flows]
        gnd_rtts = [f.ground_rtt for f in flows]
        prop_rtts = [f.propagation_rtt for f in flows]
        q_rtts = [f.queuing_rtt for f in flows]
        tx_rtts = [f.transmission_rtt for f in flows]
        losses = [f.loss_probability for f in flows]
        eff_tputs = [f.effective_throughput_gbps for f in flows]
        bnecks = [f.bottleneck_gbps for f in flows]
        demands = [f.demand_gbps for f in flows]

        total_demand = sum(demands)
        total_eff = sum(eff_tputs)
        jains = 0.0
        if eff_tputs:
            s = sum(eff_tputs)
            sq = sum(x * x for x in eff_tputs)
            jains = (s * s) / (n * sq) if sq > 0 else 1.0

        epoch_data.append({
            "epoch": i,
            "time_s": i * epoch_interval,
            "flow_count": len(flows),
            "routed_gbps": er.routed_demand_gbps,
            "unrouted_gbps": er.unrouted_demand_gbps,
            "latency": {
                "mean_total": round(sum(total_rtts) / n, 3),
                "mean_satellite": round(sum(sat_rtts) / n, 3),
                "mean_ground": round(sum(gnd_rtts) / n, 3),
                "mean_propagation": round(sum(prop_rtts) / n, 3),
                "mean_queuing": round(sum(q_rtts) / n, 3),
                "mean_transmission": round(sum(tx_rtts) / n, 3),
                "p50_total": round(_percentile(total_rtts, 50), 3),
                "p95_total": round(_percentile(total_rtts, 95), 3),
                "p99_total": round(_percentile(total_rtts, 99), 3),
                "min_total": round(min(total_rtts) if total_rtts else 0, 3),
                "max_total": round(max(total_rtts) if total_rtts else 0, 3),
            },
            "throughput": {
                "total_demand": round(total_demand, 4),
                "total_effective": round(total_eff, 4),
                "mean_bottleneck": round(sum(bnecks) / n, 2),
                "utilization": round(total_eff / total_demand, 4) if total_demand > 0 else 0,
            },
            "loss": {
                "mean": float(f"{sum(losses) / n:.6e}"),
                "max": float(f"{max(losses) if losses else 0:.6e}"),
                "p95": float(f"{_percentile(losses, 95):.6e}"),
                "flows_with_loss": sum(1 for p in losses if p > 1e-9),
            },
            "capacity": {
                "max_isl_util": round(cap.max_isl_utilization, 4),
                "max_sat_feeder_util": round(cap.max_sat_feeder_utilization, 4),
                "max_gs_feeder_util": round(cap.max_gs_feeder_utilization, 4),
                "saturated_isl": cap.saturated_isl_count,
                "saturated_sat_feeder": cap.saturated_sat_feeder_count,
                "saturated_gs_feeder": cap.saturated_gs_feeder_count,
                "top_sat_feeders": [
                    {"sat_id": s, "util": round(u, 2), "load_gbps": round(l, 2), "cap_gbps": round(c, 2)}
                    for s, u, l, c in cap.top_sat_feeders
                ],
                "top_gs_feeders": [
                    {"gs_id": g, "util": round(u, 2), "load_gbps": round(l, 2), "cap_gbps": round(c, 2)}
                    for g, u, l, c in cap.top_gs_feeders
                ],
            },
            "fairness": {"jains_index": round(jains, 6)},
        })

        avg_rtt = epoch_data[-1]["latency"]["mean_total"]
        avg_q = epoch_data[-1]["latency"]["mean_queuing"]
        utc_h = int(i * epoch_interval / 3600) % 24
        utc_m = int(i * epoch_interval / 60) % 60
        utc_s = int(i * epoch_interval) % 60
        top_sf = cap.top_sat_feeders[0] if cap.top_sat_feeders else (0, 0, 0, 0)
        print(
            f"  {utc_h:02d}:{utc_m:02d}:{utc_s:02d}  epoch {i:3d}  rtt={avg_rtt:6.1f}ms  "
            f"flows={len(flows):4d}  "
            f"sat_feed={cap.max_sat_feeder_utilization:5.1f}x (sat {top_sf[0]} {top_sf[2]:.1f}/{top_sf[3]:.0f}Gbps)"
        )

    # ── Export ──
    export = {
        "config": {
            "num_epochs": num_epochs,
            "epoch_interval_s": epoch_interval,
            "n_pops": len(ground.pops),
            "n_dests": n_dests,
            "n_cities": len(population.sources),
            "n_users": total_users,
            "n_cells": len(cell_grid),
        },
        "summary": {
            "wall_time_s": round(result.wall_time_s, 2),
            "avg_total_rtt": round(result.avg_total_rtt, 3),
        },
        "epochs": epoch_data,
    }

    out_path = DASHBOARD_DIR / "baseline_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(export, f, indent=1)

    print(f"\nWall: {result.wall_time_s:.1f}s  avg_rtt={result.avg_total_rtt:.1f}ms")
    print(f"Exported -> {out_path}")
    print(f"Open dashboard: open {DASHBOARD_DIR / 'baseline.html'}")


if __name__ == "__main__":
    main()
