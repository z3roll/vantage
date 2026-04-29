"""Vantage end-to-end simulation with auto-launched live dashboard.

Runs the Baseline (nearest-PoP) and Progressive (cascade) controllers
in lockstep against bit-identical traffic, streams per-epoch results
to ``dashboard/sim_data_<ts>.json``, and serves the interactive
dashboard (``dashboard/index.html``) on a local HTTP server.

    $ uv run python run.py              # 60-epoch demo, opens browser
    $ uv run python run.py --epochs 600 # longer run
    $ uv run python run.py --no-browser # headless

The dashboard polls its JSON source every 3 s, so progress is visible
live while the simulation is still running.
"""
from __future__ import annotations

import argparse
import gc
import json
import socket
import subprocess
import sys
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from vantage.common.seed import derive_subseed, fresh_run_seed
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    compute_cell_sat_cost,
    compute_pop_capacity,
    rank_pops_by_e2e,
)
from vantage.control.policy.greedy import ProgressiveController
from vantage.control.policy.lpround import (
    LPRoundingController,
    _build_items,
    _weighted_cost,
)
from vantage.control.policy.milp import MILPController
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import CapacityView, CellGrid, Endpoint, UsageBook
from vantage.engine.context import RunContext
from vantage.engine.feedback import GroundDelayFeedback
from vantage.forward import RoutingPlaneForward, realize
from vantage.traffic import EndpointPopulation, FlowLevelGenerator
from vantage.world.ground import (
    GeographicGroundDelay,
    GroundInfrastructure,
    GroundKnowledge,
    GroundTruth,
)
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

# ── Paths (all relative to this script so cwd doesn't matter) ───────────────
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "src" / "vantage" / "config"
XML = DATA_DIR / "Starlink.xml"
LAND_GEOJSON = DATA_DIR / "ne_countries.geojson"
CELL_CACHE = REPO_ROOT / "data" / "processed" / "land_cells_res5.json"
DASHBOARD_DIR = REPO_ROOT / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
CELL_CACHE.parent.mkdir(parents=True, exist_ok=True)

# ── Simulation knobs ────────────────────────────────────────────────────────
EPOCH_S = 1.0
REFRESH = 15
N_ANTENNAS_PER_GS = 8
SAT_FEEDER_CAP_GBPS = 20.0


def pct(data, p):
    if not data:
        return 0.0
    d = sorted(data); k = (len(d) - 1) * p / 100; f = int(k); c = min(f + 1, len(d) - 1)
    return d[f] * (c - k) + d[c] * (k - f)


def w_pct(pairs, p):
    """Demand-weighted percentile.

    ``pairs`` = ``[(value, weight), ...]``. Sorts by value ascending
    and returns the value at which cumulative weight first reaches
    ``p``% of total weight. Semantic: "the RTT above which the top
    ``100 - p``% of traffic (measured in Gbps demand) sits".
    """
    if not pairs:
        return 0.0
    s = sorted(pairs, key=lambda t: t[0])
    total = sum(w for _, w in s)
    if total <= 0:
        return 0.0
    threshold = total * p / 100.0
    cum = 0.0
    for v, w in s:
        cum += w
        if cum >= threshold:
            return v
    return s[-1][0]


def w_mean(values, weights):
    """Demand-weighted arithmetic mean: ``Σ(v·w) / Σw``."""
    if not values or not weights:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def country_of(src_name: str) -> str:
    parts = src_name.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "city" else "??"


def _round_ms(d) -> dict:
    """Round a ``{step → ms}`` mapping for JSON output (2 decimals)."""
    return {k: round(float(v), 2) for k, v in d.items()}


def port_in_use(port: int) -> bool:
    """Whether some process is already listening on localhost:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_dashboard_server(port: int, directory: Path) -> subprocess.Popen:
    """Launch a detached http.server process rooted at ``directory``."""
    return subprocess.Popen(
        [
            sys.executable, "-m", "http.server", str(port),
            "--bind", "127.0.0.1",
            "--directory", str(directory),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=60,
                   help="Number of 1-second epochs to simulate (default: 60)")
    p.add_argument("--user-scale", type=float, default=5.0,
                   help="Starlink user-count multiplier (default: 5.0)")
    p.add_argument("--port", type=int, default=8000,
                   help="HTTP port for the dashboard (default: 8000)")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the browser (dashboard server still runs)")
    p.add_argument("--no-serve", action="store_true",
                   help="Don't start the dashboard HTTP server at all (benchmark mode)")
    p.add_argument("--max-gs-per-pop", type=int, default=0,
                   help="Experimental: cap each PoP at N attached GSs "
                        "(0 = no cap). Keeps the N closest by backhaul "
                        "delay so popular PoPs hit capacity earlier, "
                        "exposing PG / DP differences under pressure.")
    p.add_argument("--seed", type=int, default=None,
                   help="Run-level seed controlling every stochastic subsystem "
                        "(traffic AR(1)/Poisson, ground-delay LogNormal sampling, "
                        "ingress-sat selection). If omitted, a fresh random seed "
                        "is drawn for this run and printed. Use the same seed to "
                        "reproduce a run exactly.")
    return p.parse_args()


def collect_refresh_gc() -> None:
    """Run a full cyclic-GC pass at routing-plane refresh boundaries.

    ``realize()`` allocates heavily enough that letting CPython trigger
    gen2 collections opportunistically inside BL/PG forward passes shows
    up as large wall-clock spikes. We keep automatic cyclic GC disabled
    during the hot loop and explicitly pay that stop-the-world cost only
    when the routing planes refresh.
    """
    gc.collect(2)


def main() -> None:
    args = parse_args()

    num_epochs: int = args.epochs
    user_scale: float = args.user_scale

    # ── Run-level seed → per-subsystem sub-seeds ────────────────────
    # Baseline and PG share exactly these seeds each run, so any
    # observed delta reflects controller behavior, not a divergent
    # random draw between them. Each sub-seed is deterministic in
    # (run_seed, tag) so traffic noise, ground-delay jitter, and
    # ingress-sat selection can each vary independently across runs
    # without coupling through a shared RNG stream.
    run_seed: int = args.seed if args.seed is not None else fresh_run_seed()
    traffic_seed = derive_subseed(run_seed, "traffic")
    ground_seed = derive_subseed(run_seed, "ground_delay")
    ingress_seed_base = derive_subseed(run_seed, "ingress")
    print(f"Run seed: {run_seed}"
          + (" (from --seed)" if args.seed is not None else " (auto-generated)"))

    # ── Dashboard: server (optional) and auto-open browser (optional)
    # are decoupled so headless agents can keep the server running
    # while the human opens the URL manually.
    url = f"http://localhost:{args.port}/"
    # Browser auto-open is deferred to after the world-build summary
    # below — opening the tab before the build finishes made it spin
    # on an empty index for ~30 s. ``run.py`` is only responsible for
    # checking whether the dashboard port is already occupied and, if
    # not, starting a detached HTTP server process. The sim run never
    # owns the server lifetime, so Ctrl-C stops only the simulation.
    should_open_browser = False
    if not args.no_serve:
        if port_in_use(args.port):
            print(f"Port {args.port} already in use — another server is "
                  f"already bound to it. Reusing that server.")
            print(f"Dashboard: {url}")
            if not args.no_browser:
                should_open_browser = True
        else:
            start_dashboard_server(args.port, DASHBOARD_DIR)
            print(f"Dashboard: {url}")
            if not args.no_browser:
                should_open_browser = True

    start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = DASHBOARD_DIR / f"sim_data_{start_ts}.json"
    index_file = DASHBOARD_DIR / "index.json"

    # Rebuild index.json from whatever sim_data_*.json files actually
    # live in dashboard/ right now. This makes the dropdown reflect
    # the disk at the start of every run — files copied in from other
    # hosts, runs whose index entry was lost, or renames all show up
    # without needing a saved index to already list them. Subsequent
    # ``update_index`` calls during the run keep the current run's
    # progress counter fresh.
    def _index_entry_from_file(path: Path) -> dict | None:
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        cfg = data.get("config", {})
        bl = data.get("baseline") or []
        pg = data.get("progressive") or []
        epochs_done = max(len(bl), len(pg))
        ts = cfg.get("started_at") or path.stem.removeprefix("sim_data_")
        return {
            "filename": path.name,
            "timestamp": ts,
            "epochs_done": epochs_done,
            "epochs_total": cfg.get("num_epochs", epochs_done),
            "user_scale": cfg.get("user_scale", 1.0),
            "max_gs_per_pop": cfg.get("max_gs_per_pop"),
            "mtime": path.stat().st_mtime,
        }

    scanned = [
        _index_entry_from_file(p)
        for p in DASHBOARD_DIR.glob("sim_data_*.json")
    ]
    entries = [e for e in scanned if e is not None]
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    tmp = index_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"files": entries}, f)
    tmp.replace(index_file)
    print(f"Index: {len(entries)} sim_data files in {DASHBOARD_DIR.name}/")

    # ── Build ───────────────────────────────────────────────────────
    print("Building world...", flush=True)
    ground = GroundInfrastructure(DATA_DIR)
    if args.max_gs_per_pop > 0:
        # Prune each PoP's GS attachments to the N closest (by
        # backhaul_delay) to create capacity pressure. Pure experiment
        # knob — mutates the frozen dataset in memory only.
        from collections import defaultdict as _dd
        _by_pop: dict[str, list] = _dd(list)
        for e in ground._gs_pop_edges:
            _by_pop[e.pop_code].append(e)
        kept: list = []
        for pop_code, edges in _by_pop.items():
            edges.sort(key=lambda e: e.backhaul_delay)
            kept.extend(edges[: args.max_gs_per_pop])
        ground._gs_pop_edges = tuple(kept)
        ground._gs_to_pops = {}
        ground._pop_to_gs = {}
        for e in ground._gs_pop_edges:
            ground._gs_to_pops.setdefault(e.gs_id, []).append(e)
            ground._pop_to_gs.setdefault(e.pop_code, []).append(e)
        _kept_per_pop = {p: len(v) for p, v in ground._pop_to_gs.items()}
        print(f"Pruned GS attachments: max {args.max_gs_per_pop} GS/PoP → "
              f"{len(ground._gs_pop_edges)} edges; "
              f"PoPs w/ GSs: {len(_kept_per_pop)}/{len(ground.pops)}")
    satellite = SatelliteSegment(
        constellation=XMLConstellationModel(str(XML), dt_s=15.0),
        topology_builder=PlusGridTopology(), shell_id=1,
        ground_stations=ground.ground_stations, visibility=SphericalAccessModel(),
    )
    world = WorldModel(satellite, ground)

    with open(DATA_DIR / "service_prefixes.json") as f:
        svc_data = json.load(f)
    dst_locs = {k: v["locations"] for k, v in svc_data.items() if v.get("locations")}
    dst_weights = {k: v.get("traffic_weight", 1.0) for k, v in svc_data.items() if v.get("locations")}
    svc_names = sorted(dst_locs.keys())
    dests = [Endpoint(k, v[0]["lat"], v[0]["lon"]) for k, v in dst_locs.items()]

    population = EndpointPopulation.from_starlink_users(
        DATA_DIR / "starlink_users.json", DATA_DIR / "world_cities.json",
        destinations=tuple(dests), user_scale=user_scale,
    )
    endpoints = {s.name: s for s in population.sources}
    endpoints.update({d.name: d for d in population.destinations})
    cell_grid = CellGrid.from_polygon_coverage(
        LAND_GEOJSON, endpoints=[(e.name, e.lat_deg, e.lon_deg) for e in endpoints.values()],
        cache_path=CELL_CACHE,
    )
    # Deterministic distance-based prior (one-way RTT, no RNG). Used
    # only as a cold-start fallback by GroundKnowledge — the realized
    # truth flows through the separate GroundTruth source below.
    geo_delay = GeographicGroundDelay(
        pop_coords={p.code: (p.lat_deg, p.lon_deg) for p in ground.pops},
        service_locations=dst_locs,
    )
    # Epoch-varying truth. Centered on the prior, LogNormal jitter
    # seeded by (run_seed → ground_seed, epoch, pop, dest), so BL and
    # PG see identical truth for any given epoch within one run.
    ground_truth = GroundTruth(prior=geo_delay, seed_base=ground_seed)
    gs_by_id = {gs.gs_id: gs for gs in world.ground_stations}

    snap0 = world.snapshot_at(0, 0.0)
    pop_gs_list: dict[str, list[str]] = {}
    for _edge in snap0.infra.gs_pop_edges:
        pop_gs_list.setdefault(_edge.pop_code, []).append(_edge.gs_id)
    pop_list = [p.code for p in ground.pops]

    def dynamic_pop_capacity(_snap) -> dict[str, float]:
        """Per-PoP theoretical cap = (#GSs) × 8 antennas × 20 Gbps."""
        return {p: len(pop_gs_list.get(p, ())) * N_ANTENNAS_PER_GS * SAT_FEEDER_CAP_GBPS
                for p in pop_list}

    print(f"Users: {sum(c.user_count for c in population.city_groups):,}  "
          f"Services: {len(svc_names)}  PoPs: {len(pop_list)}")
    print(f"Epochs: {num_epochs} × {EPOCH_S}s = {num_epochs * EPOCH_S / 60:.1f} min")
    print(f"Output: {out_file}\n")

    # Deferred browser auto-open. ``should_open_browser`` is true only
    # when the dashboard server actually started and the user did not
    # pass ``--no-browser``; a port-busy fallback or a build failure
    # earlier in this function both leave it false, so no tab opens.
    if should_open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — browser-open is best-effort
            pass

    def compute_breakdown(result, top_contrib: int = 12, top_sats: int = 12) -> dict:
        pop_country_svc: dict[str, dict[tuple[str, str], float]] = defaultdict(
            lambda: defaultdict(float))
        pop_svc_total: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        pop_gs_sat: dict[str, dict[str, dict[int, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))
        for f in result.flow_outcomes:
            c = country_of(f.flow_key.src)
            pop_country_svc[f.pop_code][(c, f.flow_key.dst)] += f.demand_gbps
            pop_svc_total[f.pop_code][f.flow_key.dst] += f.demand_gbps
            pop_gs_sat[f.pop_code][f.gs_id][int(f.egress_sat)] += f.demand_gbps

        out: dict[str, dict] = {}
        for pop in pop_country_svc:
            ranked = sorted(pop_country_svc[pop].items(), key=lambda x: -x[1])
            gs_breakdown = []
            total_sats = 0
            for gs_id, sat_loads in pop_gs_sat[pop].items():
                gs_total = sum(sat_loads.values())
                sats_sorted = sorted(sat_loads.items(), key=lambda x: -x[1])
                gs_breakdown.append({
                    "gs_id": gs_id, "load": round(gs_total, 2),
                    "sats": [{"sat_id": s, "load": round(g, 2)}
                             for s, g in sats_sorted[:top_sats]],
                    "n_sats": len(sats_sorted),
                })
                total_sats += len(sats_sorted)
            gs_breakdown.sort(key=lambda g: -g["load"])
            out[pop] = {
                "total": round(sum(g for _, g in ranked), 2),
                "n_distinct": len(ranked),
                "top": [{"country": c, "svc": s, "gbps": round(g, 2)}
                        for (c, s), g in ranked[:top_contrib]],
                "svc_gbps": {s: round(g, 2) for s, g in pop_svc_total[pop].items()},
                "gss": gs_breakdown, "n_gss": len(gs_breakdown), "n_sats": total_sats,
            }
        return out

    def collect(epoch, result, book, pop_capacity):
        """Aggregate per-flow outcomes into dashboard-shaped stats.

        Mean / p95 / p99 / sat / gnd are **demand-weighted** (each
        flow contributes in proportion to its demand Gbps, not one
        sample per flow). This matches the Control-layer convention
        so the two sections' curves are directly comparable.
        """
        flows = result.flow_outcomes
        rtts = [f.total_rtt for f in flows]
        demands = [f.demand_gbps for f in flows]
        rtt_pairs = list(zip(rtts, demands, strict=True))
        svc = defaultdict(lambda: {"rtts": [], "sat": [], "gnd": [],
                                   "demands": [], "demand": 0.0})
        pop_load = defaultdict(float)
        for f in flows:
            svc[f.flow_key.dst]["rtts"].append(f.total_rtt)
            svc[f.flow_key.dst]["sat"].append(f.satellite_rtt)
            svc[f.flow_key.dst]["gnd"].append(f.ground_rtt)
            svc[f.flow_key.dst]["demands"].append(f.demand_gbps)
            svc[f.flow_key.dst]["demand"] += f.demand_gbps
            pop_load[f.pop_code] += f.demand_gbps

        gs_over = sum(1 for g in book.gs_feeder_used if book.gs_feeder_utilization(g) > 1.0)
        sf_over = sum(1 for s in book.sat_feeder_used if book.sat_feeder_utilization(s) > 1.0)
        # Per-epoch ISL congestion. Iterate only the links that carried
        # any traffic this epoch (``isl_used`` keys are the canonical
        # ``(min, max)`` form used by :class:`UsageBook``). If no ISL
        # was used, ``max_isl_util`` is 0.0 by convention.
        isl_utils = [book.isl_utilization(a, b) for (a, b) in book.isl_used]
        isl_over = sum(1 for u in isl_utils if u > 1.0)
        max_isl_util = max(isl_utils) if isl_utils else 0.0

        svc_out = {}
        for s in svc_names:
            d = svc[s]
            if d["rtts"]:
                pairs = list(zip(d["rtts"], d["demands"], strict=True))
                svc_out[s] = {
                    "mean": round(w_mean(d["rtts"], d["demands"]), 2),
                    "p95": round(w_pct(pairs, 95), 2),
                    "p99": round(w_pct(pairs, 99), 2),
                    "sat": round(w_mean(d["sat"], d["demands"]), 2),
                    "gnd": round(w_mean(d["gnd"], d["demands"]), 2),
                    "n": len(d["rtts"]),
                    "demand_gbps": round(d["demand"], 2),
                }

        all_pops_out = []
        for p in pop_list:
            load = pop_load.get(p, 0.0)
            cap = pop_capacity.get(p, 0)
            all_pops_out.append({
                "code": p, "load": round(load, 1), "cap": cap,
                "util": round(load / cap * 100, 1) if cap > 0 else 0,
                "n_gss_total": len(pop_gs_list.get(p, ())),
            })
        all_pops_out.sort(key=lambda x: -x["load"])

        hh, mm, ss = int(epoch * EPOCH_S) // 3600, int(epoch * EPOCH_S) % 3600 // 60, int(epoch * EPOCH_S) % 60
        sat_rtts = [f.satellite_rtt for f in flows]
        gnd_rtts = [f.ground_rtt for f in flows]
        return {
            "epoch": epoch, "time_s": epoch * EPOCH_S,
            "time_str": f"{hh:02d}:{mm:02d}:{ss:02d}",
            "n_flows": len(flows),
            "demand_gbps": round(sum(demands), 1),
            # Demand-weighted mean/p95/p99 over total RTT.
            "mean_rtt": round(w_mean(rtts, demands), 2) if rtts else 0,
            "p50_rtt": round(w_pct(rtt_pairs, 50), 2),
            "p95_rtt": round(w_pct(rtt_pairs, 95), 2),
            "p99_rtt": round(w_pct(rtt_pairs, 99), 2),
            # Demand-weighted per-segment means.
            "mean_sat": round(w_mean(sat_rtts, demands), 2) if flows else 0,
            "mean_gnd": round(w_mean(gnd_rtts, demands), 2) if flows else 0,
            "mean_prop": round(w_mean([f.propagation_rtt for f in flows], demands), 2) if flows else 0,
            "mean_queue": round(w_mean([f.queuing_rtt for f in flows], demands), 2) if flows else 0,
            "mean_tx": round(w_mean([f.transmission_rtt for f in flows], demands), 2) if flows else 0,
            "gs_overloaded": gs_over, "sf_overloaded": sf_over,
            "isl_overloaded": isl_over, "max_isl_util": round(max_isl_util, 4),
            "services": svc_out, "pops": all_pops_out,
        }

    def compute_pop_compare(result_bl, result_pg) -> dict:
        bl_rtt_by_fk = {f.flow_key: f.total_rtt for f in result_bl.flow_outcomes}
        bl_demand_by_pop: dict[str, float] = defaultdict(float)
        for f in result_bl.flow_outcomes:
            bl_demand_by_pop[f.pop_code] += f.demand_gbps
        bl_total = sum(bl_demand_by_pop.values()) or 1.0

        pg_demand_by_pop: dict[str, float] = defaultdict(float)
        pg_flows_by_pop: dict[str, list] = defaultdict(list)
        for f in result_pg.flow_outcomes:
            pg_demand_by_pop[f.pop_code] += f.demand_gbps
            pg_flows_by_pop[f.pop_code].append(f)
        pg_total = sum(pg_demand_by_pop.values()) or 1.0

        out: dict[str, dict] = {}
        for pop in set(bl_demand_by_pop) | set(pg_demand_by_pop):
            sum_w = sum_pg = sum_bl = 0.0
            for fl in pg_flows_by_pop.get(pop, ()):
                bl_rtt = bl_rtt_by_fk.get(fl.flow_key)
                if bl_rtt is None:
                    continue
                sum_w += fl.demand_gbps
                sum_pg += fl.total_rtt * fl.demand_gbps
                sum_bl += bl_rtt * fl.demand_gbps
            if sum_w > 0:
                pg_e2e = sum_pg / sum_w; bl_e2e = sum_bl / sum_w
                delta_pct = (pg_e2e - bl_e2e) / bl_e2e * 100 if bl_e2e else 0.0
            else:
                pg_e2e = bl_e2e = delta_pct = None
            out[pop] = {
                "bl_pct": round(bl_demand_by_pop.get(pop, 0.0) / bl_total * 100, 2),
                "pg_pct": round(pg_demand_by_pop.get(pop, 0.0) / pg_total * 100, 2),
                "bl_e2e": round(bl_e2e, 2) if bl_e2e is not None else None,
                "pg_e2e": round(pg_e2e, 2) if pg_e2e is not None else None,
                "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
            }
        return out

    def compute_epoch_compare(result_bl, result_pg, result_lp, result_mip) -> dict:
        """Per-epoch cross-controller metrics. All mean / sat / gnd /
        p95 / p99 are **demand-weighted** (per-flow contribution
        proportional to demand Gbps)."""
        bl_by = {f.flow_key: f for f in result_bl.flow_outcomes}
        pg_by = {f.flow_key: f for f in result_pg.flow_outcomes}
        lp_by = {f.flow_key: f for f in result_lp.flow_outcomes}
        mip_by = {f.flow_key: f for f in result_mip.flow_outcomes}
        common = set(bl_by) & set(pg_by) & set(lp_by) & set(mip_by)
        TOL = 0.5  # noqa: N806 — small local tolerance constant.

        total_n = 0
        total_demand = 0.0
        rr = {"pg": [0, 0, 0, 0], "lp": [0, 0, 0, 0], "mip": [0, 0, 0, 0]}
        # rr[key] = [reroute, impr, worse, neutral]
        sums = {
            "bl": [0.0, 0.0], "pg": [0.0, 0.0],
            "lp": [0.0, 0.0], "mip": [0.0, 0.0],
        }  # [sat·d, gnd·d]
        rtt_pairs = {"bl": [], "pg": [], "lp": [], "mip": []}  # (rtt, d) pairs
        rtt_wsum = {"bl": 0.0, "pg": 0.0, "lp": 0.0, "mip": 0.0}

        series_by = {"bl": bl_by, "pg": pg_by, "lp": lp_by, "mip": mip_by}
        for fk in common:
            fb = bl_by[fk]
            d = fb.demand_gbps
            total_n += 1
            total_demand += d
            for key, by in series_by.items():
                f = by[fk]
                rtt_pairs[key].append((f.total_rtt, d))
                rtt_wsum[key] += f.total_rtt * d
                sums[key][0] += f.satellite_rtt * d
                sums[key][1] += f.ground_rtt * d
            for key in ("pg", "lp", "mip"):
                fx = series_by[key][fk]
                if fb.pop_code != fx.pop_code:
                    rr[key][0] += 1
                    d_total = fx.total_rtt - fb.total_rtt
                    if d_total < -TOL: rr[key][1] += 1
                    elif d_total > TOL: rr[key][2] += 1
                    else: rr[key][3] += 1

        def _wavg(s):
            return s / total_demand if total_demand > 0 else 0.0

        out = {
            "total_flows": total_n,
            "total_demand_gbps": round(total_demand, 2),
            "reroute_flows": rr["pg"][0],
            "reroute_pct": round(rr["pg"][0] / total_n * 100, 2) if total_n else 0,
            "impr_flows": rr["pg"][1], "worse_flows": rr["pg"][2], "neutral_flows": rr["pg"][3],
            "impr_pct_of_re": round(rr["pg"][1] / rr["pg"][0] * 100, 1) if rr["pg"][0] else 0,
            "worse_pct_of_re": round(rr["pg"][2] / rr["pg"][0] * 100, 1) if rr["pg"][0] else 0,
        }
        for key, prefix in (("bl", "bl"), ("pg", "pg"), ("lp", "lp"), ("mip", "mip")):
            out[f"{prefix}_mean_rtt"] = round(_wavg(rtt_wsum[key]), 2)
            out[f"{prefix}_sat_rtt"] = round(_wavg(sums[key][0]), 2)
            out[f"{prefix}_gnd_rtt"] = round(_wavg(sums[key][1]), 2)
            out[f"{prefix}_p95_rtt"] = round(w_pct(rtt_pairs[key], 95), 2)
            out[f"{prefix}_p99_rtt"] = round(w_pct(rtt_pairs[key], 99), 2)
        for key, prefix in (("lp", "lp"), ("mip", "mip")):
            re_n, im, wo, ne = rr[key]
            out[f"{prefix}_reroute_flows"] = re_n
            out[f"{prefix}_reroute_pct"] = round(re_n / total_n * 100, 2) if total_n else 0
            out[f"{prefix}_impr_flows"] = im
            out[f"{prefix}_worse_flows"] = wo
            out[f"{prefix}_neutral_flows"] = ne
            out[f"{prefix}_impr_pct_of_re"] = round(im / re_n * 100, 1) if re_n else 0
            out[f"{prefix}_worse_pct_of_re"] = round(wo / re_n * 100, 1) if re_n else 0
        return out

    def extract_cache_state(gk) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for (pop, dest), rtt in gk.all_entries().items():
            if ":" in dest:
                continue
            out.setdefault(pop, {})[dest] = round(rtt, 2)
        return out

    def update_index(epochs_done: int) -> None:
        by_name: dict[str, dict] = {}
        if index_file.exists():
            try:
                with open(index_file) as f:
                    for e in json.load(f).get("files", []):
                        by_name[e["filename"]] = e
            except (OSError, json.JSONDecodeError):
                pass
        by_name[out_file.name] = {
            "filename": out_file.name, "timestamp": start_ts,
            "epochs_done": epochs_done, "epochs_total": num_epochs,
            "user_scale": user_scale,
            "max_gs_per_pop": args.max_gs_per_pop,
            "mtime": time.time(),
        }
        existing = {p.name for p in DASHBOARD_DIR.glob("sim_data_*.json")}
        entries = [e for e in by_name.values() if e["filename"] in existing]
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        tmp = index_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"files": entries}, f)
        tmp.replace(index_file)

    def save_data(bl_data, pg_data, lp_data, mip_data,
                  bl_brk, pg_brk, lp_brk, mip_brk,
                  pop_cmp, ep_cmp, cache_state):
        out = {
            "config": {
                "num_epochs": num_epochs, "epoch_s": EPOCH_S, "refresh_s": REFRESH,
                "user_scale": user_scale,
                "max_gs_per_pop": args.max_gs_per_pop,
                "services": svc_names,
                "total_capacity_gbps": 8 * SAT_FEEDER_CAP_GBPS * len(pop_list),
                "started_at": start_ts,
                "run_seed": run_seed,
                "seed_source": "cli" if args.seed is not None else "auto",
                "sub_seeds": {
                    "traffic": traffic_seed,
                    "ground_delay": ground_seed,
                    "ingress": ingress_seed_base,
                },
                # Series labels for the dashboard. Four-way display:
                # BL (nearest PoP) / PG (greedy) / LP (LP-relaxation
                # + argmax rounding) / MILP (HiGHS integer optimum
                # with time budget).
                "series": ["baseline", "progressive", "lpround", "milp"],
            },
            "baseline": bl_data, "progressive": pg_data,
            "lpround": lp_data, "milp": mip_data,
            "latest_breakdown": {
                "baseline": bl_brk, "progressive": pg_brk,
                "lpround": lp_brk, "milp": mip_brk,
            },
            "latest_pop_compare": pop_cmp, "epoch_compare": ep_cmp,
            "cache_state": cache_state,
        }
        with open(out_file, "w") as f:
            json.dump(out, f)
        update_index(epochs_done=len(bl_data))

    # ── Setup both controllers ──────────────────────────────────────
    # GroundKnowledge is per-controller (so each controller observes
    # its own routed-flow feedback) but GroundTruth is shared — a
    # single epoch produces identical truth regardless of which
    # controller's data plane asked for it. Feedback is run on BOTH
    # contexts so each controller's GK learns from the same truth
    # samples its own forward pass produced.
    gk_bl = GroundKnowledge(estimator=geo_delay)
    ctx_bl = RunContext(
        world=world, endpoints=endpoints,
        ground_knowledge=gk_bl, ground_truth=ground_truth,
    )
    gk_pg = GroundKnowledge(estimator=geo_delay)
    ctx_pg = RunContext(
        world=world, endpoints=endpoints,
        ground_knowledge=gk_pg, ground_truth=ground_truth,
    )
    gk_lp = GroundKnowledge(estimator=geo_delay)
    ctx_lp = RunContext(
        world=world, endpoints=endpoints,
        ground_knowledge=gk_lp, ground_truth=ground_truth,
    )
    gk_mip = GroundKnowledge(estimator=geo_delay)
    ctx_mip = RunContext(
        world=world, endpoints=endpoints,
        ground_knowledge=gk_mip, ground_truth=ground_truth,
    )
    feedback_bl = GroundDelayFeedback(gk_bl)
    feedback_pg = GroundDelayFeedback(gk_pg)
    feedback_lp = GroundDelayFeedback(gk_lp)
    feedback_mip = GroundDelayFeedback(gk_mip)
    # Single shared traffic generator: one stochastic source per epoch
    # for both BL and PG. Using two independent generators — even with
    # the same literal seed — still lets their RNG states drift apart
    # (e.g. if one skipped a city). We generate the demand once and
    # hand the same immutable :class:`TrafficDemand` to both
    # controllers so BL and PG are compared on identical inputs.
    traffic = FlowLevelGenerator(population, config_dir=DATA_DIR, epoch_interval_s=EPOCH_S,
                                 dst_weights=dst_weights, dst_locations=dst_locs,
                                 seed=traffic_seed)
    # Controllers are held per-run (not re-constructed each refresh)
    # so run.py can read each one's ``last_timing`` after a plan
    # rebuild and export the per-step breakdown to the dashboard.
    bl_controller = NearestPoPController()
    pg_controller = ProgressiveController(
        ground_knowledge=gk_pg, dest_names=tuple(svc_names),
    )
    lp_controller = LPRoundingController(
        ground_knowledge=gk_lp, dest_names=tuple(svc_names),
    )
    mip_controller = MILPController(
        ground_knowledge=gk_mip, dest_names=tuple(svc_names),
    )

    bl_plane = pg_plane = lp_plane = mip_plane = None
    # Forward lifecycle (reuse vs. rebuild, cache invalidation) is
    # owned by RoutingPlaneForward.for_epoch — we just carry the
    # previous instances so the factory can decide.
    bl_forward: RoutingPlaneForward | None = None
    pg_forward: RoutingPlaneForward | None = None
    lp_forward: RoutingPlaneForward | None = None
    mip_forward: RoutingPlaneForward | None = None
    bl_data: list = []
    pg_data: list = []
    lp_data: list = []
    mip_data: list = []
    compare_data: list = []

    print(f"{'ep':>4} {'time':>8} {'flows':>6} {'dem':>6}  "
          f"{'BL':>6} {'PG':>6} {'LP':>6} {'MIP':>6}  "
          f"{'BL95':>6} {'PG95':>6} {'LP95':>6} {'MIP95':>6}  "
          f"{'wall':>5}")
    print("-" * 110)

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    t_wall_start = time.perf_counter()

    try:
        for epoch in range(num_epochs):
            t = epoch * EPOCH_S
            t_ep_start = time.perf_counter()

            snap = world.snapshot_at(epoch, t)
            gk_bl.set_clock(t); gk_pg.set_clock(t)
            gk_lp.set_clock(t); gk_mip.set_clock(t)
            refresh_epoch = bl_plane is None or epoch % REFRESH == 0

            # One stochastic draw of traffic per epoch; all controllers
            # realize against the exact same ``TrafficDemand``. Baseline
            # now consumes demand too so its cascade-walk can respect
            # PoP aggregate capacity (apples-to-apples with PG/LP/MIP).
            demand = traffic.generate(epoch)
            current_demand = {(fk.src, fk.dst): d for fk, d in demand.flows.items()}

            bl_plan_ms: float | None = None
            bl_plan_timing: dict = {}
            if refresh_epoch:
                t_bl_plan = time.perf_counter()
                bl_plane = bl_controller.compute_routing_plane(
                    snap, cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                bl_plan_ms = (time.perf_counter() - t_bl_plan) * 1000
                bl_plan_timing = dict(bl_controller.last_timing)

            pg_plan_ms: float | None = None
            pg_plan_timing: dict = {}
            if refresh_epoch:
                t_pg_plan = time.perf_counter()
                pg_plane = pg_controller.compute_routing_plane(
                    snapshot=snap, cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                pg_plan_ms = (time.perf_counter() - t_pg_plan) * 1000
                pg_plan_timing = dict(pg_controller.last_timing)

            lp_plan_ms: float | None = None
            lp_plan_timing: dict = {}
            if refresh_epoch:
                t_lp_plan = time.perf_counter()
                lp_plane = lp_controller.compute_routing_plane(
                    snapshot=snap, cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                lp_plan_ms = (time.perf_counter() - t_lp_plan) * 1000
                lp_plan_timing = dict(lp_controller.last_timing)

            mip_plan_ms: float | None = None
            mip_plan_timing: dict = {}
            mip_solve_meta: dict = {}
            if refresh_epoch:
                t_mip_plan = time.perf_counter()
                mip_plane = mip_controller.compute_routing_plane(
                    snapshot=snap, cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                mip_plan_ms = (time.perf_counter() - t_mip_plan) * 1000
                mip_plan_timing = dict(mip_controller.last_timing)
                mip_solve_meta = dict(mip_controller.last_solve_meta)

            # ── Score each refresh-plan on a common control-layer
            # objective Σ d·c(i,p) + 1e4·overflow. All four assignments
            # are evaluated under the SAME rankings + pop_cap (built
            # from PG's GK state, the incumbent reference), so the
            # numbers are apples-to-apples — this is the "theoretical
            # plan cost" you'd get *before* forward/realize perturbs
            # anything. ``plan_lb`` is the LP continuous-relaxation
            # optimum, a provable floor on any integer assignment.
            #
            # In addition to the scalar plan cost, we also compute
            # per-controller demand-weighted mean / sat / gnd / p95 /
            # p99 from the same (assignment, cell_sat_cost,
            # ground_cost) inputs. These are the planner's predicted
            # RTT stats *before* forward-layer queuing, sat-feeder
            # contention, or ISL congestion perturb them — the
            # "idealised" version of the dashboard's Forward curves.
            plan_costs: dict[str, float | None] = {}
            plan_stats: dict[str, dict[str, float] | None] = {}
            plan_lb: float | None = None
            if refresh_epoch:
                overflow_penalty = 1.0e4
                _ground_cost = pg_controller._make_ground_cost(
                    current_epoch=epoch,
                    pops=snap.infra.pops,
                    dest_names=svc_names,
                )
                _baseline = build_cell_to_pop_nearest(
                    cell_grid=cell_grid, pops=snap.infra.pops,
                    built_at=snap.time_s, version=epoch,
                )
                _csc = compute_cell_sat_cost(snap, cell_grid)
                _rankings = rank_pops_by_e2e(
                    cell_grid=cell_grid, pops=snap.infra.pops,
                    baseline=_baseline, cell_sat_cost=_csc,
                    ground_cost_fn=_ground_cost, dest_names=svc_names,
                )
                _pop_cap = compute_pop_capacity(snap)
                _items = _build_items(_rankings, cell_grid, current_demand)

                def _assignment_from_plane(plane, _r=_rankings):
                    per_dest = plane.cell_to_pop.per_dest
                    mapping = plane.cell_to_pop.mapping
                    out: dict[tuple[int, str], str] = {}
                    for key in _r:
                        override = per_dest.get(key)
                        if override:
                            out[key] = override[0]
                        else:
                            base = mapping.get(key[0])
                            if base:
                                out[key] = base[0]
                    return out

                def _plan_stats(assignment):
                    """Demand-weighted mean / sat / gnd / p95 / p99 on
                    the assignment's predicted RTT pool. Sat RTT from
                    ``cell_sat_cost``; ground RTT from the planner's
                    scored ``ground_cost_fn``. No forward coupling
                    modelled — this is the planner's idealised view.

                    Same demand-weighted convention as :func:`collect`
                    so Control and Forward curves are plotted on the
                    same axes.
                    """
                    rtts: list[float] = []
                    sats: list[float] = []
                    gnds: list[float] = []
                    demands: list[float] = []
                    for cell_id, dst, demand, _ in _items:
                        pop = assignment.get((cell_id, dst))
                        if pop is None:
                            continue
                        sat = _csc.get((cell_id, pop))
                        if sat is None:
                            continue
                        gnd = _ground_cost(pop, dst) or 0.0
                        rtts.append(sat + gnd)
                        sats.append(sat)
                        gnds.append(gnd)
                        demands.append(demand)
                    if not rtts:
                        return {"mean": 0.0, "sat": 0.0, "gnd": 0.0,
                                "p95": 0.0, "p99": 0.0}
                    pairs = list(zip(rtts, demands, strict=True))
                    return {
                        "mean": w_mean(rtts, demands),
                        "sat": w_mean(sats, demands),
                        "gnd": w_mean(gnds, demands),
                        "p95": w_pct(pairs, 95),
                        "p99": w_pct(pairs, 99),
                    }

                for label, plane in (
                    ("bl", bl_plane), ("pg", pg_plane),
                    ("lp", lp_plane), ("mip", mip_plane),
                ):
                    assign = _assignment_from_plane(plane)
                    plan_costs[label] = round(_weighted_cost(
                        assign, _items, _pop_cap,
                        overflow_penalty=overflow_penalty,
                    ), 2)
                    stats = _plan_stats(assign)
                    plan_stats[label] = {k: round(v, 3) for k, v in stats.items()}
                # LB comes directly from the LP solver (res.fun) —
                # LP's own GK may have drifted from PG's by this point,
                # but at refresh cadence the drift is tiny. For a
                # fully airtight LB under PG's GK we'd need another
                # LP solve here; we intentionally don't pay that cost.
                plan_lb = (
                    round(lp_controller.last_lp_opt, 2)
                    if lp_controller.last_lp_opt is not None else None
                )

            if refresh_epoch and gc_was_enabled:
                collect_refresh_gc()

            view_bl = CapacityView.from_snapshot(sat_state=snap.satellite, shell=world.shell, ground_stations=gs_by_id)
            book_bl = UsageBook(view=view_bl)
            bl_forward = RoutingPlaneForward.for_epoch(
                bl_forward, bl_plane, cell_grid, book_bl,
            )
            result_bl = realize(
                bl_forward, snap, demand, ctx_bl,
                ingress_seed_base=ingress_seed_base,
            )
            feedback_bl.observe(result_bl)

            view_pg = CapacityView.from_snapshot(sat_state=snap.satellite, shell=world.shell, ground_stations=gs_by_id)
            book_pg = UsageBook(view=view_pg)
            pg_forward = RoutingPlaneForward.for_epoch(
                pg_forward, pg_plane, cell_grid, book_pg,
            )
            result_pg = realize(
                pg_forward, snap, demand, ctx_pg,
                ingress_seed_base=ingress_seed_base,
            )
            feedback_pg.observe(result_pg)

            view_lp = CapacityView.from_snapshot(sat_state=snap.satellite, shell=world.shell, ground_stations=gs_by_id)
            book_lp = UsageBook(view=view_lp)
            lp_forward = RoutingPlaneForward.for_epoch(
                lp_forward, lp_plane, cell_grid, book_lp,
            )
            result_lp = realize(
                lp_forward, snap, demand, ctx_lp,
                ingress_seed_base=ingress_seed_base,
            )
            feedback_lp.observe(result_lp)

            view_mip = CapacityView.from_snapshot(sat_state=snap.satellite, shell=world.shell, ground_stations=gs_by_id)
            book_mip = UsageBook(view=view_mip)
            mip_forward = RoutingPlaneForward.for_epoch(
                mip_forward, mip_plane, cell_grid, book_mip,
            )
            result_mip = realize(
                mip_forward, snap, demand, ctx_mip,
                ingress_seed_base=ingress_seed_base,
            )
            feedback_mip.observe(result_mip)

            pop_capacity = dynamic_pop_capacity(snap)
            bl_data.append(collect(epoch, result_bl, book_bl, pop_capacity))
            pg_data.append(collect(epoch, result_pg, book_pg, pop_capacity))
            lp_data.append(collect(epoch, result_lp, book_lp, pop_capacity))
            mip_data.append(collect(epoch, result_mip, book_mip, pop_capacity))
            bl_brk = compute_breakdown(result_bl)
            pg_brk = compute_breakdown(result_pg)
            lp_brk = compute_breakdown(result_lp)
            mip_brk = compute_breakdown(result_mip)
            pop_cmp = compute_pop_compare(result_bl, result_pg)
            ep_cmp = compute_epoch_compare(result_bl, result_pg, result_lp, result_mip)
            ep_cmp["epoch"] = epoch
            ep_cmp["time_str"] = bl_data[-1]["time_str"]
            ep_cmp["bl_plan_ms"] = round(bl_plan_ms, 2) if bl_plan_ms is not None else None
            ep_cmp["pg_plan_ms"] = round(pg_plan_ms, 2) if pg_plan_ms is not None else None
            ep_cmp["lp_plan_ms"] = round(lp_plan_ms, 2) if lp_plan_ms is not None else None
            ep_cmp["mip_plan_ms"] = round(mip_plan_ms, 2) if mip_plan_ms is not None else None
            # Forward timing is always present (realize always runs).
            # Plan timing breakdowns are empty on cached-plan epochs.
            bl_fwd = dict(result_bl.forward_timing_ms)
            pg_fwd = dict(result_pg.forward_timing_ms)
            lp_fwd = dict(result_lp.forward_timing_ms)
            mip_fwd = dict(result_mip.forward_timing_ms)
            bl_forward_ms = bl_fwd.get("total_ms", 0.0)
            pg_forward_ms = pg_fwd.get("total_ms", 0.0)
            lp_forward_ms = lp_fwd.get("total_ms", 0.0)
            mip_forward_ms = mip_fwd.get("total_ms", 0.0)
            ep_cmp["bl_forward_ms"] = round(bl_forward_ms, 2)
            ep_cmp["pg_forward_ms"] = round(pg_forward_ms, 2)
            ep_cmp["lp_forward_ms"] = round(lp_forward_ms, 2)
            ep_cmp["mip_forward_ms"] = round(mip_forward_ms, 2)
            ep_cmp["bl_total_ms"] = round((bl_plan_ms or 0.0) + bl_forward_ms, 2)
            ep_cmp["pg_total_ms"] = round((pg_plan_ms or 0.0) + pg_forward_ms, 2)
            ep_cmp["lp_total_ms"] = round((lp_plan_ms or 0.0) + lp_forward_ms, 2)
            ep_cmp["mip_total_ms"] = round((mip_plan_ms or 0.0) + mip_forward_ms, 2)
            ep_cmp["forward_total_ms"] = round(
                bl_forward_ms + pg_forward_ms + lp_forward_ms + mip_forward_ms, 2,
            )
            # Optimality references: LP optimum is a **lower bound**
            # on any integer solution's weighted cost. MILP (if it
            # converges) is the integer optimum of the pruned
            # problem. Both are ``None`` on cached-plan epochs or
            # solver failure.
            ep_cmp["lp_opt"] = (
                round(lp_controller.last_lp_opt, 2)
                if lp_controller.last_lp_opt is not None and refresh_epoch
                else None
            )
            ep_cmp["mip_opt"] = (
                round(mip_controller.last_milp_opt, 2)
                if mip_controller.last_milp_opt is not None and refresh_epoch
                else None
            )
            # Control-layer plan cost per controller (Σ d·c on a common
            # ground-cost surface) + theoretical LP lower bound. Only
            # populated on refresh epochs; cached-plan epochs repeat
            # the previous refresh's plan so repeating the numbers is
            # misleading.
            ep_cmp["bl_plan_cost"] = plan_costs.get("bl")
            ep_cmp["pg_plan_cost"] = plan_costs.get("pg")
            ep_cmp["lp_plan_cost"] = plan_costs.get("lp")
            ep_cmp["mip_plan_cost"] = plan_costs.get("mip")
            ep_cmp["plan_lb"] = plan_lb
            # LB as a demand-weighted mean RTT: divide the LP
            # continuous optimum by the total demand it covers. The
            # LP's objective is Σ d·c evaluated on a fractional
            # assignment, so res.fun / Σd is the expected per-bit
            # RTT at that fractional solution. No integer plan can
            # get strictly below this on the demand-weighted mean
            # axis — it's the single-number RTT floor we plot as a
            # dashed line on c_plan_rtt.
            if plan_lb is not None and _items:
                _total_d = sum(d for _, _, d, _ in _items)
                ep_cmp["plan_lb_mean"] = (
                    round(plan_lb / _total_d, 3) if _total_d > 0 else None
                )
            else:
                ep_cmp["plan_lb_mean"] = None
            # Demand-weighted mean / sat / gnd / p95 / p99 for each
            # controller's plan — parallel to the Forward-side
            # collect() stats, but computed at the control layer
            # (no queuing / no feeder contention / no GroundTruth
            # sampling).
            for label in ("bl", "pg", "lp", "mip"):
                s = plan_stats.get(label) or {}
                for metric in ("mean", "sat", "gnd", "p95", "p99"):
                    ep_cmp[f"{label}_plan_{metric}"] = s.get(metric)
            ep_cmp["mip_solve_status"] = (
                mip_solve_meta.get("status") if refresh_epoch else None
            )
            forward_steps = {
                k: bl_fwd[k] + pg_fwd[k] + lp_fwd[k] + mip_fwd[k]
                for k in bl_fwd.keys() & pg_fwd.keys() & lp_fwd.keys() & mip_fwd.keys()
            }
            ep_cmp["timing"] = {
                "bl_plan_steps": _round_ms(bl_plan_timing),
                "pg_plan_steps": _round_ms(pg_plan_timing),
                "lp_plan_steps": _round_ms(lp_plan_timing),
                "mip_plan_steps": _round_ms(mip_plan_timing),
                "bl_forward_steps": _round_ms(bl_fwd),
                "pg_forward_steps": _round_ms(pg_fwd),
                "lp_forward_steps": _round_ms(lp_fwd),
                "mip_forward_steps": _round_ms(mip_fwd),
                "forward_steps": _round_ms(forward_steps),
            }
            compare_data.append(ep_cmp)
            save_data(bl_data, pg_data, lp_data, mip_data,
                      bl_brk, pg_brk, lp_brk, mip_brk,
                      pop_cmp, compare_data, extract_cache_state(gk_pg))

            t_ep = time.perf_counter() - t_ep_start

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                b = bl_data[-1]; p = pg_data[-1]
                lp = lp_data[-1]; mp = mip_data[-1]
                print(f"{epoch:>4} {b['time_str']:>8} {b['n_flows']:>6} {b['demand_gbps']:>5.0f}G  "
                      f"{b['mean_rtt']:>5.1f} {p['mean_rtt']:>5.1f} {lp['mean_rtt']:>5.1f} {mp['mean_rtt']:>5.1f}  "
                      f"{b['p95_rtt']:>5.1f} {p['p95_rtt']:>5.1f} {lp['p95_rtt']:>5.1f} {mp['p95_rtt']:>5.1f}  "
                      f"{t_ep:>4.1f}s")
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved to", out_file)
    finally:
        if gc_was_enabled:
            gc.enable()

    wall_total = time.perf_counter() - t_wall_start
    print(f"\nDone in {wall_total:.0f}s ({wall_total / 60:.1f} min)")
    print(f"Data: {out_file}")


if __name__ == "__main__":
    main()
