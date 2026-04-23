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
import errno
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from vantage.common.seed import derive_subseed, fresh_run_seed
from vantage.control.policy.greedy import ProgressiveController
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


def country_of(src_name: str) -> str:
    parts = src_name.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "city" else "??"


def _round_ms(d) -> dict:
    """Round a ``{step → ms}`` mapping for JSON output (2 decimals)."""
    return {k: round(float(v), 2) for k, v in d.items()}


def start_dashboard_server(port: int, directory: Path) -> socketserver.TCPServer:
    """Spin up http.server in a daemon thread rooted at ``directory``."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(directory), **k)

        def log_message(self, *_a, **_k):  # silence per-request spam
            pass

    class _QuietServer(socketserver.ThreadingTCPServer):
        # Browsers routinely abort in-flight GETs while the dashboard
        # polls (tab switch, dropdown change mid-fetch). Suppress the
        # connection-abort tracebacks ``ThreadingTCPServer`` would
        # otherwise dump to stderr — they are benign.
        def handle_error(self, request, client_address):
            import sys
            exc = sys.exc_info()[1]
            if isinstance(exc, (BrokenPipeError, ConnectionResetError,
                                ConnectionAbortedError)):
                return
            super().handle_error(request, client_address)

    httpd = _QuietServer(("", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


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
    p.add_argument("--seed", type=int, default=None,
                   help="Run-level seed controlling every stochastic subsystem "
                        "(traffic AR(1)/Poisson, ground-delay LogNormal sampling, "
                        "ingress-sat selection). If omitted, a fresh random seed "
                        "is drawn for this run and printed. Use the same seed to "
                        "reproduce a run exactly.")
    return p.parse_args()


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
    httpd = None
    url = f"http://localhost:{args.port}/"
    # Browser auto-open is deferred to after the world-build summary
    # below — opening the tab before the build finishes made it spin
    # on an empty index for ~30 s. ``should_open_browser`` flips on
    # whenever the dashboard URL is reachable (either we bound to it
    # ourselves or a pre-existing server is already serving the same
    # ``dashboard/`` directory on that port) and the user did not
    # pass ``--no-browser``.
    should_open_browser = False
    if not args.no_serve:
        # A port already in use is a recoverable condition — it's
        # typically another run.py still serving the same dashboard
        # directory, so the URL the user sees is still live. We print
        # it to stdout and auto-open the browser anyway. Other bind
        # errors are still fatal.
        try:
            httpd = start_dashboard_server(args.port, DASHBOARD_DIR)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            print(f"Port {args.port} already in use — another server is "
                  f"already bound to it. Reusing that server.")
            print(f"Dashboard: {url}")
            if not args.no_browser:
                should_open_browser = True
        else:
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
        flows = result.flow_outcomes
        rtts = [f.total_rtt for f in flows]
        svc = defaultdict(lambda: {"rtts": [], "sat": [], "gnd": [], "demand": 0.0})
        pop_load = defaultdict(float)
        for f in flows:
            svc[f.flow_key.dst]["rtts"].append(f.total_rtt)
            svc[f.flow_key.dst]["sat"].append(f.satellite_rtt)
            svc[f.flow_key.dst]["gnd"].append(f.ground_rtt)
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
                svc_out[s] = {
                    "mean": round(np.mean(d["rtts"]), 2),
                    "p95": round(pct(d["rtts"], 95), 2),
                    "p99": round(pct(d["rtts"], 99), 2),
                    "sat": round(np.mean(d["sat"]), 2),
                    "gnd": round(np.mean(d["gnd"]), 2),
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
        return {
            "epoch": epoch, "time_s": epoch * EPOCH_S,
            "time_str": f"{hh:02d}:{mm:02d}:{ss:02d}",
            "n_flows": len(flows),
            "demand_gbps": round(sum(f.demand_gbps for f in flows), 1),
            "mean_rtt": round(np.mean(rtts), 2) if rtts else 0,
            "p50_rtt": round(pct(rtts, 50), 2),
            "p95_rtt": round(pct(rtts, 95), 2),
            "p99_rtt": round(pct(rtts, 99), 2),
            "mean_sat": round(np.mean([f.satellite_rtt for f in flows]), 2) if flows else 0,
            "mean_gnd": round(np.mean([f.ground_rtt for f in flows]), 2) if flows else 0,
            "mean_prop": round(np.mean([f.propagation_rtt for f in flows]), 2) if flows else 0,
            "mean_queue": round(np.mean([f.queuing_rtt for f in flows]), 2) if flows else 0,
            "mean_tx": round(np.mean([f.transmission_rtt for f in flows]), 2) if flows else 0,
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

    def compute_epoch_compare(result_bl, result_pg) -> dict:
        bl_by = {f.flow_key: f for f in result_bl.flow_outcomes}
        pg_by = {f.flow_key: f for f in result_pg.flow_outcomes}
        common = set(bl_by) & set(pg_by)
        TOL = 0.5
        total_n = re_n = impr_n = worse_n = neutral_n = 0
        sum_bl_sat = sum_pg_sat = 0.0
        sum_bl_gnd = sum_pg_gnd = 0.0
        bl_rtts: list[float] = []
        pg_rtts: list[float] = []
        for fk in common:
            fb, fp = bl_by[fk], pg_by[fk]
            total_n += 1
            bl_rtts.append(fb.total_rtt); pg_rtts.append(fp.total_rtt)
            sum_bl_sat += fb.satellite_rtt; sum_pg_sat += fp.satellite_rtt
            sum_bl_gnd += fb.ground_rtt; sum_pg_gnd += fp.ground_rtt
            d_total = fp.total_rtt - fb.total_rtt
            if fb.pop_code != fp.pop_code:
                re_n += 1
                if d_total < -TOL: impr_n += 1
                elif d_total > TOL: worse_n += 1
                else: neutral_n += 1

        def _avg(s): return s / total_n if total_n else 0.0
        return {
            "total_flows": total_n, "reroute_flows": re_n,
            "reroute_pct": round(re_n / total_n * 100, 2) if total_n else 0,
            "impr_flows": impr_n, "worse_flows": worse_n, "neutral_flows": neutral_n,
            "impr_pct_of_re": round(impr_n / re_n * 100, 1) if re_n else 0,
            "worse_pct_of_re": round(worse_n / re_n * 100, 1) if re_n else 0,
            "bl_mean_rtt": round(_avg(sum(bl_rtts)), 2),
            "pg_mean_rtt": round(_avg(sum(pg_rtts)), 2),
            "bl_sat_rtt": round(_avg(sum_bl_sat), 2),
            "pg_sat_rtt": round(_avg(sum_pg_sat), 2),
            "bl_gnd_rtt": round(_avg(sum_bl_gnd), 2),
            "pg_gnd_rtt": round(_avg(sum_pg_gnd), 2),
            "bl_p95_rtt": round(pct(bl_rtts, 95), 2),
            "pg_p95_rtt": round(pct(pg_rtts, 95), 2),
            "bl_p99_rtt": round(pct(bl_rtts, 99), 2),
            "pg_p99_rtt": round(pct(pg_rtts, 99), 2),
        }

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
            "user_scale": user_scale, "mtime": time.time(),
        }
        existing = {p.name for p in DASHBOARD_DIR.glob("sim_data_*.json")}
        entries = [e for e in by_name.values() if e["filename"] in existing]
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        tmp = index_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"files": entries}, f)
        tmp.replace(index_file)

    def save_data(bl_data, pg_data, bl_brk, pg_brk, pop_cmp, ep_cmp, cache_state):
        out = {
            "config": {
                "num_epochs": num_epochs, "epoch_s": EPOCH_S, "refresh_s": REFRESH,
                "user_scale": user_scale, "services": svc_names,
                "total_capacity_gbps": 8 * SAT_FEEDER_CAP_GBPS * len(pop_list),
                "started_at": start_ts,
                "run_seed": run_seed,
                "seed_source": "cli" if args.seed is not None else "auto",
                "sub_seeds": {
                    "traffic": traffic_seed,
                    "ground_delay": ground_seed,
                    "ingress": ingress_seed_base,
                },
            },
            "baseline": bl_data, "progressive": pg_data,
            "latest_breakdown": {"baseline": bl_brk, "progressive": pg_brk},
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
    feedback_bl = GroundDelayFeedback(gk_bl)
    feedback_pg = GroundDelayFeedback(gk_pg)
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
    pg_controller = ProgressiveController(ground_knowledge=gk_pg, dest_names=tuple(svc_names))

    bl_plane = pg_plane = None
    # Forward lifecycle (reuse vs. rebuild, cache invalidation) is
    # owned by RoutingPlaneForward.for_epoch — we just carry the
    # previous instances so the factory can decide.
    bl_forward: RoutingPlaneForward | None = None
    pg_forward: RoutingPlaneForward | None = None
    bl_data: list = []
    pg_data: list = []
    compare_data: list = []

    print(f"{'ep':>5} {'time':>8} {'flows':>6} {'demand':>7}  "
          f"{'BL_rtt':>7} {'PG_rtt':>7} {'diff':>6}  "
          f"{'BL_p95':>7} {'PG_p95':>7}  "
          f"{'BL_p99':>7} {'PG_p99':>7}  "
          f"{'wall':>5}")
    print("-" * 95)

    t_wall_start = time.perf_counter()

    try:
        for epoch in range(num_epochs):
            t = epoch * EPOCH_S
            t_ep_start = time.perf_counter()

            snap = world.snapshot_at(epoch, t)
            gk_bl.set_clock(t); gk_pg.set_clock(t)

            bl_plan_ms: float | None = None
            bl_plan_timing: dict = {}
            if bl_plane is None or epoch % REFRESH == 0:
                t_bl_plan = time.perf_counter()
                bl_plane = bl_controller.compute_routing_plane(snap, cell_grid, version=epoch)
                bl_plan_ms = (time.perf_counter() - t_bl_plan) * 1000
                bl_plan_timing = dict(bl_controller.last_timing)

            view_bl = CapacityView.from_snapshot(sat_state=snap.satellite, shell=world.shell, ground_stations=gs_by_id)
            book_bl = UsageBook(view=view_bl)
            # One stochastic draw of traffic per epoch; both controllers
            # realize against the exact same ``TrafficDemand``.
            demand = traffic.generate(epoch)
            bl_forward = RoutingPlaneForward.for_epoch(
                bl_forward, bl_plane, cell_grid, book_bl,
            )
            result_bl = realize(
                bl_forward, snap, demand, ctx_bl,
                ingress_seed_base=ingress_seed_base,
            )
            feedback_bl.observe(result_bl)

            current_demand = {(fk.src, fk.dst): d for fk, d in demand.flows.items()}

            pg_plan_ms: float | None = None
            pg_plan_timing: dict = {}
            if pg_plane is None or epoch % REFRESH == 0:
                t_pg_plan = time.perf_counter()
                pg_plane = pg_controller.compute_routing_plane(
                    snapshot=snap, cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                pg_plan_ms = (time.perf_counter() - t_pg_plan) * 1000
                pg_plan_timing = dict(pg_controller.last_timing)

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

            pop_capacity = dynamic_pop_capacity(snap)
            bl_data.append(collect(epoch, result_bl, book_bl, pop_capacity))
            pg_data.append(collect(epoch, result_pg, book_pg, pop_capacity))
            bl_brk = compute_breakdown(result_bl)
            pg_brk = compute_breakdown(result_pg)
            pop_cmp = compute_pop_compare(result_bl, result_pg)
            ep_cmp = compute_epoch_compare(result_bl, result_pg)
            ep_cmp["epoch"] = epoch
            ep_cmp["time_str"] = bl_data[-1]["time_str"]
            ep_cmp["bl_plan_ms"] = round(bl_plan_ms, 2) if bl_plan_ms is not None else None
            ep_cmp["pg_plan_ms"] = round(pg_plan_ms, 2) if pg_plan_ms is not None else None
            # Forward timing is always present (realize always runs).
            # Plan timing breakdowns are empty on cached-plan epochs —
            # callers reading the dashboard must treat missing keys as
            # "not measured this epoch" rather than "zero cost".
            bl_fwd = dict(result_bl.forward_timing_ms)
            pg_fwd = dict(result_pg.forward_timing_ms)
            bl_forward_ms = bl_fwd.get("total_ms", 0.0)
            pg_forward_ms = pg_fwd.get("total_ms", 0.0)
            bl_total_ms = (bl_plan_ms or 0.0) + bl_forward_ms
            pg_total_ms = (pg_plan_ms or 0.0) + pg_forward_ms
            ep_cmp["bl_forward_ms"] = round(bl_forward_ms, 2)
            ep_cmp["pg_forward_ms"] = round(pg_forward_ms, 2)
            ep_cmp["bl_total_ms"] = round(bl_total_ms, 2)
            ep_cmp["pg_total_ms"] = round(pg_total_ms, 2)
            ep_cmp["forward_total_ms"] = round(bl_forward_ms + pg_forward_ms, 2)
            # Aggregated forward steps across BL + PG, restricted to
            # keys present in both — the dashboard uses this for a
            # combined drill-down chart so users can see where the
            # realize() pipeline spends time per epoch.
            forward_steps = {
                k: bl_fwd[k] + pg_fwd[k]
                for k in bl_fwd.keys() & pg_fwd.keys()
            }
            ep_cmp["timing"] = {
                "bl_plan_steps": _round_ms(bl_plan_timing),
                "pg_plan_steps": _round_ms(pg_plan_timing),
                "bl_forward_steps": _round_ms(bl_fwd),
                "pg_forward_steps": _round_ms(pg_fwd),
                "forward_steps": _round_ms(forward_steps),
            }
            compare_data.append(ep_cmp)
            save_data(bl_data, pg_data, bl_brk, pg_brk, pop_cmp, compare_data,
                      extract_cache_state(gk_pg))

            t_ep = time.perf_counter() - t_ep_start

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                b = bl_data[-1]; p = pg_data[-1]
                diff = p["mean_rtt"] - b["mean_rtt"]
                print(f"{epoch:>5} {b['time_str']:>8} {b['n_flows']:>6} {b['demand_gbps']:>6.0f}Gb  "
                      f"{b['mean_rtt']:>6.1f}ms {p['mean_rtt']:>6.1f}ms {diff:>+5.1f}ms  "
                      f"{b['p95_rtt']:>6.1f}ms {p['p95_rtt']:>6.1f}ms  "
                      f"{b['p99_rtt']:>6.1f}ms {p['p99_rtt']:>6.1f}ms  "
                      f"{t_ep:>4.1f}s")
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved to", out_file)

    wall_total = time.perf_counter() - t_wall_start
    print(f"\nDone in {wall_total:.0f}s ({wall_total / 60:.1f} min)")
    print(f"Data: {out_file}")
    # No trailing ``while True: sleep`` keep-alive: the HTTP server
    # runs in a daemon thread, so returning from ``main`` lets the
    # interpreter tear it down as the process exits. Users who want
    # the dashboard up longer can keep a separate instance running
    # in another terminal — but the sim run itself shouldn't require
    # a second Ctrl-C to quit.


if __name__ == "__main__":
    main()
