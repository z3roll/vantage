"""Top-level simulation loop orchestration."""

from __future__ import annotations

import gc
import time
import webbrowser
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from vantage.control.evaluation import ControlPlanEvaluation, evaluate_control_plans
from vantage.control.feedback import GroundDelayFeedback
from vantage.control.knowledge import GroundKnowledge
from vantage.control.policy.greedy import GreedyController
from vantage.control.policy.lpround import LPRoundingController
from vantage.control.policy.milp import MILPController
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.control.plane import RoutingPlane
from vantage.forward import RoutingPlaneForward, realize
from vantage.forward.execution.context import RunContext
from vantage.forward.resources.accounting import CapacityView, UsageBook
from vantage.sim.build import SimulationRuntime, build_runtime
from vantage.sim.config import (
    CELL_CACHE,
    DASHBOARD_DIR,
    EPOCH_S,
    N_ANTENNAS_PER_GS,
    REFRESH,
    SAT_FEEDER_CAP_GBPS,
    SimConfig,
)
from vantage.sim.dashboard import DashboardWriter, port_in_use, start_dashboard_server
from vantage.sim.metrics import (
    collect_epoch_summary,
    compute_breakdown,
    compute_epoch_compare,
    compute_pop_compare,
    compute_theoretical_pop_capacity,
    extract_cache_state,
    round_ms,
)

__all__ = ["collect_refresh_gc", "run_simulation"]


def collect_refresh_gc() -> None:
    """Run a full cyclic-GC pass at routing-plane refresh boundaries."""
    gc.collect(2)


def run_simulation(config: SimConfig) -> Path:
    """Run the configured simulation and return the generated data path."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    CELL_CACHE.parent.mkdir(parents=True, exist_ok=True)

    seeds = config.seeds
    print(
        f"Run seed: {seeds.run_seed}"
        + (" (from --seed)" if seeds.seed_source == "cli" else " (auto-generated)")
    )

    url = f"http://localhost:{config.port}/"
    should_open_browser = False
    if not config.no_serve:
        if port_in_use(config.port):
            print(
                f"Port {config.port} already in use - another server is "
                f"already bound to it. Reusing that server."
            )
            print(f"Dashboard: {url}")
            if not config.no_browser:
                should_open_browser = True
        else:
            start_dashboard_server(config.port, DASHBOARD_DIR)
            print(f"Dashboard: {url}")
            if not config.no_browser:
                should_open_browser = True

    start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = DASHBOARD_DIR / f"sim_data_{start_ts}.json"
    index_file = DASHBOARD_DIR / "index.json"

    print("Building world...", flush=True)
    runtime = build_runtime(config)
    if runtime.prune_summary is not None:
        print(runtime.prune_summary)

    writer = DashboardWriter(
        dashboard_dir=DASHBOARD_DIR,
        out_file=out_file,
        index_file=index_file,
        start_ts=start_ts,
        num_epochs=config.num_epochs,
        epoch_s=EPOCH_S,
        refresh_s=REFRESH,
        user_scale=config.user_scale,
        max_gs_per_pop=config.max_gs_per_pop,
        svc_names=runtime.svc_names,
        pop_list=runtime.pop_list,
        antennas_per_gs=N_ANTENNAS_PER_GS,
        sat_feeder_cap_gbps=SAT_FEEDER_CAP_GBPS,
        run_seed=seeds.run_seed,
        seed_source=seeds.seed_source,
        traffic_seed=seeds.traffic_seed,
        ground_seed=seeds.ground_seed,
        ingress_seed_base=seeds.ingress_seed_base,
    )
    n_indexed = writer.rebuild_index()
    print(f"Index: {n_indexed} sim_data files in {DASHBOARD_DIR.name}/")

    print(
        f"Users: {sum(city.user_count for city in runtime.population.city_groups):,}  "
        f"Services: {len(runtime.svc_names)}  PoPs: {len(runtime.pop_list)}"
    )
    print(
        f"Epochs: {config.num_epochs} x {EPOCH_S}s = "
        f"{config.num_epochs * EPOCH_S / 60:.1f} min"
    )
    print(f"Output: {out_file}\n")

    if should_open_browser:
        with suppress(Exception):
            webbrowser.open(url)

    return _run_epoch_loop(config, runtime, writer)


def _run_epoch_loop(
    config: SimConfig,
    runtime: SimulationRuntime,
    writer: DashboardWriter,
) -> Path:
    world = runtime.world
    cell_grid = runtime.cell_grid

    gk_bl = GroundKnowledge(estimator=runtime.geo_delay)
    ctx_bl = RunContext(
        world=world,
        endpoints=runtime.endpoints,
        ground_knowledge=gk_bl,
        ground_truth=runtime.ground_truth,
    )
    gk_greedy = GroundKnowledge(estimator=runtime.geo_delay)
    ctx_greedy = RunContext(
        world=world,
        endpoints=runtime.endpoints,
        ground_knowledge=gk_greedy,
        ground_truth=runtime.ground_truth,
    )
    gk_lp = GroundKnowledge(estimator=runtime.geo_delay)
    ctx_lp = RunContext(
        world=world,
        endpoints=runtime.endpoints,
        ground_knowledge=gk_lp,
        ground_truth=runtime.ground_truth,
    )
    gk_mip = GroundKnowledge(estimator=runtime.geo_delay)
    ctx_mip = RunContext(
        world=world,
        endpoints=runtime.endpoints,
        ground_knowledge=gk_mip,
        ground_truth=runtime.ground_truth,
    )
    feedback_bl = GroundDelayFeedback(gk_bl)
    feedback_greedy = GroundDelayFeedback(gk_greedy)
    feedback_lp = GroundDelayFeedback(gk_lp)
    feedback_mip = GroundDelayFeedback(gk_mip)

    bl_controller = NearestPoPController()
    greedy_controller = GreedyController(
        ground_knowledge=gk_greedy,
        dest_names=tuple(runtime.svc_names),
    )
    lp_controller = LPRoundingController(
        ground_knowledge=gk_lp,
        dest_names=tuple(runtime.svc_names),
    )
    mip_controller = MILPController(
        ground_knowledge=gk_mip,
        dest_names=tuple(runtime.svc_names),
    )

    bl_plane: RoutingPlane | None = None
    greedy_plane: RoutingPlane | None = None
    lp_plane: RoutingPlane | None = None
    mip_plane: RoutingPlane | None = None
    bl_forward: RoutingPlaneForward | None = None
    greedy_forward: RoutingPlaneForward | None = None
    lp_forward: RoutingPlaneForward | None = None
    mip_forward: RoutingPlaneForward | None = None

    bl_data: list = []
    greedy_data: list = []
    lp_data: list = []
    mip_data: list = []
    compare_data: list = []

    print(
        f"{'ep':>4} {'time':>8} {'flows':>6} {'dem':>6}  "
        f"{'BL':>6} {'Greedy':>6} {'LP':>6} {'MIP':>6}  "
        f"{'BL95':>6} {'Gre95':>6} {'LP95':>6} {'MIP95':>6}  "
        f"{'wall':>5}"
    )
    print("-" * 110)

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    t_wall_start = time.perf_counter()

    try:
        for epoch in range(config.num_epochs):
            t = epoch * EPOCH_S
            t_ep_start = time.perf_counter()

            snap = world.snapshot_at(epoch, t)
            gk_bl.set_clock(t)
            gk_greedy.set_clock(t)
            gk_lp.set_clock(t)
            gk_mip.set_clock(t)
            refresh_epoch = bl_plane is None or epoch % REFRESH == 0

            demand = runtime.traffic.generate(epoch)
            current_demand = {
                (flow_key.src, flow_key.dst): flow_demand
                for flow_key, flow_demand in demand.flows.items()
            }

            bl_plan_ms: float | None = None
            bl_plan_timing: dict = {}
            if refresh_epoch:
                t_bl_plan = time.perf_counter()
                bl_plane = bl_controller.compute_routing_plane(
                    snap,
                    cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                bl_plan_ms = (time.perf_counter() - t_bl_plan) * 1000
                bl_plan_timing = dict(bl_controller.last_timing)

            greedy_plan_ms: float | None = None
            greedy_plan_timing: dict = {}
            if refresh_epoch:
                t_greedy_plan = time.perf_counter()
                greedy_plane = greedy_controller.compute_routing_plane(
                    snapshot=snap,
                    cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                greedy_plan_ms = (time.perf_counter() - t_greedy_plan) * 1000
                greedy_plan_timing = dict(greedy_controller.last_timing)

            lp_plan_ms: float | None = None
            lp_plan_timing: dict = {}
            if refresh_epoch:
                t_lp_plan = time.perf_counter()
                lp_plane = lp_controller.compute_routing_plane(
                    snapshot=snap,
                    cell_grid=cell_grid,
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
                    snapshot=snap,
                    cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                mip_plan_ms = (time.perf_counter() - t_mip_plan) * 1000
                mip_plan_timing = dict(mip_controller.last_timing)
                mip_solve_meta = dict(mip_controller.last_solve_meta)

            plan_eval = ControlPlanEvaluation.empty()
            if refresh_epoch:
                plan_eval = evaluate_control_plans(
                    snapshot=snap,
                    cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    routing_planes={
                        "bl": bl_plane,
                        "greedy": greedy_plane,
                        "lp": lp_plane,
                        "mip": mip_plane,
                    },
                    reference_ground_knowledge=gk_greedy,
                    dest_names=runtime.svc_names,
                    current_epoch=epoch,
                    lp_lower_bound=lp_controller.last_lp_opt,
                )

            if refresh_epoch and gc_was_enabled:
                collect_refresh_gc()

            view_bl = CapacityView.from_snapshot(
                sat_state=snap.satellite,
                shell=world.shell,
                ground_stations=runtime.gs_by_id,
            )
            book_bl = UsageBook(view=view_bl)
            bl_forward = RoutingPlaneForward.for_epoch(
                bl_forward,
                bl_plane,
                cell_grid,
                book_bl,
            )
            result_bl = realize(
                bl_forward,
                snap,
                demand,
                ctx_bl,
                ingress_seed_base=config.seeds.ingress_seed_base,
            )
            feedback_bl.observe(result_bl)

            view_greedy = CapacityView.from_snapshot(
                sat_state=snap.satellite,
                shell=world.shell,
                ground_stations=runtime.gs_by_id,
            )
            book_greedy = UsageBook(view=view_greedy)
            greedy_forward = RoutingPlaneForward.for_epoch(
                greedy_forward,
                greedy_plane,
                cell_grid,
                book_greedy,
            )
            result_greedy = realize(
                greedy_forward,
                snap,
                demand,
                ctx_greedy,
                ingress_seed_base=config.seeds.ingress_seed_base,
            )
            feedback_greedy.observe(result_greedy)

            view_lp = CapacityView.from_snapshot(
                sat_state=snap.satellite,
                shell=world.shell,
                ground_stations=runtime.gs_by_id,
            )
            book_lp = UsageBook(view=view_lp)
            lp_forward = RoutingPlaneForward.for_epoch(
                lp_forward,
                lp_plane,
                cell_grid,
                book_lp,
            )
            result_lp = realize(
                lp_forward,
                snap,
                demand,
                ctx_lp,
                ingress_seed_base=config.seeds.ingress_seed_base,
            )
            feedback_lp.observe(result_lp)

            view_mip = CapacityView.from_snapshot(
                sat_state=snap.satellite,
                shell=world.shell,
                ground_stations=runtime.gs_by_id,
            )
            book_mip = UsageBook(view=view_mip)
            mip_forward = RoutingPlaneForward.for_epoch(
                mip_forward,
                mip_plane,
                cell_grid,
                book_mip,
            )
            result_mip = realize(
                mip_forward,
                snap,
                demand,
                ctx_mip,
                ingress_seed_base=config.seeds.ingress_seed_base,
            )
            feedback_mip.observe(result_mip)

            pop_capacity = compute_theoretical_pop_capacity(
                runtime.pop_gs_list,
                runtime.pop_list,
                antennas_per_gs=N_ANTENNAS_PER_GS,
                sat_feeder_cap_gbps=SAT_FEEDER_CAP_GBPS,
            )
            bl_data.append(_collect_epoch(epoch, result_bl, book_bl, pop_capacity, runtime))
            greedy_data.append(_collect_epoch(epoch, result_greedy, book_greedy, pop_capacity, runtime))
            lp_data.append(_collect_epoch(epoch, result_lp, book_lp, pop_capacity, runtime))
            mip_data.append(_collect_epoch(epoch, result_mip, book_mip, pop_capacity, runtime))

            latest_breakdown = {
                "baseline": compute_breakdown(result_bl),
                "greedy": compute_breakdown(result_greedy),
                "lpround": compute_breakdown(result_lp),
                "milp": compute_breakdown(result_mip),
            }
            pop_cmp = compute_pop_compare(result_bl, result_greedy)
            ep_cmp = compute_epoch_compare(result_bl, result_greedy, result_lp, result_mip)
            ep_cmp["epoch"] = epoch
            ep_cmp["time_str"] = bl_data[-1]["time_str"]
            ep_cmp["bl_plan_ms"] = round(bl_plan_ms, 2) if bl_plan_ms is not None else None
            ep_cmp["greedy_plan_ms"] = round(greedy_plan_ms, 2) if greedy_plan_ms is not None else None
            ep_cmp["lp_plan_ms"] = round(lp_plan_ms, 2) if lp_plan_ms is not None else None
            ep_cmp["mip_plan_ms"] = round(mip_plan_ms, 2) if mip_plan_ms is not None else None

            bl_fwd = dict(result_bl.forward_timing_ms)
            greedy_fwd = dict(result_greedy.forward_timing_ms)
            lp_fwd = dict(result_lp.forward_timing_ms)
            mip_fwd = dict(result_mip.forward_timing_ms)
            bl_forward_ms = bl_fwd.get("total_ms", 0.0)
            greedy_forward_ms = greedy_fwd.get("total_ms", 0.0)
            lp_forward_ms = lp_fwd.get("total_ms", 0.0)
            mip_forward_ms = mip_fwd.get("total_ms", 0.0)
            ep_cmp["bl_forward_ms"] = round(bl_forward_ms, 2)
            ep_cmp["greedy_forward_ms"] = round(greedy_forward_ms, 2)
            ep_cmp["lp_forward_ms"] = round(lp_forward_ms, 2)
            ep_cmp["mip_forward_ms"] = round(mip_forward_ms, 2)
            ep_cmp["bl_total_ms"] = round((bl_plan_ms or 0.0) + bl_forward_ms, 2)
            ep_cmp["greedy_total_ms"] = round((greedy_plan_ms or 0.0) + greedy_forward_ms, 2)
            ep_cmp["lp_total_ms"] = round((lp_plan_ms or 0.0) + lp_forward_ms, 2)
            ep_cmp["mip_total_ms"] = round((mip_plan_ms or 0.0) + mip_forward_ms, 2)
            ep_cmp["forward_total_ms"] = round(
                bl_forward_ms + greedy_forward_ms + lp_forward_ms + mip_forward_ms,
                2,
            )
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
            ep_cmp.update(plan_eval.dashboard_fields())
            ep_cmp["mip_solve_status"] = (
                mip_solve_meta.get("status") if refresh_epoch else None
            )
            forward_steps = {
                key: bl_fwd[key] + greedy_fwd[key] + lp_fwd[key] + mip_fwd[key]
                for key in bl_fwd.keys() & greedy_fwd.keys() & lp_fwd.keys() & mip_fwd.keys()
            }
            ep_cmp["timing"] = {
                "bl_plan_steps": round_ms(bl_plan_timing),
                "greedy_plan_steps": round_ms(greedy_plan_timing),
                "lp_plan_steps": round_ms(lp_plan_timing),
                "mip_plan_steps": round_ms(mip_plan_timing),
                "bl_forward_steps": round_ms(bl_fwd),
                "greedy_forward_steps": round_ms(greedy_fwd),
                "lp_forward_steps": round_ms(lp_fwd),
                "mip_forward_steps": round_ms(mip_fwd),
                "forward_steps": round_ms(forward_steps),
            }
            compare_data.append(ep_cmp)
            writer.save_data(
                baseline=bl_data,
                greedy=greedy_data,
                lpround=lp_data,
                milp=mip_data,
                latest_breakdown=latest_breakdown,
                latest_pop_compare=pop_cmp,
                epoch_compare=compare_data,
                cache_state=extract_cache_state(gk_greedy),
            )

            t_ep = time.perf_counter() - t_ep_start
            if epoch % 10 == 0 or epoch == config.num_epochs - 1:
                b = bl_data[-1]
                p = greedy_data[-1]
                lp = lp_data[-1]
                mp = mip_data[-1]
                print(
                    f"{epoch:>4} {b['time_str']:>8} {b['n_flows']:>6} "
                    f"{b['demand_gbps']:>5.0f}G  "
                    f"{b['mean_rtt']:>5.1f} {p['mean_rtt']:>5.1f} "
                    f"{lp['mean_rtt']:>5.1f} {mp['mean_rtt']:>5.1f}  "
                    f"{b['p95_rtt']:>5.1f} {p['p95_rtt']:>5.1f} "
                    f"{lp['p95_rtt']:>5.1f} {mp['p95_rtt']:>5.1f}  "
                    f"{t_ep:>4.1f}s"
                )
    except KeyboardInterrupt:
        print("\nInterrupted - partial results saved to", writer.out_file)
    finally:
        if gc_was_enabled:
            gc.enable()

    wall_total = time.perf_counter() - t_wall_start
    print(f"\nDone in {wall_total:.0f}s ({wall_total / 60:.1f} min)")
    print(f"Data: {writer.out_file}")
    return writer.out_file


def _collect_epoch(
    epoch: int,
    result,
    usage_book: UsageBook,
    pop_capacity: dict[str, float],
    runtime: SimulationRuntime,
) -> dict:
    return collect_epoch_summary(
        epoch=epoch,
        result=result,
        usage_book=usage_book,
        pop_capacity=pop_capacity,
        svc_names=runtime.svc_names,
        pop_list=runtime.pop_list,
        pop_gs_list=runtime.pop_gs_list,
        epoch_s=EPOCH_S,
    )
