"""Top-level simulation loop orchestration."""

from __future__ import annotations

import gc
import time
import webbrowser
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vantage.control.evaluation import ControlPlanEvaluation, evaluate_control_plans
from vantage.control.feedback import GroundDelayFeedback
from vantage.control.knowledge import GroundKnowledge
from vantage.control.policy.greedy import GreedyController
from vantage.control.policy.lpround import LPRoundingController
from vantage.control.policy.milp import MILPController
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.control.policy.optimizer import (
    PathAwareNearestBaselineController,
    PathAwareOptimizerController,
)
from vantage.control.policy.progressive import ProgressiveSpilloverController
from vantage.control.plane import RoutingPlane
from vantage.forward import PlannedRoutingPlaneForward, RoutingPlaneForward, realize
from vantage.forward.execution.context import RunContext
from vantage.forward.resources.accounting import CapacityView, UsageBook
from vantage.sim.build import SimulationRuntime, build_runtime
from vantage.sim.config import (
    CELL_CACHE,
    DASHBOARD_DIR,
    EPOCH_S,
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


_CONTROL_PREFIX = {
    "baseline": "bl",
    "optimizer_baseline": "optbl",
    "progressive": "progressive",
    "optimizer": "opt",
    "greedy": "greedy",
    "lpround": "lp",
    "milp": "mip",
}
_CONTROL_MEAN_HEADER = {
    "baseline": "BL",
    "optimizer_baseline": "OptBL",
    "progressive": "Prog",
    "optimizer": "Opt",
    "greedy": "Greedy",
    "lpround": "LP",
    "milp": "MIP",
}
_CONTROL_P95_HEADER = {
    "baseline": "BL95",
    "optimizer_baseline": "OBL95",
    "progressive": "Pr95",
    "optimizer": "Opt95",
    "greedy": "Gre95",
    "lpround": "LP95",
    "milp": "MIP95",
}


@dataclass(slots=True)
class _ControlSeries:
    name: str
    prefix: str
    mean_header: str
    p95_header: str
    controller: Any
    ground_knowledge: GroundKnowledge
    context: RunContext
    feedback: GroundDelayFeedback
    plane: RoutingPlane | None = None
    forward: Any | None = None
    data: list = field(default_factory=list)
    last_result: Any | None = None
    last_book: UsageBook | None = None
    plan_ms: float | None = None
    plan_timing: dict[str, float] = field(default_factory=dict)
    solve_meta: dict[str, Any] = field(default_factory=dict)


def collect_refresh_gc() -> None:
    """Run a full cyclic-GC pass at routing-plane refresh boundaries."""
    gc.collect(2)


def _make_control_series(
    name: str,
    runtime: SimulationRuntime,
) -> _ControlSeries:
    gk = GroundKnowledge(estimator=runtime.geo_delay)
    context = RunContext(
        world=runtime.world,
        endpoints=runtime.endpoints,
        ground_knowledge=gk,
        ground_truth=runtime.ground_truth,
    )
    feedback = GroundDelayFeedback(gk)
    dest_names = tuple(runtime.svc_names)

    if name == "baseline":
        controller = NearestPoPController()
    elif name == "optimizer_baseline":
        controller = PathAwareNearestBaselineController(
            ground_knowledge=gk,
            dest_names=dest_names,
        )
    elif name == "progressive":
        controller = ProgressiveSpilloverController(
            ground_knowledge=gk,
            dest_names=dest_names,
        )
    elif name == "greedy":
        controller = GreedyController(
            ground_knowledge=gk,
            dest_names=dest_names,
        )
    elif name == "optimizer":
        controller = PathAwareOptimizerController(
            ground_knowledge=gk,
            dest_names=dest_names,
        )
    elif name == "lpround":
        controller = LPRoundingController(
            ground_knowledge=gk,
            dest_names=dest_names,
        )
    elif name == "milp":
        controller = MILPController(
            ground_knowledge=gk,
            dest_names=dest_names,
        )
    else:
        raise ValueError(f"unknown control algorithm {name!r}")

    return _ControlSeries(
        name=name,
        prefix=_CONTROL_PREFIX[name],
        mean_header=_CONTROL_MEAN_HEADER[name],
        p95_header=_CONTROL_P95_HEADER[name],
        controller=controller,
        ground_knowledge=gk,
        context=context,
        feedback=feedback,
    )


def _reference_ground_knowledge(states: list[_ControlSeries]) -> GroundKnowledge:
    for state in states:
        if state.name == "greedy":
            return state.ground_knowledge
    return states[0].ground_knowledge


def _find_series(
    states: list[_ControlSeries],
    name: str,
) -> _ControlSeries | None:
    for state in states:
        if state.name == name:
            return state
    return None


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
    display_pop_capacity = compute_theoretical_pop_capacity(
        runtime.pop_gs_list,
        runtime.pop_list,
        antennas_per_gs=config.egress_top_k,
        sat_feeder_cap_gbps=SAT_FEEDER_CAP_GBPS,
    )

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
        egress_top_k=config.egress_top_k,
        enforce_isl_capacity=config.enforce_isl_capacity,
        series=config.control_algorithms,
        svc_names=runtime.svc_names,
        pop_list=runtime.pop_list,
        total_capacity_gbps=sum(display_pop_capacity.values()),
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
    print(f"Controls: {', '.join(config.control_algorithms)}")
    print(f"Egress top-k per GS: {config.egress_top_k}")
    print(
        "ISL capacity: "
        f"{'enabled' if config.enforce_isl_capacity else 'disabled'}"
    )
    print(f"Output: {out_file}\n")

    if should_open_browser:
        with suppress(Exception):
            webbrowser.open(url)

    return _run_epoch_loop(config, runtime, writer, display_pop_capacity)


def _run_epoch_loop(
    config: SimConfig,
    runtime: SimulationRuntime,
    writer: DashboardWriter,
    pop_capacity: dict[str, float],
) -> Path:
    world = runtime.world
    cell_grid = runtime.cell_grid
    states = [
        _make_control_series(name, runtime)
        for name in config.control_algorithms
    ]
    compare_data: list = []
    mean_header = " ".join(f"{state.mean_header:>6}" for state in states)
    p95_header = " ".join(f"{state.p95_header:>6}" for state in states)
    print(
        f"{'ep':>4} {'time':>8} {'flows':>6} {'dem':>6}  "
        f"{mean_header}  {p95_header}  {'wall':>5}"
    )
    print("-" * max(58, 38 + len(states) * 14))

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    t_wall_start = time.perf_counter()

    try:
        for epoch in range(config.num_epochs):
            t = epoch * EPOCH_S
            t_ep_start = time.perf_counter()

            snap = world.snapshot_at(epoch, t)
            for state in states:
                state.ground_knowledge.set_clock(t)
            refresh_epoch = (
                any(state.plane is None for state in states)
                or epoch % REFRESH == 0
            )

            demand = runtime.traffic.generate(epoch)
            current_demand = {
                (flow_key.src, flow_key.dst): flow_demand
                for flow_key, flow_demand in demand.flows.items()
            }

            for state in states:
                state.plan_ms = None
                state.plan_timing = {}
                state.solve_meta = {}
                if not refresh_epoch:
                    continue
                t_plan = time.perf_counter()
                state.plane = state.controller.compute_routing_plane(
                    snapshot=snap,
                    cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    version=epoch,
                )
                state.plan_ms = (time.perf_counter() - t_plan) * 1000
                state.plan_timing = dict(state.controller.last_timing)
                if state.name == "milp":
                    state.solve_meta = dict(state.controller.last_solve_meta)

            plan_eval = ControlPlanEvaluation.empty()
            if refresh_epoch:
                lp_state = _find_series(states, "lpround")
                lp_lower_bound = (
                    None if lp_state is None
                    else lp_state.controller.last_lp_opt
                )
                plan_eval = evaluate_control_plans(
                    snapshot=snap,
                    cell_grid=cell_grid,
                    demand_per_pair=current_demand,
                    routing_planes={
                        state.prefix: state.plane
                        for state in states
                        if state.plane is not None
                    },
                    reference_ground_knowledge=_reference_ground_knowledge(states),
                    dest_names=runtime.svc_names,
                    current_epoch=epoch,
                    lp_lower_bound=lp_lower_bound,
                )

            if refresh_epoch and gc_was_enabled:
                collect_refresh_gc()

            results: dict[str, Any] = {}
            latest_breakdown: dict[str, dict] = {}
            forward_timings: dict[str, dict] = {}
            for state in states:
                if state.plane is None:
                    raise RuntimeError(f"{state.name} control plane was not built")
                view = CapacityView.from_snapshot(
                    sat_state=snap.satellite,
                    shell=world.shell,
                    ground_stations=runtime.gs_by_id,
                )
                book = UsageBook(view=view)
                if state.plane.path_hints is not None:
                    planned_previous = (
                        state.forward
                        if isinstance(state.forward, PlannedRoutingPlaneForward)
                        else None
                    )
                    state.forward = PlannedRoutingPlaneForward.for_epoch(
                        planned_previous,
                        state.plane,
                        cell_grid,
                        book,
                    )
                else:
                    routing_previous = (
                        state.forward
                        if isinstance(state.forward, RoutingPlaneForward)
                        else None
                    )
                    state.forward = RoutingPlaneForward.for_epoch(
                        routing_previous,
                        state.plane,
                        cell_grid,
                        book,
                        enforce_isl_capacity=config.enforce_isl_capacity,
                    )
                result = realize(
                    state.forward,
                    snap,
                    demand,
                    state.context,
                    ingress_seed_base=config.seeds.ingress_seed_base,
                )
                state.feedback.observe(result)
                state.last_result = result
                state.last_book = book
                state.data.append(
                    _collect_epoch(epoch, result, book, pop_capacity, runtime)
                )
                results[state.name] = result
                latest_breakdown[state.name] = compute_breakdown(result)
                forward_timings[state.name] = dict(result.forward_timing_ms)

            if "baseline" in results and "greedy" in results:
                pop_cmp = compute_pop_compare(results["baseline"], results["greedy"])
            else:
                pop_cmp = {}

            ep_cmp = compute_epoch_compare(results=results)
            ep_cmp["epoch"] = epoch
            ep_cmp["time_str"] = states[0].data[-1]["time_str"]

            forward_total_ms = 0.0
            timing: dict[str, dict] = {}
            for state in states:
                fwd = forward_timings[state.name]
                forward_ms = fwd.get("total_ms", 0.0)
                forward_total_ms += forward_ms
                ep_cmp[f"{state.prefix}_plan_ms"] = (
                    round(state.plan_ms, 2)
                    if state.plan_ms is not None
                    else None
                )
                ep_cmp[f"{state.prefix}_forward_ms"] = round(forward_ms, 2)
                ep_cmp[f"{state.prefix}_total_ms"] = round(
                    (state.plan_ms or 0.0) + forward_ms,
                    2,
                )
                timing[f"{state.prefix}_plan_steps"] = round_ms(
                    state.plan_timing
                )
                timing[f"{state.prefix}_forward_steps"] = round_ms(fwd)

            ep_cmp["forward_total_ms"] = round(forward_total_ms, 2)
            lp_state = _find_series(states, "lpround")
            if lp_state is not None:
                lp_opt = lp_state.controller.last_lp_opt
                ep_cmp["lp_opt"] = (
                    round(lp_opt, 2)
                    if lp_opt is not None and refresh_epoch
                    else None
                )
            mip_state = _find_series(states, "milp")
            if mip_state is not None:
                mip_opt = mip_state.controller.last_milp_opt
                ep_cmp["mip_opt"] = (
                    round(mip_opt, 2)
                    if mip_opt is not None and refresh_epoch
                    else None
                )
                ep_cmp["mip_solve_status"] = (
                    mip_state.solve_meta.get("status") if refresh_epoch else None
                )

            ep_cmp.update(
                plan_eval.dashboard_fields(
                    labels=tuple(state.prefix for state in states)
                )
            )
            common_forward_keys = set.intersection(
                *(set(timing_by_name) for timing_by_name in forward_timings.values())
            )
            forward_steps = {
                key: sum(fwd[key] for fwd in forward_timings.values())
                for key in common_forward_keys
            }
            timing["forward_steps"] = round_ms(forward_steps)
            ep_cmp["timing"] = timing
            compare_data.append(ep_cmp)
            writer.save_data(
                series_data={state.name: state.data for state in states},
                latest_breakdown=latest_breakdown,
                latest_pop_compare=pop_cmp,
                epoch_compare=compare_data,
                cache_state=extract_cache_state(_reference_ground_knowledge(states)),
            )

            t_ep = time.perf_counter() - t_ep_start
            if epoch % 10 == 0 or epoch == config.num_epochs - 1:
                first = states[0].data[-1]
                means = " ".join(
                    f"{state.data[-1]['mean_rtt']:>6.1f}"
                    for state in states
                )
                p95s = " ".join(
                    f"{state.data[-1]['p95_rtt']:>6.1f}"
                    for state in states
                )
                print(
                    f"{epoch:>4} {first['time_str']:>8} "
                    f"{first['n_flows']:>6} {first['demand_gbps']:>5.0f}G  "
                    f"{means}  {p95s}  "
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
