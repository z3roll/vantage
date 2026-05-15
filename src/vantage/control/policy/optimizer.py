"""Path-aware optimizer controller.

This experimental controller bridges the standalone routing optimizer
into the normal simulation loop. It emits a regular
:class:`~vantage.control.plane.RoutingPlane`, plus path-level hints
that :class:`vantage.forward.PlannedRoutingPlaneForward` can execute
without running another capacity search.
"""

from __future__ import annotations

import argparse
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import (
    CellToPopTable,
    PlannedPath,
    PlannedPathTable,
    RoutingPlane,
)
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_demand_items,
    build_pop_egress_table,
    compute_cell_access,
    build_sat_path_table,
    compute_cell_ingress,
    compute_pop_capacity,
)
from vantage.model import CellGrid, NetworkSnapshot
from vantage.sim.config import SAT_FEEDER_CAP_GBPS

__all__ = ["PathAwareNearestBaselineController", "PathAwareOptimizerController"]


class PathAwareNearestBaselineController:
    """Standalone nearest-PoP baseline emitted as planned paths."""

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
        dest_names: tuple[str, ...] = (),
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()
        self._dest_names = dest_names
        self._last_timing: dict[str, float] = {}
        self._last_objective: float | None = None
        self._last_demand_mean: float | None = None

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> dict[str, float]:
        return self._last_timing

    @property
    def last_objective(self) -> float | None:
        return self._last_objective

    @property
    def last_demand_mean(self) -> float | None:
        return self._last_demand_mean

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        demand_per_pair: dict[tuple[str, str], float] | None = None,
        version: int = 0,
    ) -> RoutingPlane:
        perf = time.perf_counter
        t0 = perf()
        opt = _load_optimizer_module()
        problem = _build_optimizer_problem(
            opt,
            snapshot=snapshot,
            cell_grid=cell_grid,
            ground_knowledge=self._gk,
            demand_per_pair=demand_per_pair or {},
            dest_names=self._dest_names,
        )
        t_problem = perf()
        runner = opt.DijkstraRunner(problem, static_paths=True)
        dist, pred, dijkstra_seconds = runner.run(
            np.zeros(problem.edge_base_loss.shape[0], dtype=np.float64)
        )
        t_dijkstra = perf()
        result = opt.nearest_pop_baseline_solution(
            problem,
            pred,
            dist,
            deadline=t_dijkstra + 15.0,
            max_hops=128,
            path_variants=1,
        )
        t_solve = perf()

        sat_paths = build_sat_path_table(snapshot, version=version)
        pop_egress = build_pop_egress_table(snapshot, version=version)
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid,
            pops=snapshot.infra.pops,
            built_at=snapshot.time_s,
            version=version,
        )
        cell_to_pop, path_hints = _decode_optimizer_plane(
            problem,
            result,
            baseline=baseline,
            version=version,
            built_at=snapshot.time_s,
            fallback_predecessors=pred,
            max_hops=128,
        )
        t_assemble = perf()
        self._last_objective = float(result.objective)
        _flow_mean, demand_mean, _n, _demand = opt.latency_means(
            problem, result.assignment, result.trace.item_path_loss,
        )
        self._last_demand_mean = float(demand_mean)
        self._last_timing = {
            "problem_ms": (t_problem - t0) * 1000.0,
            "dijkstra_ms": dijkstra_seconds * 1000.0,
            "baseline_solve_ms": (t_solve - t_dijkstra) * 1000.0,
            "assemble_ms": (t_assemble - t_solve) * 1000.0,
            "objective": float(result.objective),
            "demand_mean_ms": float(demand_mean),
        }
        return RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_paths=sat_paths,
            pop_egress=pop_egress,
            version=version,
            built_at=snapshot.time_s,
            path_hints=path_hints,
        )


class PathAwareOptimizerController:
    """Joint PoP/GS/egress/ISL-path controller backed by the optimizer."""

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
        dest_names: tuple[str, ...] = (),
        *,
        main_loop_budget: float = 1.0,
        repair_budget: float = 6.0,
        local_search_budget: float = 0.0,
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()
        self._dest_names = dest_names
        self._main_loop_budget = float(main_loop_budget)
        self._repair_budget = float(repair_budget)
        self._local_search_budget = float(local_search_budget)
        self._last_timing: dict[str, float] = {}
        self._last_objective: float | None = None
        self._last_demand_mean: float | None = None

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> dict[str, float]:
        return self._last_timing

    @property
    def last_objective(self) -> float | None:
        return self._last_objective

    @property
    def last_demand_mean(self) -> float | None:
        return self._last_demand_mean

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        demand_per_pair: dict[tuple[str, str], float] | None = None,
        version: int = 0,
    ) -> RoutingPlane:
        perf = time.perf_counter
        t0 = perf()
        opt = _load_optimizer_module()
        problem = _build_optimizer_problem(
            opt,
            snapshot=snapshot,
            cell_grid=cell_grid,
            ground_knowledge=self._gk,
            demand_per_pair=demand_per_pair or {},
            dest_names=self._dest_names,
        )
        t_problem = perf()
        args = argparse.Namespace(
            main_loop_budget=self._main_loop_budget,
            repair_budget=self._repair_budget,
            local_search_budget=self._local_search_budget,
            max_iterations=200,
            max_hops=128,
            polyak_beta=1.8,
            dynamic_isl_prices=False,
            first_fit_top_k=256,
            isl_path_variants=1,
        )
        solve_start = perf()
        result = opt.solve(
            problem,
            args,
            global_deadline=solve_start + 15.0,
            program_start=solve_start,
        )
        t_solve = perf()

        sat_paths = build_sat_path_table(snapshot, version=version)
        pop_egress = build_pop_egress_table(snapshot, version=version)
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid,
            pops=snapshot.infra.pops,
            built_at=snapshot.time_s,
            version=version,
        )
        cell_to_pop, path_hints = _decode_optimizer_plane(
            problem,
            result,
            baseline=baseline,
            version=version,
            built_at=snapshot.time_s,
        )
        t_assemble = perf()

        self._last_objective = float(result.objective)
        _flow_mean, demand_mean, _n, _demand = opt.latency_means(
            problem, result.assignment, result.trace.item_path_loss,
        )
        self._last_demand_mean = float(demand_mean)
        self._last_timing = {
            "problem_ms": (t_problem - t0) * 1000.0,
            "solve_ms": (t_solve - solve_start) * 1000.0,
            "assemble_ms": (t_assemble - t_solve) * 1000.0,
            "objective": float(result.objective),
            "demand_mean_ms": float(demand_mean),
            "iterations": float(result.iterations),
        }
        return RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_paths=sat_paths,
            pop_egress=pop_egress,
            version=version,
            built_at=snapshot.time_s,
            path_hints=path_hints,
        )


def _load_optimizer_module() -> Any:
    """Import the standalone optimizer lazily.

    The bridge is intentionally local to the experimental controller so
    normal policy imports do not pay scipy/numba import costs unless
    ``--control optimizer`` is requested.
    """

    import test_routing_optimizer as opt  # type: ignore

    return opt


def _build_optimizer_problem(
    opt: Any,
    *,
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
    ground_knowledge: GroundKnowledge,
    demand_per_pair: dict[tuple[str, str], float],
    dest_names: tuple[str, ...] = (),
) -> Any:
    edges = tuple(snapshot.satellite.graph.edges)
    edge_src = np.asarray([edge.sat_a for edge in edges], dtype=np.int32)
    edge_dst = np.asarray([edge.sat_b for edge in edges], dtype=np.int32)
    edge_base_loss = np.asarray([edge.delay * 2.0 for edge in edges], dtype=np.float32)
    edge_cap = np.asarray([edge.capacity_gbps for edge in edges], dtype=np.float32)
    sparse_row = np.concatenate([edge_src, edge_dst]).astype(np.int32, copy=False)
    sparse_col = np.concatenate([edge_dst, edge_src]).astype(np.int32, copy=False)

    n_sats = snapshot.satellite.num_sats
    edge_id = np.full((n_sats, n_sats), -1, dtype=np.int32)
    adjacency_lists: list[list[tuple[int, int, float]]] = [
        [] for _ in range(n_sats)
    ]
    for idx, (a, b) in enumerate(zip(edge_src, edge_dst, strict=True)):
        ai = int(a)
        bi = int(b)
        edge_id[ai, bi] = idx
        edge_id[bi, ai] = idx
        loss = float(edge_base_loss[idx])
        adjacency_lists[ai].append((bi, idx, loss))
        adjacency_lists[bi].append((ai, idx, loss))
    adjacency = tuple(tuple(neighbors) for neighbors in adjacency_lists)

    pop_egress = build_pop_egress_table(snapshot)
    pop_cap_dict = compute_pop_capacity(snapshot)
    gs_by_id = {gs.gs_id: gs for gs in snapshot.infra.ground_stations}
    pop_names: list[str] = []
    pop_cap: list[float] = []
    pop_code_to_idx: dict[str, int] = {}
    gs_names: list[str] = []
    gs_cap: list[float] = []
    gs_id_to_idx: dict[str, int] = {}
    egress_nodes: list[int] = []
    egress_to_row: dict[int, int] = {}
    candidate_egress_row: list[int] = []
    candidate_node: list[int] = []
    candidate_pop_idx: list[int] = []
    candidate_gs_idx: list[int] = []
    candidate_sat: list[int] = []
    candidate_access_loss: list[float] = []
    candidate_label: list[str] = []

    for pop in snapshot.infra.pops:
        egress_ids, base_cost, gs_ids = pop_egress.for_pop(pop.code)
        cap = float(pop_cap_dict.get(pop.code, 0.0))
        if egress_ids.size == 0 or cap <= 0.0:
            continue
        pop_idx = len(pop_names)
        pop_code_to_idx[pop.code] = pop_idx
        pop_names.append(pop.code)
        pop_cap.append(cap)
        for raw_idx in np.argsort(base_cost, kind="stable"):
            idx = int(raw_idx)
            sat_id = int(egress_ids[idx])
            gs_id = gs_ids[idx]
            gs = gs_by_id.get(gs_id)
            if gs is None:
                continue
            gs_idx = gs_id_to_idx.get(gs_id)
            if gs_idx is None:
                gs_idx = len(gs_names)
                gs_id_to_idx[gs_id] = gs_idx
                gs_names.append(gs_id)
                gs_cap.append(float(gs.max_capacity))
            egress_row = egress_to_row.get(sat_id)
            if egress_row is None:
                egress_row = len(egress_nodes)
                egress_to_row[sat_id] = egress_row
                egress_nodes.append(sat_id)
            candidate_egress_row.append(egress_row)
            candidate_node.append(sat_id)
            candidate_pop_idx.append(pop_idx)
            candidate_gs_idx.append(gs_idx)
            candidate_sat.append(sat_id)
            candidate_access_loss.append(float(base_cost[idx]))
            candidate_label.append(f"{pop.code}/{gs_id}/sat{sat_id}")

    demand_items = build_demand_items(demand_per_pair, cell_grid)
    cell_ingress = compute_cell_ingress(snapshot, cell_grid)
    plan_dest_names = tuple(dest_names) or tuple(
        sorted({dest for _src, dest in demand_per_pair})
    )
    service_to_idx = {name: idx for idx, name in enumerate(plan_dest_names)}
    demand_by_cell_dest = {
        (cell_id, service): float(weight)
        for cell_id, service, weight in demand_items
    }
    seen_items = set(demand_by_cell_dest)
    active_cells = sorted(set(cell_grid.endpoint_to_cell.values()))
    if plan_dest_names:
        extended_items = list(demand_items)
        for cell_id in active_cells:
            for service in plan_dest_names:
                key = (cell_id, service)
                if key not in seen_items:
                    extended_items.append((cell_id, service, 0.0))
        demand_items = extended_items
    origins: list[int] = []
    weights: list[float] = []
    item_cell: list[int] = []
    item_service: list[str] = []
    item_service_idx: list[int] = []
    for cell_id, service, weight in demand_items:
        ingress = cell_ingress.get(cell_id)
        svc_idx = service_to_idx.get(service)
        if weight < 0.0 or ingress is None or svc_idx is None:
            continue
        origins.append(int(ingress))
        weights.append(float(weight))
        item_cell.append(int(cell_id))
        item_service.append(service)
        item_service_idx.append(svc_idx)
    if not origins:
        raise RuntimeError("No routable optimizer demand items were generated.")

    service_pop_ground = np.empty((len(plan_dest_names), len(pop_names)), dtype=np.float32)
    current_epoch = int(round(snapshot.time_s))
    for si, service in enumerate(plan_dest_names):
        for pi, pop_code in enumerate(pop_names):
            rtt = ground_knowledge.score(
                pop_code, service, current_epoch=current_epoch,
            )
            if rtt is None:
                rtt = ground_knowledge.get_or_estimate(pop_code, service)
            service_pop_ground[si, pi] = np.float32(rtt)
    item_service_idx_arr = np.asarray(item_service_idx, dtype=np.int32)
    ground_loss = service_pop_ground[item_service_idx_arr, :].astype(
        np.float32, copy=True,
    )

    nearest_table = build_cell_to_pop_nearest(
        cell_grid,
        snapshot.infra.pops,
        built_at=snapshot.time_s,
    )
    default_pop_order = tuple(range(len(pop_names)))
    nearest_pop_rows: list[list[int]] = []
    for cell_id in item_cell:
        ranked: list[int] = []
        seen: set[int] = set()
        for pop_code in nearest_table.mapping.get(cell_id, ()):
            pop_idx = pop_code_to_idx.get(pop_code)
            if pop_idx is None or pop_idx in seen:
                continue
            ranked.append(pop_idx)
            seen.add(pop_idx)
        for pop_idx in default_pop_order:
            if pop_idx not in seen:
                ranked.append(pop_idx)
        nearest_pop_rows.append(ranked)
    cell_access = {
        int(cell_id): tuple(
            (int(link.sat_id), float(link.delay * 2.0))
            for link in links
        )
        for cell_id, links in compute_cell_access(snapshot, cell_grid).items()
    }

    active_sat_ids = np.asarray(sorted(set(candidate_sat)), dtype=np.int32)
    return opt.ProblemData(
        n_sats=n_sats,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_base_loss=edge_base_loss,
        edge_cap=edge_cap,
        adjacency=adjacency,
        sparse_row=sparse_row,
        sparse_col=sparse_col,
        edge_id=edge_id,
        egress_nodes=np.asarray(egress_nodes, dtype=np.int32),
        candidate_egress_row=np.asarray(candidate_egress_row, dtype=np.int32),
        candidate_node=np.asarray(candidate_node, dtype=np.int32),
        candidate_pop_idx=np.asarray(candidate_pop_idx, dtype=np.int32),
        candidate_gs_idx=np.asarray(candidate_gs_idx, dtype=np.int32),
        candidate_sat=np.asarray(candidate_sat, dtype=np.int32),
        candidate_access_loss=np.asarray(candidate_access_loss, dtype=np.float32),
        candidate_label=tuple(candidate_label),
        pop_names=tuple(pop_names),
        pop_cap=np.asarray(pop_cap, dtype=np.float32),
        gs_names=tuple(gs_names),
        gs_cap=np.asarray(gs_cap, dtype=np.float32),
        active_sat_ids=active_sat_ids,
        sat_cap=np.full(
            active_sat_ids.shape, SAT_FEEDER_CAP_GBPS, dtype=np.float32,
        ),
        origins=np.asarray(origins, dtype=np.int32),
        cell_access=cell_access,
        weights=np.asarray(weights, dtype=np.float32),
        ground_loss=np.ascontiguousarray(ground_loss, dtype=np.float32),
        nearest_pop_order=np.asarray(nearest_pop_rows, dtype=np.int32),
        item_cell=np.asarray(item_cell, dtype=np.int64),
        item_service=tuple(item_service),
        n_cells_total=len(set(cell_grid.endpoint_to_cell.values())),
        service_names=tuple(plan_dest_names),
        raw_demand_gbps=float(sum(weight for _, _, weight in demand_items)),
        dropped_no_ingress_items=0,
        dropped_no_ingress_gbps=0.0,
        setup_seconds=0.0,
    )


def _decode_optimizer_plane(
    problem: Any,
    result: Any,
    *,
    baseline: CellToPopTable,
    version: int,
    built_at: float,
    fallback_predecessors: np.ndarray | None = None,
    max_hops: int = 128,
) -> tuple[CellToPopTable, PlannedPathTable]:
    per_dest: dict[tuple[int, str], tuple[str, ...]] = {}
    paths: dict[tuple[int, int, str], tuple[PlannedPath, ...]] = {}
    all_pops = tuple(problem.pop_names)
    valid_items = np.flatnonzero(result.assignment >= 0)
    for item_idx_raw in valid_items:
        item_idx = int(item_idx_raw)
        candidate = int(result.assignment[item_idx])
        pop_code = problem.pop_names[int(problem.candidate_pop_idx[candidate])]
        gs_id = problem.gs_names[int(problem.candidate_gs_idx[candidate])]
        egress_sat = int(problem.candidate_sat[candidate])
        cell_id = int(problem.item_cell[item_idx])
        service = problem.item_service[item_idx]
        ingress = int(result.trace.item_origin[item_idx])
        path_count = int(result.trace.path_counts[item_idx])
        if path_count < 0:
            continue
        edge_ids = result.trace.path_edges[item_idx, :path_count]
        isl_links = tuple(
            _edge_key(
                int(problem.edge_src[int(edge_id)]),
                int(problem.edge_dst[int(edge_id)]),
            )
            for edge_id in edge_ids
        )
        planned = PlannedPath(
            pop_code=pop_code,
            gs_id=gs_id,
            egress_sat=egress_sat,
            isl_links=isl_links,
            access_rtt=float(problem.candidate_access_loss[candidate]),
            expected_rtt=(
                float(result.trace.item_path_loss[item_idx])
                + float(problem.candidate_access_loss[candidate])
                + float(problem.ground_loss[item_idx, int(problem.candidate_pop_idx[candidate])])
            ),
        )
        paths[(ingress, cell_id, service)] = (planned,)
        fallback_origin = int(problem.origins[item_idx])
        if (
            fallback_predecessors is not None
            and fallback_origin != ingress
        ):
            fallback_edges = _trace_candidate_edges(
                problem,
                fallback_predecessors,
                fallback_origin,
                candidate,
                max_hops=max_hops,
            )
            if fallback_edges is not None:
                fallback_loss = sum(
                    float(problem.edge_base_loss[int(edge_id)])
                    for edge_id in fallback_edges
                )
                fallback_links = tuple(
                    _edge_key(
                        int(problem.edge_src[int(edge_id)]),
                        int(problem.edge_dst[int(edge_id)]),
                    )
                    for edge_id in fallback_edges
                )
                fallback_planned = PlannedPath(
                    pop_code=pop_code,
                    gs_id=gs_id,
                    egress_sat=egress_sat,
                    isl_links=fallback_links,
                    access_rtt=float(problem.candidate_access_loss[candidate]),
                    expected_rtt=(
                        fallback_loss
                        + float(problem.candidate_access_loss[candidate])
                        + float(problem.ground_loss[
                            item_idx,
                            int(problem.candidate_pop_idx[candidate]),
                        ])
                    ),
                )
                paths[(fallback_origin, cell_id, service)] = (fallback_planned,)
        tail = tuple(pop for pop in all_pops if pop != pop_code)
        per_dest[(cell_id, service)] = (pop_code,) + tail

    cell_to_pop = CellToPopTable(
        mapping=baseline.mapping,
        version=version,
        built_at=built_at,
        per_dest=MappingProxyType(per_dest),
    )
    path_hints = PlannedPathTable(
        paths=MappingProxyType(paths),
        version=version,
        built_at=built_at,
    )
    return cell_to_pop, path_hints


def _trace_candidate_edges(
    problem: Any,
    predecessors: np.ndarray,
    origin: int,
    candidate: int,
    *,
    max_hops: int,
) -> tuple[int, ...] | None:
    cur = int(origin)
    row = int(problem.candidate_egress_row[candidate])
    dst_node = int(problem.candidate_node[candidate])
    out: list[int] = []
    hops = 0
    while cur != dst_node:
        if hops >= max_hops:
            return None
        nxt = int(predecessors[row, cur])
        if nxt < 0:
            return None
        edge_id = int(problem.edge_id[cur, nxt])
        if edge_id < 0:
            return None
        out.append(edge_id)
        cur = nxt
        hops += 1
    return tuple(out)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)
