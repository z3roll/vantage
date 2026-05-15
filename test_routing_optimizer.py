#!/usr/bin/env python3
"""多商品卫星路由优化器实验脚本。

直接运行：

    python test_routing_optimizer.py

如果当前 Python 环境没有安装项目依赖，推荐在仓库根目录运行：

    uv run python test_routing_optimizer.py

可选加速依赖：

    pip install numba joblib

没有 numba 时脚本会自动降级到纯 Python 路径追踪，结果语义不变，
但 15 秒预算内的迭代轮数会明显减少。
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from numba import jit as _numba_jit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    _numba_jit = None

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False
    Parallel = None
    delayed = None

from vantage.control.policy.common.fib_builder import (  # noqa: E402
    build_cell_to_pop_nearest,
    build_demand_items,
    build_pop_egress_table,
    compute_cell_access,
    compute_cell_ingress,
    compute_pop_capacity,
)
from vantage.sim.build import build_runtime  # noqa: E402
from vantage.sim.config import EPOCH_S, SeedBundle, SimConfig  # noqa: E402


TOTAL_TIME_LIMIT = 15.0
MAIN_LOOP_BUDGET = 1.0
REPAIR_BUDGET = 6.0
LOCAL_SEARCH_BUDGET = 0.0
INF = np.float32(1.0e30)
EPS = 1.0e-6


@dataclass(slots=True)
class ProblemData:
    """优化器的紧凑数组输入。

    复杂度：
    - 图存储 O(|V| + |E|)
    - commodity × destination 损耗矩阵 O(|K|·|D|)
    """

    n_sats: int
    edge_src: NDArray[np.int32]
    edge_dst: NDArray[np.int32]
    edge_base_loss: NDArray[np.float32]
    edge_cap: NDArray[np.float32]
    adjacency: tuple[tuple[tuple[int, int, float], ...], ...]
    sparse_row: NDArray[np.int32]
    sparse_col: NDArray[np.int32]
    edge_id: NDArray[np.int32]
    egress_nodes: NDArray[np.int32]
    candidate_egress_row: NDArray[np.int32]
    candidate_node: NDArray[np.int32]
    candidate_pop_idx: NDArray[np.int32]
    candidate_gs_idx: NDArray[np.int32]
    candidate_sat: NDArray[np.int32]
    candidate_access_loss: NDArray[np.float32]
    candidate_label: tuple[str, ...]
    pop_names: tuple[str, ...]
    pop_cap: NDArray[np.float32]
    gs_names: tuple[str, ...]
    gs_cap: NDArray[np.float32]
    active_sat_ids: NDArray[np.int32]
    sat_cap: NDArray[np.float32]
    origins: NDArray[np.int32]
    cell_access: dict[int, tuple[tuple[int, float], ...]]
    weights: NDArray[np.float32]
    ground_loss: NDArray[np.float32]
    nearest_pop_order: NDArray[np.int32]
    item_cell: NDArray[np.int64]
    item_service: tuple[str, ...]
    n_cells_total: int
    service_names: tuple[str, ...]
    raw_demand_gbps: float
    dropped_no_ingress_items: int
    dropped_no_ingress_gbps: float
    setup_seconds: float


@dataclass(slots=True)
class TraceResult:
    """一次 assignment 的路径追踪结果。"""

    edge_load: NDArray[np.float64]
    item_path_loss: NDArray[np.float32]
    path_edges: NDArray[np.int32]
    path_counts: NDArray[np.int32]
    item_origin: NDArray[np.int32]
    invalid_count: int


@dataclass(slots=True)
class OptimizerResult:
    """求解器结果和性能计数。"""

    assignment: NDArray[np.int32]
    trace: TraceResult
    pop_load: NDArray[np.float64]
    sat_load: NDArray[np.float64]
    gs_load: NDArray[np.float64]
    objective: float
    best_lower_bound: float
    iterations: int
    total_seconds: float
    solve_seconds: float
    main_seconds: float
    repair_seconds: float
    local_search_seconds: float
    dijkstra_seconds: float
    trace_seconds: float
    feasible_seen: bool


@dataclass(slots=True)
class CandidateMipData:
    """受限候选集的 0-1 MIP 输入。"""

    top_k: int
    c: NDArray[np.float64]
    constraints: LinearConstraint
    var_item: NDArray[np.int32]
    var_candidate: NDArray[np.int32]
    build_seconds: float
    skipped_items: int
    forced_incumbent_vars: int
    n_rows: int
    n_resource_rows: int
    nnz: int


@dataclass(slots=True)
class OptimalCertificateResult:
    """LP/MILP certificate 结果。"""

    top_k: int
    build_seconds: float
    lp_seconds: float
    mip_seconds: float
    lp_fun: float | None
    lp_success: bool
    lp_status: int
    lp_message: str
    mip_fun: float | None
    mip_dual_bound: float | None
    mip_gap: float | None
    mip_success: bool
    mip_status: int | None
    mip_message: str | None
    mip_nodes: int | None
    skipped_items: int
    forced_incumbent_vars: int
    n_vars: int
    n_rows: int
    nnz: int
    violations: tuple[int, int, int, int] | None
    invalid_count: int | None


@dataclass(slots=True)
class ExtraColumn:
    """列生成加入的路径变量。"""

    item_idx: int
    candidate_idx: int
    path_edges: tuple[int, ...]
    path_loss: float


@dataclass(slots=True)
class ColumnGenerationResult:
    """列生成/定价式 LP certificate 结果。"""

    iterations: int
    converged: bool
    lp_fun: float
    corrected_lower_bound: float
    best_reduced_cost: float
    incumbent_objective: float
    gap_percent: float
    final_columns: int
    extra_columns: int
    added_columns: int
    build_seconds: float
    lp_seconds: float
    pricing_seconds: float
    total_seconds: float
    last_lp_success: bool
    last_lp_status: int
    last_lp_message: str


def _jit_or_plain(func: Any) -> Any:
    """numba 存在时 JIT 编译；否则返回原函数作为降级实现。"""

    if NUMBA_AVAILABLE and _numba_jit is not None:
        return _numba_jit(nopython=True, cache=True)(func)
    return func


@_jit_or_plain
def _trace_loads_kernel(
    predecessors: NDArray[np.int32],
    egress_nodes: NDArray[np.int32],
    candidate_egress_row: NDArray[np.int32],
    candidate_node: NDArray[np.int32],
    origins: NDArray[np.int32],
    assignment: NDArray[np.int32],
    weights: NDArray[np.float32],
    edge_id: NDArray[np.int32],
    edge_base_loss: NDArray[np.float32],
    max_hops: int,
) -> tuple[NDArray[np.float64], NDArray[np.float32], int]:
    """轻量路径追踪：只统计 ISL 负载和路径损耗，不保存每条路径。

    复杂度 O(sum path_hops)。主循环用这个版本，避免每轮写
    ``n_items × max_hops`` 的 path_edges 大矩阵。
    """

    n_items = origins.shape[0]
    n_edges = edge_base_loss.shape[0]
    edge_load = np.zeros(n_edges, dtype=np.float64)
    item_path_loss = np.empty(n_items, dtype=np.float32)
    invalid_count = 0

    for i in range(n_items):
        item_path_loss[i] = INF
        c = assignment[i]
        if c < 0:
            invalid_count += 1
            continue

        row = candidate_egress_row[c]
        cur = origins[i]
        dst_node = candidate_node[c]
        total_loss = np.float32(0.0)
        hop = 0
        valid = True

        while cur != dst_node:
            if hop >= max_hops:
                valid = False
                break
            nxt = predecessors[row, cur]
            if nxt < 0:
                valid = False
                break
            eid = edge_id[cur, nxt]
            if eid < 0:
                valid = False
                break
            edge_load[eid] += weights[i]
            total_loss += edge_base_loss[eid]
            cur = nxt
            hop += 1

        if valid:
            item_path_loss[i] = total_loss
        else:
            invalid_count += 1

    return edge_load, item_path_loss, invalid_count


@_jit_or_plain
def _trace_paths_kernel(
    predecessors: NDArray[np.int32],
    egress_nodes: NDArray[np.int32],
    candidate_egress_row: NDArray[np.int32],
    candidate_node: NDArray[np.int32],
    origins: NDArray[np.int32],
    assignment: NDArray[np.int32],
    weights: NDArray[np.float32],
    edge_id: NDArray[np.int32],
    edge_base_loss: NDArray[np.float32],
    max_hops: int,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int32], NDArray[np.int32], int]:
    """追踪每个 commodity 的 ISL 路径并统计边负载。

    复杂度 O(sum path_hops)。这是主循环里除 Dijkstra 外最热的步骤。
    """

    n_items = origins.shape[0]
    n_edges = edge_base_loss.shape[0]
    edge_load = np.zeros(n_edges, dtype=np.float64)
    item_path_loss = np.empty(n_items, dtype=np.float32)
    path_edges = np.empty((n_items, max_hops), dtype=np.int32)
    path_counts = np.empty(n_items, dtype=np.int32)
    invalid_count = 0

    for i in range(n_items):
        for h in range(max_hops):
            path_edges[i, h] = -1
        item_path_loss[i] = INF
        path_counts[i] = -1

        d = assignment[i]
        if d < 0:
            invalid_count += 1
            continue

        row = candidate_egress_row[d]
        cur = origins[i]
        dst_node = candidate_node[d]
        total_loss = np.float32(0.0)
        hop = 0
        valid = True

        while cur != dst_node:
            if hop >= max_hops:
                valid = False
                break
            nxt = predecessors[row, cur]
            if nxt < 0:
                valid = False
                break
            eid = edge_id[cur, nxt]
            if eid < 0:
                valid = False
                break
            path_edges[i, hop] = eid
            edge_load[eid] += weights[i]
            total_loss += edge_base_loss[eid]
            cur = nxt
            hop += 1

        if valid:
            item_path_loss[i] = total_loss
            path_counts[i] = hop
        else:
            invalid_count += 1

    return edge_load, item_path_loss, path_edges, path_counts, invalid_count


def trace_paths(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    assignment: NDArray[np.int32],
    *,
    max_hops: int,
) -> TraceResult:
    """调用 JIT/fallback 路径追踪内核。"""

    edge_load, item_path_loss, path_edges, path_counts, invalid_count = _trace_paths_kernel(
        predecessors,
        problem.egress_nodes,
        problem.candidate_egress_row,
        problem.candidate_node,
        problem.origins,
        assignment,
        problem.weights,
        problem.edge_id,
        problem.edge_base_loss,
        int(max_hops),
    )
    return TraceResult(
        edge_load=edge_load,
        item_path_loss=item_path_loss,
        path_edges=path_edges,
        path_counts=path_counts,
        item_origin=problem.origins.copy(),
        invalid_count=int(invalid_count),
    )


def trace_loads(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    assignment: NDArray[np.int32],
    *,
    max_hops: int,
) -> TraceResult:
    """主循环轻量追踪，只保留负载和每个 item 的路径损耗。"""

    edge_load, item_path_loss, invalid_count = _trace_loads_kernel(
        predecessors,
        problem.egress_nodes,
        problem.candidate_egress_row,
        problem.candidate_node,
        problem.origins,
        assignment,
        problem.weights,
        problem.edge_id,
        problem.edge_base_loss,
        int(max_hops),
    )
    return TraceResult(
        edge_load=edge_load,
        item_path_loss=item_path_loss,
        path_edges=np.empty((0, 0), dtype=np.int32),
        path_counts=np.empty(0, dtype=np.int32),
        item_origin=problem.origins.copy(),
        invalid_count=int(invalid_count),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lagrangian relaxation optimizer for Argus satellite routing."
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--user-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-gs-per-pop", type=int, default=3)
    parser.add_argument("--egress-top-k", type=int, default=8, choices=range(1, 9))
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10_000,
        help="Hard cap; wall-clock budgets normally stop first.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=192,
        help="Maximum ISL hops recorded per commodity path.",
    )
    parser.add_argument(
        "--polyak-beta",
        type=float,
        default=1.0,
        help="Multiplier on the standard Polyak step size.",
    )
    parser.add_argument(
        "--main-loop-budget",
        type=float,
        default=MAIN_LOOP_BUDGET,
        help="Seconds spent on Lagrangian price updates before repair.",
    )
    parser.add_argument(
        "--repair-budget",
        type=float,
        default=REPAIR_BUDGET,
        help="Seconds reserved for forward-style feasibility repair.",
    )
    parser.add_argument(
        "--local-search-budget",
        type=float,
        default=LOCAL_SEARCH_BUDGET,
        help="Seconds spent on optional final local search.",
    )
    parser.add_argument(
        "--first-fit-top-k",
        type=int,
        default=16,
        help="Candidate window per commodity before full fallback in first-fit.",
    )
    parser.add_argument(
        "--isl-path-variants",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help=(
            "Number of simple shortest ISL paths to lazily try per "
            "(ingress, egress) during feasible placement."
        ),
    )
    parser.add_argument(
        "--dynamic-isl-prices",
        action="store_true",
        help=(
            "Re-run Dijkstra with ISL Lagrange prices every iteration. "
            "Default is faster: reuse base shortest paths and let final "
            "forward-style first-fit enforce ISL capacity."
        ),
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help=(
            "Build a top-k candidate MIP certificate instead of only "
            "running the heuristic optimizer."
        ),
    )
    parser.add_argument(
        "--optimal-top-k",
        type=int,
        default=8,
        help="Per-commodity candidates retained for the MIP certificate.",
    )
    parser.add_argument(
        "--optimal-time-limit",
        type=float,
        default=15.0,
        help="HiGHS MILP time limit in seconds for --find-optimal.",
    )
    parser.add_argument(
        "--optimal-lp-only",
        action="store_true",
        help="Only solve the LP relaxation for --find-optimal.",
    )
    parser.add_argument(
        "--column-generation",
        action="store_true",
        help="Run a path-pricing column-generation LP certificate.",
    )
    parser.add_argument(
        "--cg-initial-top-k",
        type=int,
        default=16,
        help="Initial per-commodity base shortest-path candidates.",
    )
    parser.add_argument(
        "--cg-max-iterations",
        type=int,
        default=5,
        help="Maximum column-generation pricing rounds.",
    )
    parser.add_argument(
        "--cg-max-new-columns",
        type=int,
        default=20_000,
        help="Maximum negative reduced-cost columns added per round.",
    )
    parser.add_argument(
        "--cg-time-limit",
        type=float,
        default=60.0,
        help="Wall-clock budget for column generation after data build.",
    )
    parser.add_argument(
        "--cg-pricing-tol",
        type=float,
        default=1.0e-6,
        help="Reduced-cost tolerance for adding priced columns.",
    )
    return parser.parse_args()


def build_problem(args: argparse.Namespace) -> ProblemData:
    """复用项目 runtime 构建真实星座、真实 traffic 和真实地面目的地。

    复杂度主要来自项目已有的星座预计算和一次 traffic 生成。
    """

    setup_start = time.perf_counter()
    config = SimConfig(
        num_epochs=1,
        user_scale=args.user_scale,
        control_algorithms=("baseline",),
        egress_top_k=args.egress_top_k,
        port=8000,
        no_browser=True,
        no_serve=True,
        max_gs_per_pop=args.max_gs_per_pop,
        seeds=SeedBundle.from_cli_seed(args.seed),
        enforce_isl_capacity=True,
    )
    runtime = build_runtime(config)
    snapshot = runtime.world.snapshot_at(args.epoch, args.epoch * EPOCH_S)
    demand = runtime.traffic.generate(args.epoch)
    demand_per_pair = {
        (flow_key.src, flow_key.dst): flow_demand
        for flow_key, flow_demand in demand.flows.items()
    }
    demand_items = build_demand_items(demand_per_pair, runtime.cell_grid)
    cell_ingress = compute_cell_ingress(snapshot, runtime.cell_grid)

    # ---- ISL 图：节点=卫星，边=ISLEdge，容量直接复用 edge.capacity_gbps ----
    edges = tuple(snapshot.satellite.graph.edges)
    edge_src = np.asarray([edge.sat_a for edge in edges], dtype=np.int32)
    edge_dst = np.asarray([edge.sat_b for edge in edges], dtype=np.int32)
    # 项目的 delay 是 one-way ms；forward RTT 中 ISL 会乘 2，这里用 RTT 损耗。
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
        edge_id[int(a), int(b)] = idx
        edge_id[int(b), int(a)] = idx
        loss = float(edge_base_loss[idx])
        adjacency_lists[int(a)].append((int(b), idx, loss))
        adjacency_lists[int(b)].append((int(a), idx, loss))
    adjacency = tuple(tuple(neighbors) for neighbors in adjacency_lists)

    # ---- 候选目的地：每个 PoP 下保留项目已有 top-k GS→egress sat 候选。 ----
    #
    # 这里的一个 candidate 对应 forward 里的一个可执行下行出口：
    #     (PoP, GS, egress_sat)
    # ``build_pop_egress_table`` 已经按 CLI ``--egress-top-k`` 截断
    # 每个 GS 的最高仰角卫星，因此默认就是每个 GS 8 个 egress sat。
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
        egress_ids, base_cost, _gs_ids = pop_egress.for_pop(pop.code)
        cap = float(pop_cap_dict.get(pop.code, 0.0))
        if egress_ids.size == 0 or cap <= 0.0:
            continue
        pop_idx = len(pop_names)
        pop_code_to_idx[pop.code] = pop_idx
        pop_names.append(pop.code)
        pop_cap.append(cap)

        order = np.argsort(base_cost, kind="stable")
        for raw_idx in order:
            sat_id = int(egress_ids[int(raw_idx)])
            gs_id = _gs_ids[int(raw_idx)]
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
            # base_cost 已经是 2*downlink + 2*GS-PoP backhaul。
            candidate_access_loss.append(float(base_cost[int(raw_idx)]))
            candidate_label.append(f"{pop.code}/{gs_id}/sat{sat_id}")

    if not candidate_label:
        raise RuntimeError("No PoP destinations with visible egress satellites.")

    active_sat_ids = np.asarray(sorted(set(candidate_sat)), dtype=np.int32)
    sat_cap = np.full(
        active_sat_ids.shape,
        float(runtime.world.shell.feeder_capacity_gbps),
        dtype=np.float32,
    )

    service_names = tuple(runtime.svc_names)
    service_to_idx = {name: i for i, name in enumerate(service_names)}

    # service×PoP 的随机地面 RTT。中位数来自项目的 GeographicGroundDelay，
    # 抖动使用 GroundTruth 的 sigma，但用 numpy RNG 一次性生成，避免
    # 对每个 (commodity, PoP) 调 GroundTruth.sample 的 Python/RNG 开销。
    service_pop_median = np.empty((len(service_names), len(pop_names)), dtype=np.float32)
    for si, service in enumerate(service_names):
        for pi, pop_code in enumerate(pop_names):
            service_pop_median[si, pi] = np.float32(
                runtime.geo_delay.estimate(pop_code, service) * 2.0
            )

    raw_demand_gbps = float(sum(weight for _, _, weight in demand_items))
    dropped_no_ingress_items = 0
    dropped_no_ingress_gbps = 0.0

    origins: list[int] = []
    weights: list[float] = []
    item_cell: list[int] = []
    item_service: list[str] = []
    item_service_idx: list[int] = []
    for cell_id, service, weight in demand_items:
        ingress = cell_ingress.get(cell_id)
        svc_idx = service_to_idx.get(service)
        if weight <= 0.0 or svc_idx is None:
            continue
        if ingress is None:
            dropped_no_ingress_items += 1
            dropped_no_ingress_gbps += float(weight)
            continue
        origins.append(int(ingress))
        weights.append(float(weight))
        item_cell.append(int(cell_id))
        item_service.append(service)
        item_service_idx.append(svc_idx)

    if not origins:
        raise RuntimeError("No routable (cell, service) commodities were generated.")

    item_service_idx_arr = np.asarray(item_service_idx, dtype=np.int32)
    median_ground = service_pop_median[item_service_idx_arr, :].astype(np.float32, copy=True)
    rng = np.random.default_rng(config.seeds.ground_seed + args.epoch * 1_000_003)
    noise = rng.normal(
        loc=0.0,
        scale=runtime.ground_truth.sigma,
        size=median_ground.shape,
    ).astype(np.float32)
    ground_loss = np.exp(np.log(np.maximum(median_ground, 1.0e-6)) + noise).astype(np.float32)
    nearest_table = build_cell_to_pop_nearest(
        runtime.cell_grid,
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
        if len(ranked) < len(pop_names):
            for pop_idx in default_pop_order:
                if pop_idx not in seen:
                    ranked.append(pop_idx)
        nearest_pop_rows.append(ranked)
    cell_access = {
        int(cell_id): tuple(
            (int(link.sat_id), float(link.delay * 2.0))
            for link in links
        )
        for cell_id, links in compute_cell_access(snapshot, runtime.cell_grid).items()
    }

    return ProblemData(
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
        sat_cap=sat_cap,
        origins=np.asarray(origins, dtype=np.int32),
        cell_access=cell_access,
        weights=np.asarray(weights, dtype=np.float32),
        ground_loss=np.ascontiguousarray(ground_loss, dtype=np.float32),
        nearest_pop_order=np.asarray(nearest_pop_rows, dtype=np.int32),
        item_cell=np.asarray(item_cell, dtype=np.int64),
        item_service=tuple(item_service),
        n_cells_total=len(set(runtime.cell_grid.endpoint_to_cell.values())),
        service_names=service_names,
        raw_demand_gbps=raw_demand_gbps,
        dropped_no_ingress_items=dropped_no_ingress_items,
        dropped_no_ingress_gbps=dropped_no_ingress_gbps,
        setup_seconds=time.perf_counter() - setup_start,
    )


class DijkstraRunner:
    """反向 Dijkstra 执行器。

    默认用 scipy sparse csgraph.dijkstra 的 multi-source indices；
    如果首轮超过 20ms 且安装了 joblib，则后续可切换到线程并行。
    """

    def __init__(self, problem: ProblemData, *, static_paths: bool) -> None:
        self.problem = problem
        self.static_paths = static_paths
        self._cached: tuple[NDArray[np.float32], NDArray[np.int32]] | None = None
        self.use_parallel = False
        self._checked_parallel = False

    def run(self, sigma: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.int32], float]:
        if self.static_paths and self._cached is not None:
            dist, pred = self._cached
            return dist, pred, 0.0
        t0 = time.perf_counter()
        if self.static_paths:
            weights = self.problem.edge_base_loss.astype(np.float64)
        else:
            weights = self.problem.edge_base_loss.astype(np.float64) + sigma
        data = np.concatenate([weights, weights])
        graph = csr_matrix(
            (data, (self.problem.sparse_row, self.problem.sparse_col)),
            shape=(self.problem.n_sats, self.problem.n_sats),
        )

        if self.use_parallel and JOBLIB_AVAILABLE and Parallel is not None and delayed is not None:
            results = Parallel(
                n_jobs=min(os.cpu_count() or 1, len(self.problem.egress_nodes)),
                prefer="threads",
            )(
                delayed(_dijkstra_one)(graph, int(node))
                for node in self.problem.egress_nodes
            )
            dist = np.vstack([r[0] for r in results])
            pred = np.vstack([r[1] for r in results])
        else:
            dist, pred = dijkstra(
                graph,
                directed=False,
                indices=self.problem.egress_nodes,
                return_predecessors=True,
            )

        elapsed = time.perf_counter() - t0
        if (
            not self._checked_parallel
            and elapsed > 0.020
            and JOBLIB_AVAILABLE
            and len(self.problem.egress_nodes) > 1
        ):
            self.use_parallel = True
        self._checked_parallel = True

        pred = np.asarray(pred, dtype=np.int32)
        pred[pred == -9999] = -1
        dist = np.asarray(dist, dtype=np.float32)
        if self.static_paths:
            self._cached = (dist, pred)
        return dist, pred, elapsed


def _dijkstra_one(graph: csr_matrix, node: int) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    dist, pred = dijkstra(
        graph,
        directed=False,
        indices=np.asarray([node], dtype=np.int32),
        return_predecessors=True,
    )
    pred = np.asarray(pred, dtype=np.int32)
    pred[pred == -9999] = -1
    return dist, pred


def assign_commodities(
    problem: ProblemData,
    dist_matrix: NDArray[np.float32],
    pi_pop: NDArray[np.float64],
    rho_sat: NDArray[np.float64],
    gamma_gs: NDArray[np.float64],
    base_cost: NDArray[np.float32] | None = None,
) -> NDArray[np.int32]:
    """向量化分配 commodity 到候选 ``(PoP, GS, egress_sat)``。

    复杂度 O(|K|·|C|)，没有对 commodity 的 Python 循环。
    """

    if base_cost is None:
        base_cost = candidate_base_costs(problem, dist_matrix)
    penalty = (
        pi_pop[problem.candidate_pop_idx].astype(np.float32)
        + rho_sat[problem.candidate_sat].astype(np.float32)
        + gamma_gs[problem.candidate_gs_idx].astype(np.float32)
    )
    total = base_cost + penalty[None, :]
    assignment = np.argmin(total, axis=1).astype(np.int32)
    best = np.min(total, axis=1)
    assignment[~np.isfinite(best)] = -1
    return assignment


def candidate_base_costs(
    problem: ProblemData,
    dist_matrix: NDArray[np.float32],
) -> NDArray[np.float32]:
    """构建 ``commodity × candidate`` 的无惩罚端到端成本矩阵。

    静态 ISL 路径时这个矩阵在所有次梯度迭代中不变，缓存后可以
    复用拓扑相似性，避免每轮重复 materialize 大矩阵。
    """

    path_by_egress = dist_matrix[:, problem.origins].T
    return (
        path_by_egress[:, problem.candidate_egress_row]
        + problem.candidate_access_loss[None, :]
        + problem.ground_loss[:, problem.candidate_pop_idx]
    ).astype(np.float32, copy=False)


def assignment_path_loss(
    problem: ProblemData,
    dist_matrix: NDArray[np.float32],
    assignment: NDArray[np.int32],
) -> NDArray[np.float32]:
    """从 Dijkstra dist 直接读取每个已分配 commodity 的 ISL 路径损耗。"""

    item_path_loss = np.full(assignment.shape[0], INF, dtype=np.float32)
    valid = assignment >= 0
    if np.any(valid):
        chosen = assignment[valid]
        item_path_loss[valid] = dist_matrix[
            problem.candidate_egress_row[chosen],
            problem.origins[valid],
        ]
    return item_path_loss


def resource_loads(
    problem: ProblemData,
    assignment: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """统计 PoP / sat feeder / GS feeder 负载。复杂度 O(|K|)。"""

    valid = assignment >= 0
    if not np.any(valid):
        return (
            np.zeros(len(problem.pop_names), dtype=np.float64),
            np.zeros(problem.n_sats, dtype=np.float64),
            np.zeros(len(problem.gs_names), dtype=np.float64),
        )
    chosen = assignment[valid]
    weights = problem.weights[valid].astype(np.float64)
    pop_load = np.bincount(
        problem.candidate_pop_idx[chosen],
        weights=weights,
        minlength=len(problem.pop_names),
    ).astype(np.float64)
    sat_load = np.bincount(
        problem.candidate_sat[chosen],
        weights=weights,
        minlength=problem.n_sats,
    ).astype(np.float64)
    gs_load = np.bincount(
        problem.candidate_gs_idx[chosen],
        weights=weights,
        minlength=len(problem.gs_names),
    ).astype(np.float64)
    return pop_load, sat_load, gs_load


def actual_objective(
    problem: ProblemData,
    assignment: NDArray[np.int32],
    item_path_loss: NDArray[np.float32],
) -> float:
    """计算当前整数解真实损耗。复杂度 O(|K|)。"""

    valid = (assignment >= 0) & np.isfinite(item_path_loss)
    if not np.any(valid):
        return math.inf
    idx = np.arange(problem.weights.shape[0])[valid]
    chosen = assignment[valid]
    per_item = (
        item_path_loss[valid].astype(np.float64)
        + problem.candidate_access_loss[chosen].astype(np.float64)
        + problem.ground_loss[idx, problem.candidate_pop_idx[chosen]].astype(np.float64)
    )
    return float(np.dot(problem.weights[valid].astype(np.float64), per_item))


def latency_means(
    problem: ProblemData,
    assignment: NDArray[np.int32],
    item_path_loss: NDArray[np.float32],
) -> tuple[float, float, int, float]:
    """返回 E2E RTT 的 flow-count 平均和 demand 加权平均。"""

    valid = (assignment >= 0) & np.isfinite(item_path_loss)
    if not np.any(valid):
        return math.nan, math.nan, 0, 0.0
    idx = np.arange(problem.weights.shape[0])[valid]
    chosen = assignment[valid]
    per_item = (
        item_path_loss[valid].astype(np.float64)
        + problem.candidate_access_loss[chosen].astype(np.float64)
        + problem.ground_loss[idx, problem.candidate_pop_idx[chosen]].astype(np.float64)
    )
    weights = problem.weights[valid].astype(np.float64)
    demand_total = float(weights.sum())
    demand_mean = (
        float(np.dot(weights, per_item) / demand_total)
        if demand_total > 0.0
        else math.nan
    )
    return float(np.mean(per_item)), demand_mean, int(per_item.size), demand_total


def lagrangian_lower_bound(
    problem: ProblemData,
    dist_matrix: NDArray[np.float32],
    sigma: NDArray[np.float64],
    pi_pop: NDArray[np.float64],
    rho_sat: NDArray[np.float64],
    gamma_gs: NDArray[np.float64],
    base_cost: NDArray[np.float32] | None = None,
) -> float:
    """计算拉格朗日对偶下界。复杂度 O(|K|·|C|)。"""

    if base_cost is None:
        base_cost = candidate_base_costs(problem, dist_matrix)
    penalty = (
        pi_pop[problem.candidate_pop_idx]
        + rho_sat[problem.candidate_sat]
        + gamma_gs[problem.candidate_gs_idx]
    )
    total = base_cost.astype(np.float64) + penalty[None, :]
    best = np.min(total, axis=1)
    finite = np.isfinite(best)
    if not np.any(finite):
        return -math.inf
    lower = float(np.dot(problem.weights[finite].astype(np.float64), best[finite]))
    lower -= float(np.dot(sigma, problem.edge_cap.astype(np.float64)))
    lower -= float(np.dot(pi_pop, problem.pop_cap.astype(np.float64)))
    lower -= float(np.dot(rho_sat[problem.active_sat_ids], problem.sat_cap.astype(np.float64)))
    lower -= float(np.dot(gamma_gs, problem.gs_cap.astype(np.float64)))
    return lower


def count_violations(
    problem: ProblemData,
    trace: TraceResult,
    pop_load: NDArray[np.float64],
    sat_load: NDArray[np.float64],
    gs_load: NDArray[np.float64],
) -> tuple[int, int, int, int]:
    isl_over = int(np.count_nonzero(trace.edge_load > problem.edge_cap + EPS))
    pop_over = int(np.count_nonzero(pop_load > problem.pop_cap + EPS))
    sat_over = int(
        np.count_nonzero(
            sat_load[problem.active_sat_ids] > problem.sat_cap.astype(np.float64) + EPS
        )
    )
    gs_over = int(np.count_nonzero(gs_load > problem.gs_cap + EPS))
    return isl_over, pop_over, sat_over, gs_over


def trace_single_path(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    origin: int,
    candidate_idx: int,
    max_hops: int,
) -> tuple[list[int] | None, float]:
    """追踪一个候选 ``(PoP, GS, egress_sat)`` 的路径，用于 repair/local-search。

    复杂度 O(path_hops)。主循环不调用这个函数。
    """

    cur = int(origin)
    row = int(problem.candidate_egress_row[candidate_idx])
    dst_node = int(problem.candidate_node[candidate_idx])
    out: list[int] = []
    loss = 0.0
    hops = 0
    while cur != dst_node:
        if hops >= max_hops:
            return None, math.inf
        nxt = int(predecessors[row, cur])
        if nxt < 0:
            return None, math.inf
        eid = int(problem.edge_id[cur, nxt])
        if eid < 0:
            return None, math.inf
        out.append(eid)
        loss += float(problem.edge_base_loss[eid])
        cur = nxt
        hops += 1
    return out, loss


def _edge_path_nodes(
    problem: ProblemData,
    origin: int,
    edges: tuple[int, ...],
) -> tuple[int, ...] | None:
    """把 edge-id path 转成 node path。"""

    nodes = [int(origin)]
    cur = int(origin)
    for eid_raw in edges:
        eid = int(eid_raw)
        a = int(problem.edge_src[eid])
        b = int(problem.edge_dst[eid])
        if cur == a:
            cur = b
        elif cur == b:
            cur = a
        else:
            return None
        nodes.append(cur)
    return tuple(nodes)


def _shortest_path_with_bans(
    problem: ProblemData,
    origin: int,
    target: int,
    *,
    banned_edges: set[int],
    banned_nodes: set[int],
    max_hops: int,
) -> tuple[tuple[int, ...] | None, float]:
    """Dijkstra with temporary banned edges/nodes for Yen spur search."""

    if origin in banned_nodes or target in banned_nodes:
        return None, math.inf
    heap: list[tuple[float, int]] = [(0.0, int(origin))]
    dist: dict[int, float] = {int(origin): 0.0}
    prev: dict[int, tuple[int, int]] = {}
    target = int(target)
    while heap:
        cost, node = heapq.heappop(heap)
        if cost != dist.get(node):
            continue
        if node == target:
            break
        for nxt, eid, weight in problem.adjacency[node]:
            if eid in banned_edges or nxt in banned_nodes:
                continue
            new_cost = cost + weight
            if new_cost + EPS < dist.get(nxt, math.inf):
                dist[nxt] = new_cost
                prev[nxt] = (node, eid)
                heapq.heappush(heap, (new_cost, nxt))
    if target not in dist:
        return None, math.inf
    edges_rev: list[int] = []
    cur = target
    while cur != origin:
        if len(edges_rev) >= max_hops:
            return None, math.inf
        parent, eid = prev[cur]
        edges_rev.append(eid)
        cur = parent
    edges_rev.reverse()
    return tuple(edges_rev), float(dist[target])


def k_shortest_isl_paths(
    problem: ProblemData,
    origin: int,
    candidate_idx: int,
    *,
    max_hops: int,
    k: int,
    primary_edges: list[int] | None,
    primary_loss: float,
) -> list[tuple[tuple[int, ...], float]]:
    """Lazy Yen top-k simple paths for one ``(ingress, egress)`` pair.

    The first path is the controller's precomputed shortest path. Extra
    paths are generated only when capacity checks need them.
    """

    target = int(problem.candidate_node[candidate_idx])
    if k <= 1:
        if primary_edges is None:
            return []
        return [(tuple(primary_edges), float(primary_loss))]
    if primary_edges is None:
        first_edges, first_loss = _shortest_path_with_bans(
            problem,
            origin,
            target,
            banned_edges=set(),
            banned_nodes=set(),
            max_hops=max_hops,
        )
        if first_edges is None:
            return []
    else:
        first_edges = tuple(int(e) for e in primary_edges)
        first_loss = float(primary_loss)

    accepted: list[tuple[tuple[int, ...], float]] = [(first_edges, first_loss)]
    accepted_keys = {first_edges}
    candidates: list[tuple[float, tuple[int, ...]]] = []
    candidate_keys: set[tuple[int, ...]] = set()

    while len(accepted) < k:
        last_edges = accepted[-1][0]
        last_nodes = _edge_path_nodes(problem, origin, last_edges)
        if last_nodes is None or len(last_nodes) <= 1:
            break
        root_loss = 0.0
        for spur_idx in range(len(last_nodes) - 1):
            spur_node = last_nodes[spur_idx]
            root_nodes = last_nodes[: spur_idx + 1]
            root_edges = last_edges[:spur_idx]
            if spur_idx > 0:
                root_loss += float(problem.edge_base_loss[last_edges[spur_idx - 1]])
            banned_edges: set[int] = set()
            for path_edges, _loss in accepted:
                nodes = _edge_path_nodes(problem, origin, path_edges)
                if (
                    nodes is not None
                    and len(path_edges) > spur_idx
                    and nodes[: spur_idx + 1] == root_nodes
                ):
                    banned_edges.add(int(path_edges[spur_idx]))
            banned_nodes = set(root_nodes[:-1])
            spur_edges, spur_loss = _shortest_path_with_bans(
                problem,
                spur_node,
                target,
                banned_edges=banned_edges,
                banned_nodes=banned_nodes,
                max_hops=max_hops - len(root_edges),
            )
            if spur_edges is None:
                continue
            total_edges = tuple(root_edges) + spur_edges
            if total_edges in accepted_keys or total_edges in candidate_keys:
                continue
            if len(total_edges) > max_hops:
                continue
            total_loss = root_loss + spur_loss
            heapq.heappush(candidates, (total_loss, total_edges))
            candidate_keys.add(total_edges)
        if not candidates:
            break
        loss, edges = heapq.heappop(candidates)
        accepted.append((edges, loss))
        accepted_keys.add(edges)
    return accepted


def _path_has_capacity(
    edge_load: NDArray[np.float64],
    edge_cap: NDArray[np.float32],
    edges: tuple[int, ...] | list[int],
    demand: float,
) -> bool:
    return all(
        edge_load[int(eid)] + demand <= edge_cap[int(eid)] + EPS
        for eid in edges
    )


def choose_capacity_feasible_path(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    origin: int,
    candidate: int,
    edge_load: NDArray[np.float64],
    demand: float,
    *,
    max_hops: int,
    path_variants: int,
    variant_cache: dict[tuple[int, int], list[tuple[tuple[int, ...], float]]],
) -> tuple[tuple[int, ...] | None, float]:
    """Try shortest path, then optional 2nd/3rd shortest ISL paths."""

    primary_edges, primary_loss = trace_single_path(
        problem,
        predecessors,
        origin,
        candidate,
        max_hops,
    )
    if primary_edges is None:
        return None, math.inf
    if _path_has_capacity(edge_load, problem.edge_cap, primary_edges, demand):
        return tuple(primary_edges), primary_loss
    if path_variants <= 1:
        return None, math.inf

    cache_key = (int(origin), int(candidate))
    variants = variant_cache.get(cache_key)
    if variants is None or len(variants) < path_variants:
        variants = k_shortest_isl_paths(
            problem,
            origin,
            candidate,
            max_hops=max_hops,
            k=path_variants,
            primary_edges=primary_edges,
            primary_loss=primary_loss,
        )
        variant_cache[cache_key] = variants
    for edges, loss in variants[1:path_variants]:
        if _path_has_capacity(edge_load, problem.edge_cap, edges, demand):
            return edges, loss
    return None, math.inf


def _item_old_edges(trace: TraceResult, item_idx: int) -> set[int]:
    count = int(trace.path_counts[item_idx])
    if count <= 0:
        return set()
    return {int(e) for e in trace.path_edges[item_idx, :count] if int(e) >= 0}


def can_move_item(
    problem: ProblemData,
    trace: TraceResult,
    pop_load: NDArray[np.float64],
    sat_load: NDArray[np.float64],
    gs_load: NDArray[np.float64],
    item_idx: int,
    old_candidate: int,
    new_candidate: int,
    new_edges: list[int],
) -> bool:
    """检查迁移一个 commodity 后是否仍满足容量。复杂度 O(path_hops)。"""

    w = float(problem.weights[item_idx])
    old_pop = int(problem.candidate_pop_idx[old_candidate])
    new_pop = int(problem.candidate_pop_idx[new_candidate])
    if pop_load[new_pop] + w - (w if new_pop == old_pop else 0.0) > problem.pop_cap[new_pop] + EPS:
        return False
    old_sat = int(problem.candidate_sat[old_candidate])
    new_sat = int(problem.candidate_sat[new_candidate])
    sat_cap = float(problem.sat_cap[0]) if problem.sat_cap.size else 0.0
    if sat_load[new_sat] + w - (w if new_sat == old_sat else 0.0) > sat_cap + EPS:
        return False
    old_gs = int(problem.candidate_gs_idx[old_candidate])
    new_gs = int(problem.candidate_gs_idx[new_candidate])
    if gs_load[new_gs] + w - (w if new_gs == old_gs else 0.0) > problem.gs_cap[new_gs] + EPS:
        return False
    old_edges = _item_old_edges(trace, item_idx)
    for eid in new_edges:
        after_release = trace.edge_load[eid] - (w if eid in old_edges else 0.0)
        if after_release + w > problem.edge_cap[eid] + EPS:
            return False
    return True


def apply_move(
    problem: ProblemData,
    trace: TraceResult,
    pop_load: NDArray[np.float64],
    sat_load: NDArray[np.float64],
    gs_load: NDArray[np.float64],
    assignment: NDArray[np.int32],
    item_idx: int,
    new_candidate: int,
    new_edges: list[int],
    new_path_loss: float,
) -> None:
    """原地应用一个 commodity 改道。复杂度 O(old_hops + new_hops)。"""

    old_candidate = int(assignment[item_idx])
    w = float(problem.weights[item_idx])
    old_pop = int(problem.candidate_pop_idx[old_candidate])
    new_pop = int(problem.candidate_pop_idx[new_candidate])
    old_sat = int(problem.candidate_sat[old_candidate])
    new_sat = int(problem.candidate_sat[new_candidate])
    old_gs = int(problem.candidate_gs_idx[old_candidate])
    new_gs = int(problem.candidate_gs_idx[new_candidate])
    pop_load[old_pop] -= w
    pop_load[new_pop] += w
    sat_load[old_sat] -= w
    sat_load[new_sat] += w
    gs_load[old_gs] -= w
    gs_load[new_gs] += w

    old_count = int(trace.path_counts[item_idx])
    if old_count > 0:
        for eid_raw in trace.path_edges[item_idx, :old_count]:
            eid = int(eid_raw)
            if eid >= 0:
                trace.edge_load[eid] -= w
    trace.path_edges[item_idx, :] = -1
    for h, eid in enumerate(new_edges[: trace.path_edges.shape[1]]):
        trace.path_edges[item_idx, h] = int(eid)
        trace.edge_load[int(eid)] += w
    trace.path_counts[item_idx] = min(len(new_edges), trace.path_edges.shape[1])
    trace.item_path_loss[item_idx] = np.float32(new_path_loss)
    assignment[item_idx] = int(new_candidate)


def repair_solution(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    assignment: NDArray[np.int32],
    trace: TraceResult,
    pop_load: NDArray[np.float64],
    sat_load: NDArray[np.float64],
    gs_load: NDArray[np.float64],
    *,
    deadline: float,
    max_hops: int,
) -> None:
    """修复超载资源：优先移动改道损耗增量最低的 commodity。

    这是预算内的启发式 repair，不保证最优；目标是快速得到可行解。
    """

    while time.perf_counter() < deadline:
        overloaded_edges = np.flatnonzero(trace.edge_load > problem.edge_cap + EPS)
        overloaded_pops = np.flatnonzero(pop_load > problem.pop_cap + EPS)
        overloaded_sats = problem.active_sat_ids[
            sat_load[problem.active_sat_ids] > problem.sat_cap.astype(np.float64) + EPS
        ]
        overloaded_gs = np.flatnonzero(gs_load > problem.gs_cap + EPS)
        if (
            overloaded_edges.size == 0
            and overloaded_pops.size == 0
            and overloaded_sats.size == 0
            and overloaded_gs.size == 0
        ):
            return

        resource_kind = "edge"
        if overloaded_edges.size > 0:
            edge_ratio = float(
                np.max(trace.edge_load[overloaded_edges] / problem.edge_cap[overloaded_edges])
            )
        else:
            edge_ratio = -1.0
        if overloaded_pops.size > 0:
            pop_ratio = float(np.max(pop_load[overloaded_pops] / problem.pop_cap[overloaded_pops]))
        else:
            pop_ratio = -1.0
        if overloaded_sats.size > 0:
            sat_cap = float(problem.sat_cap[0]) if problem.sat_cap.size else 0.0
            sat_ratio = float(np.max(sat_load[overloaded_sats] / sat_cap))
        else:
            sat_ratio = -1.0
        if overloaded_gs.size > 0:
            gs_ratio = float(np.max(gs_load[overloaded_gs] / problem.gs_cap[overloaded_gs]))
        else:
            gs_ratio = -1.0
        max_ratio = max(edge_ratio, pop_ratio, sat_ratio, gs_ratio)

        valid_assignment = assignment >= 0
        safe_assignment = np.where(valid_assignment, assignment, 0)
        if max_ratio == edge_ratio:
            resource_kind = "edge"
            resource = int(overloaded_edges[np.argmax(trace.edge_load[overloaded_edges] / problem.edge_cap[overloaded_edges])])
            affected_mask = valid_assignment & np.any(trace.path_edges == resource, axis=1)
        elif max_ratio == sat_ratio:
            resource_kind = "sat"
            sat_cap = float(problem.sat_cap[0]) if problem.sat_cap.size else 0.0
            resource = int(overloaded_sats[np.argmax(sat_load[overloaded_sats] / sat_cap)])
            affected_mask = valid_assignment & (problem.candidate_sat[safe_assignment] == resource)
        elif max_ratio == gs_ratio:
            resource_kind = "gs"
            resource = int(overloaded_gs[np.argmax(gs_load[overloaded_gs] / problem.gs_cap[overloaded_gs])])
            affected_mask = valid_assignment & (problem.candidate_gs_idx[safe_assignment] == resource)
        else:
            resource_kind = "pop"
            resource = int(overloaded_pops[np.argmax(pop_load[overloaded_pops] / problem.pop_cap[overloaded_pops])])
            affected_mask = valid_assignment & (problem.candidate_pop_idx[safe_assignment] == resource)

        affected = np.flatnonzero(affected_mask)
        if affected.size == 0:
            return

        # 大 demand 更容易快速消除超载；每个 item 的候选仍按损耗增量排序。
        affected = affected[np.argsort(-problem.weights[affected])]
        affected = affected[: min(1500, affected.size)]
        moves: list[tuple[float, int, int, list[int], float]] = []

        now = time.perf_counter()
        if now >= deadline:
            return
        for item_idx_raw in affected:
            if time.perf_counter() >= deadline:
                break
            item_idx = int(item_idx_raw)
            old_candidate = int(assignment[item_idx])
            if old_candidate < 0:
                continue
            old_loss = (
                float(trace.item_path_loss[item_idx])
                + float(problem.candidate_access_loss[old_candidate])
                + float(problem.ground_loss[item_idx, problem.candidate_pop_idx[old_candidate]])
            )
            best: tuple[float, int, list[int], float] | None = None
            static_loss = (
                problem.candidate_access_loss
                + problem.ground_loss[item_idx, problem.candidate_pop_idx]
            )
            order_limit = (
                len(problem.candidate_label)
                if resource_kind == "pop"
                else min(256, len(problem.candidate_label))
            )
            approx_order = np.argsort(static_loss, kind="stable")[:order_limit]
            for new_candidate_raw in approx_order:
                new_candidate = int(new_candidate_raw)
                if new_candidate == old_candidate:
                    continue
                if resource_kind == "pop" and int(problem.candidate_pop_idx[new_candidate]) == resource:
                    continue
                if resource_kind == "sat" and int(problem.candidate_sat[new_candidate]) == resource:
                    continue
                if resource_kind == "gs" and int(problem.candidate_gs_idx[new_candidate]) == resource:
                    continue
                new_edges, path_loss = trace_single_path(
                    problem, predecessors, int(problem.origins[item_idx]), new_candidate, max_hops
                )
                if new_edges is None:
                    continue
                if resource_kind == "edge" and resource in new_edges:
                    continue
                if not can_move_item(
                    problem, trace, pop_load, sat_load, gs_load,
                    item_idx, old_candidate, new_candidate, new_edges,
                ):
                    continue
                new_loss = (
                    path_loss
                    + float(problem.candidate_access_loss[new_candidate])
                    + float(problem.ground_loss[item_idx, problem.candidate_pop_idx[new_candidate]])
                )
                delta = float(problem.weights[item_idx]) * (new_loss - old_loss)
                if best is None or delta < best[0]:
                    best = (delta, new_candidate, new_edges, path_loss)
            if best is not None:
                moves.append((best[0], item_idx, best[1], best[2], best[3]))

        if not moves:
            return
        moves.sort(key=lambda x: x[0])
        moved = False
        for _delta, item_idx, new_candidate, new_edges, path_loss in moves:
            if time.perf_counter() >= deadline:
                return
            old_candidate = int(assignment[item_idx])
            if old_candidate < 0:
                continue
            if can_move_item(
                problem, trace, pop_load, sat_load, gs_load,
                item_idx, old_candidate, new_candidate, new_edges,
            ):
                apply_move(
                    problem, trace, pop_load, sat_load, gs_load,
                    assignment, item_idx, new_candidate, new_edges, path_loss,
                )
                moved = True
                if resource_kind == "edge" and trace.edge_load[resource] <= problem.edge_cap[resource] + EPS:
                    break
                if resource_kind == "pop" and pop_load[resource] <= problem.pop_cap[resource] + EPS:
                    break
                if resource_kind == "sat":
                    sat_cap = float(problem.sat_cap[0]) if problem.sat_cap.size else 0.0
                    if sat_load[resource] <= sat_cap + EPS:
                        break
                if resource_kind == "gs" and gs_load[resource] <= problem.gs_cap[resource] + EPS:
                    break
        if not moved:
            return


def local_search(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    assignment: NDArray[np.int32],
    trace: TraceResult,
    pop_load: NDArray[np.float64],
    sat_load: NDArray[np.float64],
    gs_load: NDArray[np.float64],
    *,
    deadline: float,
    max_hops: int,
) -> None:
    """最后 1 秒局部搜索：尝试降低损耗最高的 Top 500 commodity。"""

    valid = (assignment >= 0) & np.isfinite(trace.item_path_loss)
    if not np.any(valid):
        return
    idx = np.arange(assignment.shape[0])
    current_loss = np.full(assignment.shape[0], -np.inf, dtype=np.float64)
    chosen = assignment[valid]
    current_loss[valid] = problem.weights[valid] * (
        trace.item_path_loss[valid].astype(np.float64)
        + problem.candidate_access_loss[chosen].astype(np.float64)
        + problem.ground_loss[idx[valid], problem.candidate_pop_idx[chosen]].astype(np.float64)
    )
    top = np.argsort(-current_loss)[: min(500, assignment.shape[0])]

    for item_idx_raw in top:
        if time.perf_counter() >= deadline:
            return
        item_idx = int(item_idx_raw)
        old_candidate = int(assignment[item_idx])
        if old_candidate < 0:
            continue
        old_loss = (
            float(trace.item_path_loss[item_idx])
            + float(problem.candidate_access_loss[old_candidate])
            + float(problem.ground_loss[item_idx, problem.candidate_pop_idx[old_candidate]])
        )
        static_loss = (
            problem.candidate_access_loss
            + problem.ground_loss[item_idx, problem.candidate_pop_idx]
        )
        for new_candidate_raw in np.argsort(static_loss, kind="stable"):
            if time.perf_counter() >= deadline:
                return
            new_candidate = int(new_candidate_raw)
            if new_candidate == old_candidate:
                continue
            new_edges, path_loss = trace_single_path(
                problem, predecessors, int(problem.origins[item_idx]), new_candidate, max_hops
            )
            if new_edges is None:
                continue
            if not can_move_item(
                problem, trace, pop_load, sat_load, gs_load,
                item_idx, old_candidate, new_candidate, new_edges,
            ):
                continue
            new_loss = (
                path_loss
                + float(problem.candidate_access_loss[new_candidate])
                + float(problem.ground_loss[item_idx, problem.candidate_pop_idx[new_candidate]])
            )
            if new_loss + EPS < old_loss:
                apply_move(
                    problem, trace, pop_load, sat_load, gs_load,
                    assignment, item_idx, new_candidate, new_edges, path_loss,
                )
                break


def greedy_feasible_solution(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    dist_matrix: NDArray[np.float32],
    *,
    deadline: float,
    max_hops: int,
    top_k: int = 16,
    base_cost: NDArray[np.float32] | None = None,
    path_variants: int = 1,
) -> tuple[NDArray[np.int32], TraceResult, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """把 control 计划可行化成 forward 可执行路径。

    复杂度近似 O(|K|·top_k·path_hops)，但通常第一个可行候选就命中。
    这是把 forward 的 capacity charge 提前到 control 的关键步骤。
    """

    n_items = problem.weights.shape[0]
    n_candidates = len(problem.candidate_label)
    effective_top_k = min(top_k, n_candidates)
    cost = base_cost if base_cost is not None else candidate_base_costs(problem, dist_matrix)
    if effective_top_k >= n_candidates:
        candidate_order = np.argsort(cost, axis=1, kind="stable")
    else:
        part = np.argpartition(cost, kth=effective_top_k - 1, axis=1)[:, :effective_top_k]
        rows = np.arange(n_items)[:, None]
        order = np.argsort(cost[rows, part], axis=1, kind="stable")
        candidate_order = part[rows, order]

    assignment = np.full(n_items, -1, dtype=np.int32)
    edge_load = np.zeros(problem.edge_cap.shape[0], dtype=np.float64)
    pop_load = np.zeros(len(problem.pop_names), dtype=np.float64)
    sat_load = np.zeros(problem.n_sats, dtype=np.float64)
    gs_load = np.zeros(len(problem.gs_names), dtype=np.float64)
    item_path_loss = np.full(n_items, INF, dtype=np.float32)
    path_edges = np.full((n_items, max_hops), -1, dtype=np.int32)
    path_counts = np.full(n_items, -1, dtype=np.int32)
    sat_cap = float(problem.sat_cap[0]) if problem.sat_cap.size else 0.0
    variant_cache: dict[tuple[int, int], list[tuple[tuple[int, ...], float]]] = {}

    # 大流量先放，避免小流量碎片化后把大流量挤到不可行。
    item_order = np.argsort(-problem.weights, kind="stable")
    invalid_count = 0
    for item_idx_raw in item_order:
        if time.perf_counter() >= deadline:
            invalid_count += int(n_items - np.count_nonzero(assignment >= 0))
            break
        item_idx = int(item_idx_raw)
        w = float(problem.weights[item_idx])
        chosen_edges: tuple[int, ...] | None = None
        chosen_path_loss = math.inf
        chosen_candidate = -1

        for candidate_raw in candidate_order[item_idx]:
            candidate = int(candidate_raw)
            pop_idx = int(problem.candidate_pop_idx[candidate])
            if pop_load[pop_idx] + w > problem.pop_cap[pop_idx] + EPS:
                continue
            sat_id = int(problem.candidate_sat[candidate])
            if sat_load[sat_id] + w > sat_cap + EPS:
                continue
            gs_idx = int(problem.candidate_gs_idx[candidate])
            if gs_load[gs_idx] + w > problem.gs_cap[gs_idx] + EPS:
                continue
            edges, path_loss = choose_capacity_feasible_path(
                problem,
                predecessors,
                int(problem.origins[item_idx]),
                candidate,
                edge_load,
                w,
                max_hops=max_hops,
                path_variants=path_variants,
                variant_cache=variant_cache,
            )
            if edges is None:
                continue
            chosen_edges = edges
            chosen_path_loss = path_loss
            chosen_candidate = candidate
            break

        if chosen_edges is None and effective_top_k < n_candidates:
            # 精度-速度折中：绝大多数 item 只看 top_k；只有 top_k 内
            # 没路时才全量扫描，保证可行性优先于速度。
            for candidate_raw in np.argsort(cost[item_idx], kind="stable"):
                candidate = int(candidate_raw)
                pop_idx = int(problem.candidate_pop_idx[candidate])
                if pop_load[pop_idx] + w > problem.pop_cap[pop_idx] + EPS:
                    continue
                sat_id = int(problem.candidate_sat[candidate])
                if sat_load[sat_id] + w > sat_cap + EPS:
                    continue
                gs_idx = int(problem.candidate_gs_idx[candidate])
                if gs_load[gs_idx] + w > problem.gs_cap[gs_idx] + EPS:
                    continue
                edges, path_loss = choose_capacity_feasible_path(
                    problem,
                    predecessors,
                    int(problem.origins[item_idx]),
                    candidate,
                    edge_load,
                    w,
                    max_hops=max_hops,
                    path_variants=path_variants,
                    variant_cache=variant_cache,
                )
                if edges is None:
                    continue
                chosen_edges = edges
                chosen_path_loss = path_loss
                chosen_candidate = candidate
                break

        if chosen_edges is None:
            invalid_count += 1
            continue

        assignment[item_idx] = chosen_candidate
        item_path_loss[item_idx] = np.float32(chosen_path_loss)
        path_counts[item_idx] = min(len(chosen_edges), max_hops)
        for h, eid in enumerate(chosen_edges[:max_hops]):
            path_edges[item_idx, h] = int(eid)
            edge_load[int(eid)] += w
        pop_load[int(problem.candidate_pop_idx[chosen_candidate])] += w
        sat_load[int(problem.candidate_sat[chosen_candidate])] += w
        gs_load[int(problem.candidate_gs_idx[chosen_candidate])] += w

    trace = TraceResult(
        edge_load=edge_load,
        item_path_loss=item_path_loss,
        path_edges=path_edges,
        path_counts=path_counts,
        item_origin=problem.origins.copy(),
        invalid_count=invalid_count,
    )
    return assignment, trace, pop_load, sat_load, gs_load


def nearest_pop_baseline_solution(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    dist_matrix: NDArray[np.float32],
    *,
    deadline: float,
    max_hops: int,
    path_variants: int = 1,
) -> OptimizerResult:
    """最近 PoP baseline：按地理 PoP cascade 做 forward first-fit。

    对每个 commodity，先走最近 PoP；在一个 PoP 内优先尝试
    bent-pipe 候选（同一颗卫星同时可见 source cell 与 PoP
    attached GS，所以 ingress == egress，ISL 段为空）。若该 PoP
    内没有容量可行的 bent-pipe 候选，再用最高仰角 ingress 并按
    ``shortest-ISL RTT + downlink/backhaul`` 从低到高尝试 egress。
    该 PoP 全部候选不可用后，才走第二近 PoP。
    """

    start = time.perf_counter()
    n_items = problem.weights.shape[0]
    base_cost = candidate_base_costs(problem, dist_matrix)
    pop_candidates = [
        np.flatnonzero(problem.candidate_pop_idx == pop_idx).astype(np.int32)
        for pop_idx in range(len(problem.pop_names))
    ]
    assignment = np.full(n_items, -1, dtype=np.int32)
    edge_load = np.zeros(problem.edge_cap.shape[0], dtype=np.float64)
    pop_load = np.zeros(len(problem.pop_names), dtype=np.float64)
    sat_load = np.zeros(problem.n_sats, dtype=np.float64)
    gs_load = np.zeros(len(problem.gs_names), dtype=np.float64)
    item_path_loss = np.full(n_items, INF, dtype=np.float32)
    path_edges = np.full((n_items, max_hops), -1, dtype=np.int32)
    path_counts = np.full(n_items, -1, dtype=np.int32)
    item_origin = problem.origins.copy()
    sat_cap = float(problem.sat_cap[0]) if problem.sat_cap.size else 0.0
    item_order = np.argsort(-problem.weights, kind="stable")
    variant_cache: dict[tuple[int, int], list[tuple[tuple[int, ...], float]]] = {}
    invalid_count = 0

    for item_idx_raw in item_order:
        if time.perf_counter() >= deadline:
            invalid_count += int(n_items - np.count_nonzero(assignment >= 0))
            break
        item_idx = int(item_idx_raw)
        w = float(problem.weights[item_idx])
        chosen_edges: list[int] | None = None
        chosen_candidate = -1
        chosen_path_loss = math.inf
        chosen_origin = int(problem.origins[item_idx])
        access_by_sat = dict(problem.cell_access.get(int(problem.item_cell[item_idx]), ()))

        for pop_idx_raw in problem.nearest_pop_order[item_idx]:
            pop_idx = int(pop_idx_raw)
            if pop_load[pop_idx] + w > problem.pop_cap[pop_idx] + EPS:
                continue
            candidates = pop_candidates[pop_idx]
            if candidates.size == 0:
                continue

            bent_candidates: list[int] = []
            if access_by_sat:
                for candidate_raw in candidates:
                    candidate = int(candidate_raw)
                    if int(problem.candidate_sat[candidate]) in access_by_sat:
                        bent_candidates.append(candidate)
                bent_candidates.sort(
                    key=lambda candidate: (
                        access_by_sat[int(problem.candidate_sat[candidate])]
                        + float(problem.candidate_access_loss[candidate]),
                        float(problem.candidate_access_loss[candidate]),
                    )
                )

            for candidate in bent_candidates:
                sat_id = int(problem.candidate_sat[candidate])
                if sat_load[sat_id] + w > sat_cap + EPS:
                    continue
                gs_idx = int(problem.candidate_gs_idx[candidate])
                if gs_load[gs_idx] + w > problem.gs_cap[gs_idx] + EPS:
                    continue
                chosen_edges = []
                chosen_candidate = candidate
                chosen_path_loss = 0.0
                chosen_origin = sat_id
                break
            if chosen_edges is not None:
                break

            local_order = candidates[
                np.argsort(base_cost[item_idx, candidates], kind="stable")
            ]
            fallback_origin = int(problem.origins[item_idx])
            for candidate_raw in local_order:
                candidate = int(candidate_raw)
                sat_id = int(problem.candidate_sat[candidate])
                if sat_load[sat_id] + w > sat_cap + EPS:
                    continue
                gs_idx = int(problem.candidate_gs_idx[candidate])
                if gs_load[gs_idx] + w > problem.gs_cap[gs_idx] + EPS:
                    continue
                edges, path_loss = choose_capacity_feasible_path(
                    problem,
                    predecessors,
                    fallback_origin,
                    candidate,
                    edge_load,
                    w,
                    max_hops=max_hops,
                    path_variants=path_variants,
                    variant_cache=variant_cache,
                )
                if edges is None:
                    continue
                chosen_edges = edges
                chosen_candidate = candidate
                chosen_path_loss = path_loss
                chosen_origin = fallback_origin
                break
            if chosen_edges is not None:
                break

        if chosen_edges is None:
            invalid_count += 1
            continue

        assignment[item_idx] = chosen_candidate
        item_origin[item_idx] = chosen_origin
        item_path_loss[item_idx] = np.float32(chosen_path_loss)
        path_counts[item_idx] = min(len(chosen_edges), max_hops)
        for h, eid in enumerate(chosen_edges[:max_hops]):
            path_edges[item_idx, h] = int(eid)
            edge_load[int(eid)] += w
        pop_load[int(problem.candidate_pop_idx[chosen_candidate])] += w
        sat_load[int(problem.candidate_sat[chosen_candidate])] += w
        gs_load[int(problem.candidate_gs_idx[chosen_candidate])] += w

    trace = TraceResult(
        edge_load=edge_load,
        item_path_loss=item_path_loss,
        path_edges=path_edges,
        path_counts=path_counts,
        item_origin=item_origin,
        invalid_count=invalid_count,
    )
    objective = actual_objective(problem, assignment, trace.item_path_loss)
    elapsed = time.perf_counter() - start
    return OptimizerResult(
        assignment=assignment,
        trace=trace,
        pop_load=pop_load,
        sat_load=sat_load,
        gs_load=gs_load,
        objective=objective,
        best_lower_bound=math.nan,
        iterations=0,
        total_seconds=elapsed,
        solve_seconds=elapsed,
        main_seconds=0.0,
        repair_seconds=elapsed,
        local_search_seconds=0.0,
        dijkstra_seconds=0.0,
        trace_seconds=elapsed,
        feasible_seen=trace.invalid_count == 0,
    )


def build_candidate_mip(
    problem: ProblemData,
    predecessors: NDArray[np.int32],
    dist_matrix: NDArray[np.float32],
    *,
    top_k: int,
    max_hops: int,
    incumbent_assignment: NDArray[np.int32] | None = None,
    candidate_mask: NDArray[np.bool_] | None = None,
    extra_columns: list[ExtraColumn] | None = None,
) -> CandidateMipData:
    """构建 top-k 候选集上的 exact MIP。

    每个变量表示一个不可拆 commodity 选择一个
    ``(PoP, GS, egress_sat, shortest ISL path)`` candidate。约束包括：
    每个 commodity 选一个、PoP 容量、egress sat feeder、GS feeder 和
    每条 ISL 容量。复杂度约为 O(|K|·top_k·path_hops)。
    """

    t0 = time.perf_counter()
    n_items = problem.weights.shape[0]
    n_candidates = len(problem.candidate_label)
    effective_top_k = max(1, min(int(top_k), n_candidates))
    base_cost = candidate_base_costs(problem, dist_matrix)
    if candidate_mask is None:
        if effective_top_k >= n_candidates:
            candidate_order = np.argsort(base_cost, axis=1, kind="stable")
        else:
            part = np.argpartition(base_cost, kth=effective_top_k - 1, axis=1)[
                :, :effective_top_k
            ]
            item_rows = np.arange(n_items)[:, None]
            order = np.argsort(base_cost[item_rows, part], axis=1, kind="stable")
            candidate_order = part[item_rows, order]
    else:
        candidate_order = None

    n_pops = len(problem.pop_names)
    n_gs = len(problem.gs_names)
    n_active_sats = len(problem.active_sat_ids)
    n_edges = len(problem.edge_cap)
    pop_offset = n_items
    sat_offset = pop_offset + n_pops
    gs_offset = sat_offset + n_active_sats
    edge_offset = gs_offset + n_gs
    n_rows = edge_offset + n_edges
    sat_resource_idx = np.full(problem.n_sats, -1, dtype=np.int32)
    sat_resource_idx[problem.active_sat_ids] = np.arange(n_active_sats, dtype=np.int32)

    lb = np.zeros(n_rows, dtype=np.float64)
    ub = np.full(n_rows, np.inf, dtype=np.float64)
    lb[:n_items] = 1.0
    ub[:n_items] = 1.0
    ub[pop_offset:sat_offset] = problem.pop_cap.astype(np.float64)
    ub[sat_offset:gs_offset] = problem.sat_cap.astype(np.float64)
    ub[gs_offset:edge_offset] = problem.gs_cap.astype(np.float64)
    ub[edge_offset:] = problem.edge_cap.astype(np.float64)
    ub[pop_offset:] += EPS

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    c: list[float] = []
    var_item: list[int] = []
    var_candidate: list[int] = []
    path_cache: dict[tuple[int, int], tuple[list[int] | None, float]] = {}
    skipped_items = 0
    forced_incumbent_vars = 0
    col = 0

    for item_idx in range(n_items):
        w = float(problem.weights[item_idx])
        origin = int(problem.origins[item_idx])
        item_added = 0
        if candidate_mask is None:
            assert candidate_order is not None
            item_candidates = candidate_order[item_idx]
        else:
            item_candidates = np.flatnonzero(candidate_mask[item_idx]).astype(np.int32)
            if item_candidates.size:
                item_candidates = item_candidates[
                    np.argsort(base_cost[item_idx, item_candidates], kind="stable")
                ]
        if incumbent_assignment is not None and candidate_mask is None:
            incumbent_candidate = int(incumbent_assignment[item_idx])
            if incumbent_candidate >= 0 and not np.any(item_candidates == incumbent_candidate):
                item_candidates = np.concatenate([
                    item_candidates,
                    np.asarray([incumbent_candidate], dtype=item_candidates.dtype),
                ])
                forced_incumbent_vars += 1
        for candidate_raw in item_candidates:
            candidate = int(candidate_raw)
            path_key = (origin, int(problem.candidate_egress_row[candidate]))
            path = path_cache.get(path_key)
            if path is None:
                path = trace_single_path(
                    problem,
                    predecessors,
                    origin,
                    candidate,
                    max_hops,
                )
                path_cache[path_key] = path
            edges, path_loss = path
            if edges is None:
                continue

            pop_idx = int(problem.candidate_pop_idx[candidate])
            sat_idx = int(problem.candidate_sat[candidate])
            sat_row = int(sat_resource_idx[sat_idx])
            gs_idx = int(problem.candidate_gs_idx[candidate])
            if sat_row < 0:
                continue
            unit_loss = (
                path_loss
                + float(problem.candidate_access_loss[candidate])
                + float(problem.ground_loss[item_idx, pop_idx])
            )
            c.append(w * unit_loss)
            var_item.append(item_idx)
            var_candidate.append(candidate)

            rows.append(item_idx)
            cols.append(col)
            data.append(1.0)
            rows.append(pop_offset + pop_idx)
            cols.append(col)
            data.append(w)
            rows.append(sat_offset + sat_row)
            cols.append(col)
            data.append(w)
            rows.append(gs_offset + gs_idx)
            cols.append(col)
            data.append(w)
            for eid in edges:
                rows.append(edge_offset + int(eid))
                cols.append(col)
                data.append(w)

            col += 1
            item_added += 1
        if item_added == 0:
            skipped_items += 1

    if extra_columns:
        for extra in extra_columns:
            item_idx = int(extra.item_idx)
            candidate = int(extra.candidate_idx)
            w = float(problem.weights[item_idx])
            pop_idx = int(problem.candidate_pop_idx[candidate])
            sat_idx = int(problem.candidate_sat[candidate])
            sat_row = int(sat_resource_idx[sat_idx])
            gs_idx = int(problem.candidate_gs_idx[candidate])
            if sat_row < 0:
                continue
            unit_loss = (
                float(extra.path_loss)
                + float(problem.candidate_access_loss[candidate])
                + float(problem.ground_loss[item_idx, pop_idx])
            )
            c.append(w * unit_loss)
            var_item.append(item_idx)
            var_candidate.append(candidate)
            rows.append(item_idx)
            cols.append(col)
            data.append(1.0)
            rows.append(pop_offset + pop_idx)
            cols.append(col)
            data.append(w)
            rows.append(sat_offset + sat_row)
            cols.append(col)
            data.append(w)
            rows.append(gs_offset + gs_idx)
            cols.append(col)
            data.append(w)
            for eid in extra.path_edges:
                rows.append(edge_offset + int(eid))
                cols.append(col)
                data.append(w)
            col += 1

    matrix = csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(cols, dtype=np.int32),
            ),
        ),
        shape=(n_rows, col),
    )
    return CandidateMipData(
        top_k=effective_top_k,
        c=np.asarray(c, dtype=np.float64),
        constraints=LinearConstraint(matrix, lb, ub),
        var_item=np.asarray(var_item, dtype=np.int32),
        var_candidate=np.asarray(var_candidate, dtype=np.int32),
        build_seconds=time.perf_counter() - t0,
        skipped_items=skipped_items,
        forced_incumbent_vars=forced_incumbent_vars,
        n_rows=n_rows,
        n_resource_rows=n_rows - n_items,
        nnz=matrix.nnz,
    )


def decode_mip_assignment(
    problem: ProblemData,
    mip_data: CandidateMipData,
    x: NDArray[np.float64],
) -> NDArray[np.int32]:
    """把 MIP 解向量转回 optimizer assignment。"""

    assignment = np.full(problem.weights.shape[0], -1, dtype=np.int32)
    chosen_cols = np.flatnonzero(x > 0.5)
    for col in chosen_cols:
        assignment[int(mip_data.var_item[col])] = int(mip_data.var_candidate[col])
    return assignment


def run_optimal_certificate(
    problem: ProblemData,
    args: argparse.Namespace,
    incumbent_assignment: NDArray[np.int32] | None = None,
) -> OptimalCertificateResult:
    """求解 top-k candidate universe 的 LP/MILP certificate。"""

    runner = DijkstraRunner(problem, static_paths=True)
    dist, pred, _elapsed = runner.run(np.zeros(problem.edge_base_loss.shape[0]))
    mip_data = build_candidate_mip(
        problem,
        pred,
        dist,
        top_k=args.optimal_top_k,
        max_hops=args.max_hops,
        incumbent_assignment=incumbent_assignment,
    )
    bounds = Bounds(
        np.zeros(mip_data.c.shape[0], dtype=np.float64),
        np.ones(mip_data.c.shape[0], dtype=np.float64),
    )
    lp_start = time.perf_counter()
    lp_res = milp(
        mip_data.c,
        integrality=np.zeros(mip_data.c.shape[0], dtype=np.int8),
        bounds=bounds,
        constraints=(mip_data.constraints,),
        options={"time_limit": max(1.0, min(30.0, args.optimal_time_limit))},
    )
    lp_seconds = time.perf_counter() - lp_start

    mip_res = None
    mip_seconds = 0.0
    if not args.optimal_lp_only:
        mip_start = time.perf_counter()
        mip_res = milp(
            mip_data.c,
            integrality=np.ones(mip_data.c.shape[0], dtype=np.int8),
            bounds=bounds,
            constraints=(mip_data.constraints,),
            options={
                "time_limit": max(1.0, args.optimal_time_limit),
                "mip_rel_gap": 0.0,
            },
        )
        mip_seconds = time.perf_counter() - mip_start

    violations: tuple[int, int, int, int] | None = None
    invalid_count: int | None = None
    if mip_res is not None and getattr(mip_res, "x", None) is not None:
        assignment = decode_mip_assignment(problem, mip_data, mip_res.x)
        trace = trace_paths(problem, pred, assignment, max_hops=args.max_hops)
        pop_load, sat_load, gs_load = resource_loads(problem, assignment)
        violations = count_violations(problem, trace, pop_load, sat_load, gs_load)
        invalid_count = trace.invalid_count

    return OptimalCertificateResult(
        top_k=mip_data.top_k,
        build_seconds=mip_data.build_seconds,
        lp_seconds=lp_seconds,
        mip_seconds=mip_seconds,
        lp_fun=float(lp_res.fun) if getattr(lp_res, "fun", None) is not None else None,
        lp_success=bool(lp_res.success),
        lp_status=int(lp_res.status),
        lp_message=str(lp_res.message),
        mip_fun=(
            float(mip_res.fun)
            if mip_res is not None and getattr(mip_res, "fun", None) is not None
            else None
        ),
        mip_dual_bound=(
            float(mip_res.mip_dual_bound)
            if mip_res is not None
            and getattr(mip_res, "mip_dual_bound", None) is not None
            else None
        ),
        mip_gap=(
            float(mip_res.mip_gap)
            if mip_res is not None and getattr(mip_res, "mip_gap", None) is not None
            else None
        ),
        mip_success=bool(mip_res.success) if mip_res is not None else False,
        mip_status=int(mip_res.status) if mip_res is not None else None,
        mip_message=str(mip_res.message) if mip_res is not None else None,
        mip_nodes=(
            int(mip_res.mip_node_count)
            if mip_res is not None
            and getattr(mip_res, "mip_node_count", None) is not None
            else None
        ),
        skipped_items=mip_data.skipped_items,
        forced_incumbent_vars=mip_data.forced_incumbent_vars,
        n_vars=mip_data.c.shape[0],
        n_rows=mip_data.n_rows,
        nnz=mip_data.nnz,
        violations=violations,
        invalid_count=invalid_count,
    )


def initial_candidate_mask(
    problem: ProblemData,
    base_cost: NDArray[np.float32],
    *,
    top_k: int,
    incumbent_assignment: NDArray[np.int32] | None,
) -> NDArray[np.bool_]:
    """列生成初始列池：top-k base shortest-path + incumbent。"""

    n_items, n_candidates = base_cost.shape
    effective_top_k = max(1, min(int(top_k), n_candidates))
    mask = np.zeros((n_items, n_candidates), dtype=np.bool_)
    if effective_top_k >= n_candidates:
        mask[:, :] = True
    else:
        part = np.argpartition(base_cost, kth=effective_top_k - 1, axis=1)[
            :, :effective_top_k
        ]
        rows = np.arange(n_items)[:, None]
        mask[rows, part] = True
    if incumbent_assignment is not None:
        valid = incumbent_assignment >= 0
        mask[np.flatnonzero(valid), incumbent_assignment[valid]] = True
    return mask


def solve_restricted_lp(
    mip_data: CandidateMipData,
    *,
    n_items: int,
    time_limit: float,
) -> Any:
    """解当前列池 LP，并保留 dual marginals 供 pricing 使用。"""

    matrix = csr_matrix(mip_data.constraints.A)
    return linprog(
        mip_data.c,
        A_ub=matrix[n_items:, :],
        b_ub=mip_data.constraints.ub[n_items:],
        A_eq=matrix[:n_items, :],
        b_eq=np.ones(n_items, dtype=np.float64),
        bounds=(0.0, None),
        method="highs",
        options={"time_limit": max(1.0, time_limit)},
    )


def run_column_generation(
    problem: ProblemData,
    args: argparse.Namespace,
    incumbent: OptimizerResult,
) -> ColumnGenerationResult:
    """列生成 + Lagrangian pricing，逼近全候选 path-based LP 下界。

    初始列池只含每个 commodity 的 top-k base shortest-path
    candidates，再补入 incumbent 保证可行。每轮解 restricted LP，
    读取 dual price 后用 modified ISL weights 做一次多源 Dijkstra，
    为 reduced cost 为负的 commodity 加入新列。
    """

    started = time.perf_counter()
    deadline = started + max(1.0, args.cg_time_limit)
    n_items = problem.weights.shape[0]
    base_runner = DijkstraRunner(problem, static_paths=True)
    base_dist, base_pred, _ = base_runner.run(np.zeros(problem.edge_base_loss.shape[0]))
    base_cost = candidate_base_costs(problem, base_dist)
    candidate_mask = initial_candidate_mask(
        problem,
        base_cost,
        top_k=args.cg_initial_top_k,
        incumbent_assignment=incumbent.assignment,
    )
    extra_columns: list[ExtraColumn] = []
    extra_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    pricing_runner = DijkstraRunner(problem, static_paths=False)

    total_build = 0.0
    total_lp = 0.0
    total_pricing = 0.0
    lp_fun = math.inf
    corrected_lb = -math.inf
    best_reduced = math.inf
    added_total = 0
    converged = False
    last_success = False
    last_status = -1
    last_message = ""
    final_cols = 0
    completed_iterations = 0

    for iteration in range(1, max(1, args.cg_max_iterations) + 1):
        if time.perf_counter() >= deadline:
            break
        completed_iterations = iteration
        build_start = time.perf_counter()
        mip_data = build_candidate_mip(
            problem,
            base_pred,
            base_dist,
            top_k=args.cg_initial_top_k,
            max_hops=args.max_hops,
            candidate_mask=candidate_mask,
            extra_columns=extra_columns,
        )
        total_build += time.perf_counter() - build_start
        final_cols = mip_data.c.shape[0]

        lp_start = time.perf_counter()
        lp_res = solve_restricted_lp(
            mip_data,
            n_items=n_items,
            time_limit=max(1.0, deadline - time.perf_counter()),
        )
        total_lp += time.perf_counter() - lp_start
        last_success = bool(lp_res.success)
        last_status = int(lp_res.status)
        last_message = str(lp_res.message)
        if not lp_res.success:
            break
        lp_fun = float(lp_res.fun)

        pricing_start = time.perf_counter()
        item_dual = np.asarray(lp_res.eqlin.marginals, dtype=np.float64)
        resource_dual = np.asarray(lp_res.ineqlin.marginals, dtype=np.float64)
        n_pops = len(problem.pop_names)
        n_active_sats = len(problem.active_sat_ids)
        n_gs = len(problem.gs_names)
        pop_price = np.maximum(0.0, -resource_dual[:n_pops])
        sat_price_active = np.maximum(0.0, -resource_dual[n_pops:n_pops + n_active_sats])
        gs_start = n_pops + n_active_sats
        edge_start = gs_start + n_gs
        gs_price = np.maximum(0.0, -resource_dual[gs_start:edge_start])
        edge_price = np.maximum(0.0, -resource_dual[edge_start:])
        sat_price = np.zeros(problem.n_sats, dtype=np.float64)
        sat_price[problem.active_sat_ids] = sat_price_active

        priced_dist, priced_pred, _ = pricing_runner.run(edge_price)
        path_by_egress = priced_dist[:, problem.origins].T.astype(np.float64)
        unit_cost = (
            path_by_egress[:, problem.candidate_egress_row]
            + problem.candidate_access_loss[None, :].astype(np.float64)
            + problem.ground_loss[:, problem.candidate_pop_idx].astype(np.float64)
            + pop_price[problem.candidate_pop_idx][None, :]
            + sat_price[problem.candidate_sat][None, :]
            + gs_price[problem.candidate_gs_idx][None, :]
        )
        reduced = unit_cost * problem.weights[:, None].astype(np.float64) - item_dual[:, None]
        min_reduced = np.min(reduced, axis=1)
        best_reduced = float(np.min(min_reduced))
        corrected_lb = float(lp_fun + np.minimum(min_reduced, 0.0).sum())

        best_candidate = np.argmin(reduced, axis=1).astype(np.int32)
        negative_items = np.flatnonzero(min_reduced < -abs(args.cg_pricing_tol))
        if negative_items.size == 0:
            converged = True
            total_pricing += time.perf_counter() - pricing_start
            break
        if negative_items.size > args.cg_max_new_columns:
            keep = np.argpartition(
                min_reduced[negative_items],
                kth=args.cg_max_new_columns - 1,
            )[: args.cg_max_new_columns]
            negative_items = negative_items[keep]

        added_this_round = 0
        for item_idx_raw in negative_items:
            if time.perf_counter() >= deadline:
                break
            item_idx = int(item_idx_raw)
            candidate = int(best_candidate[item_idx])
            edges, path_loss = trace_single_path(
                problem,
                priced_pred,
                int(problem.origins[item_idx]),
                candidate,
                args.max_hops,
            )
            if edges is None:
                continue
            edge_tuple = tuple(int(e) for e in edges)
            if candidate_mask[item_idx, candidate]:
                base_edges, _base_loss = trace_single_path(
                    problem,
                    base_pred,
                    int(problem.origins[item_idx]),
                    candidate,
                    args.max_hops,
                )
                if base_edges is not None and tuple(int(e) for e in base_edges) == edge_tuple:
                    continue
            key = (item_idx, candidate, edge_tuple)
            if key in extra_keys:
                continue
            if not candidate_mask[item_idx, candidate]:
                base_edges, _base_loss = trace_single_path(
                    problem,
                    base_pred,
                    int(problem.origins[item_idx]),
                    candidate,
                    args.max_hops,
                )
                if base_edges is not None and tuple(int(e) for e in base_edges) == edge_tuple:
                    candidate_mask[item_idx, candidate] = True
                    added_this_round += 1
                    continue
            extra_keys.add(key)
            extra_columns.append(ExtraColumn(
                item_idx=item_idx,
                candidate_idx=candidate,
                path_edges=edge_tuple,
                path_loss=float(path_loss),
            ))
            added_this_round += 1
        added_total += added_this_round
        total_pricing += time.perf_counter() - pricing_start
        if added_this_round == 0:
            break

    if math.isfinite(corrected_lb) and incumbent.objective > 0:
        gap_percent = max(0.0, (incumbent.objective - corrected_lb) / incumbent.objective * 100.0)
    else:
        gap_percent = math.nan
    return ColumnGenerationResult(
        iterations=completed_iterations,
        converged=converged,
        lp_fun=lp_fun,
        corrected_lower_bound=corrected_lb,
        best_reduced_cost=best_reduced,
        incumbent_objective=incumbent.objective,
        gap_percent=gap_percent,
        final_columns=final_cols,
        extra_columns=len(extra_columns),
        added_columns=added_total,
        build_seconds=total_build,
        lp_seconds=total_lp,
        pricing_seconds=total_pricing,
        total_seconds=time.perf_counter() - started,
        last_lp_success=last_success,
        last_lp_status=last_status,
        last_lp_message=last_message,
    )


def solve(
    problem: ProblemData,
    args: argparse.Namespace,
    *,
    global_deadline: float,
    program_start: float,
) -> OptimizerResult:
    """拉格朗日松弛 + 次梯度主循环 + repair + local search。"""

    solve_start = time.perf_counter()
    remaining_total = max(0.0, global_deadline - solve_start)
    main_budget = min(
        max(0.0, args.main_loop_budget),
        max(0.0, remaining_total - args.repair_budget - args.local_search_budget),
    )
    main_deadline = min(solve_start + main_budget, global_deadline)
    sigma = np.zeros(problem.edge_base_loss.shape[0], dtype=np.float64)
    pi_pop = np.zeros(len(problem.pop_names), dtype=np.float64)
    rho_sat = np.zeros(problem.n_sats, dtype=np.float64)
    gamma_gs = np.zeros(len(problem.gs_names), dtype=np.float64)
    runner = DijkstraRunner(problem, static_paths=not args.dynamic_isl_prices)

    best_lb = -math.inf
    best_assignment: NDArray[np.int32] | None = None
    best_trace: TraceResult | None = None
    best_pop_load: NDArray[np.float64] | None = None
    best_sat_load: NDArray[np.float64] | None = None
    best_gs_load: NDArray[np.float64] | None = None
    best_obj = math.inf
    feasible_seen = False
    iterations = 0
    dijkstra_seconds = 0.0
    trace_seconds = 0.0
    static_paths = not args.dynamic_isl_prices
    base_cost: NDArray[np.float32] | None = None
    last_assignment: NDArray[np.int32] | None = None
    last_trace: TraceResult | None = None
    last_pop_load: NDArray[np.float64] | None = None
    last_sat_load: NDArray[np.float64] | None = None
    last_gs_load: NDArray[np.float64] | None = None
    last_pred: NDArray[np.int32] | None = None
    last_dist: NDArray[np.float32] | None = None

    while time.perf_counter() < main_deadline and iterations < args.max_iterations:
        if iterations > 0:
            avg_iter = (time.perf_counter() - solve_start) / iterations
            if time.perf_counter() + avg_iter > main_deadline:
                break
        dist, pred, dijkstra_elapsed = runner.run(sigma)
        dijkstra_seconds += dijkstra_elapsed
        if static_paths and base_cost is None:
            base_cost = candidate_base_costs(problem, dist)
        assignment = assign_commodities(
            problem,
            dist,
            pi_pop,
            rho_sat,
            gamma_gs,
            base_cost=base_cost if static_paths else None,
        )

        t_trace = time.perf_counter()
        if static_paths:
            item_path_loss = assignment_path_loss(problem, dist, assignment)
            trace = TraceResult(
                edge_load=np.zeros(problem.edge_cap.shape[0], dtype=np.float64),
                item_path_loss=item_path_loss,
                path_edges=np.empty((0, 0), dtype=np.int32),
                path_counts=np.empty(0, dtype=np.int32),
                item_origin=problem.origins.copy(),
                invalid_count=int(np.count_nonzero(~np.isfinite(item_path_loss))),
            )
        else:
            trace = trace_loads(problem, pred, assignment, max_hops=args.max_hops)
        trace_elapsed = time.perf_counter() - t_trace
        trace_seconds += trace_elapsed
        pop_load, sat_load, gs_load = resource_loads(problem, assignment)

        lb = lagrangian_lower_bound(
            problem,
            dist,
            sigma,
            pi_pop,
            rho_sat,
            gamma_gs,
            base_cost=base_cost if static_paths else None,
        )
        if lb > best_lb:
            best_lb = lb

        obj = actual_objective(problem, assignment, trace.item_path_loss)
        isl_over_count, pop_over_count, sat_over_count, gs_over_count = count_violations(
            problem, trace, pop_load, sat_load, gs_load,
        )
        if (
            not static_paths
            and trace.invalid_count == 0
            and isl_over_count == 0
            and pop_over_count == 0
            and sat_over_count == 0
            and gs_over_count == 0
            and obj < best_obj
        ):
            feasible_seen = True
            best_obj = obj
            best_assignment = assignment.copy()
            best_trace = trace
            best_pop_load = pop_load.copy()
            best_sat_load = sat_load.copy()
            best_gs_load = gs_load.copy()

        pop_subgrad = pop_load - problem.pop_cap.astype(np.float64)
        sat_subgrad = (
            sat_load[problem.active_sat_ids]
            - problem.sat_cap.astype(np.float64)
        )
        gs_subgrad = gs_load - problem.gs_cap.astype(np.float64)
        if static_paths:
            edge_subgrad = np.zeros(problem.edge_base_loss.shape[0], dtype=np.float64)
            subgrad_norm_sq = float(
                np.dot(pop_subgrad, pop_subgrad)
                + np.dot(sat_subgrad, sat_subgrad)
                + np.dot(gs_subgrad, gs_subgrad)
            )
        else:
            edge_subgrad = trace.edge_load - problem.edge_cap.astype(np.float64)
            subgrad_norm_sq = float(
                np.dot(edge_subgrad, edge_subgrad)
                + np.dot(pop_subgrad, pop_subgrad)
                + np.dot(sat_subgrad, sat_subgrad)
                + np.dot(gs_subgrad, gs_subgrad)
            )

        if subgrad_norm_sq <= EPS:
            iterations += 1
            last_assignment = assignment
            last_trace = trace
            last_pop_load = pop_load
            last_sat_load = sat_load
            last_gs_load = gs_load
            last_pred = pred
            last_dist = dist
            break

        # 标准 Polyak 对偶上升步长：UB - LB。若按“LB - UB”会让超载资源惩罚下降。
        if math.isfinite(best_obj):
            gap = max(best_obj - lb, 1.0)
            alpha = args.polyak_beta * gap / subgrad_norm_sq
        else:
            alpha = 0.25 / math.sqrt(iterations + 1.0)
        alpha = min(alpha, 5.0)

        if not static_paths:
            sigma = np.maximum(0.0, sigma + alpha * edge_subgrad)
        pi_pop = np.maximum(0.0, pi_pop + alpha * pop_subgrad)
        rho_sat[problem.active_sat_ids] = np.maximum(
            0.0,
            rho_sat[problem.active_sat_ids] + alpha * sat_subgrad,
        )
        gamma_gs = np.maximum(0.0, gamma_gs + alpha * gs_subgrad)

        iterations += 1
        last_assignment = assignment
        last_trace = trace
        last_pop_load = pop_load
        last_sat_load = sat_load
        last_gs_load = gs_load
        last_pred = pred
        last_dist = dist

    main_seconds = time.perf_counter() - solve_start
    if (
        last_assignment is None
        or last_trace is None
        or last_pop_load is None
        or last_sat_load is None
        or last_gs_load is None
        or last_pred is None
        or last_dist is None
    ):
        raise RuntimeError("Optimizer produced no assignment.")

    if (
        feasible_seen
        and best_assignment is not None
        and best_trace is not None
        and best_pop_load is not None
        and best_sat_load is not None
        and best_gs_load is not None
    ):
        assignment = best_assignment
        pop_load = best_pop_load
        sat_load = best_sat_load
        gs_load = best_gs_load
        objective = best_obj
        pred_for_repair = last_pred
    else:
        assignment = last_assignment.copy()
        pop_load = last_pop_load.copy()
        sat_load = last_sat_load.copy()
        gs_load = last_gs_load.copy()
        objective = actual_objective(problem, assignment, last_trace.item_path_loss)
        pred_for_repair = last_pred

    trace = trace_paths(problem, pred_for_repair, assignment, max_hops=args.max_hops)
    pop_load, sat_load, gs_load = resource_loads(problem, assignment)
    objective = actual_objective(problem, assignment, trace.item_path_loss)

    repair_start = time.perf_counter()
    repair_deadline = min(
        solve_start + main_budget + max(0.0, args.repair_budget),
        solve_start + TOTAL_TIME_LIMIT - max(0.0, args.local_search_budget),
        global_deadline - max(0.0, args.local_search_budget),
    )
    isl_over_count, pop_over_count, sat_over_count, gs_over_count = count_violations(
        problem, trace, pop_load, sat_load, gs_load,
    )
    if (
        trace.invalid_count > 0
        or isl_over_count > 0
        or pop_over_count > 0
        or sat_over_count > 0
        or gs_over_count > 0
    ):
        assignment, trace, pop_load, sat_load, gs_load = greedy_feasible_solution(
            problem,
            pred_for_repair,
            last_dist,
            deadline=repair_deadline,
            max_hops=args.max_hops,
            top_k=max(1, int(args.first_fit_top_k)),
            base_cost=base_cost if static_paths else None,
            path_variants=max(1, int(args.isl_path_variants)),
        )
        objective = actual_objective(problem, assignment, trace.item_path_loss)
        isl_over_count, pop_over_count, sat_over_count, gs_over_count = count_violations(
            problem, trace, pop_load, sat_load, gs_load,
        )
    if (
        trace.invalid_count > 0
        or isl_over_count > 0
        or pop_over_count > 0
        or sat_over_count > 0
        or gs_over_count > 0
    ):
        repair_solution(
            problem,
            pred_for_repair,
            assignment,
            trace,
            pop_load,
            sat_load,
            gs_load,
            deadline=repair_deadline,
            max_hops=args.max_hops,
        )
        objective = actual_objective(problem, assignment, trace.item_path_loss)
    repair_seconds = time.perf_counter() - repair_start

    local_start = time.perf_counter()
    isl_over_count, pop_over_count, sat_over_count, gs_over_count = count_violations(
        problem, trace, pop_load, sat_load, gs_load,
    )
    if (
        trace.invalid_count == 0
        and isl_over_count == 0
        and pop_over_count == 0
        and sat_over_count == 0
        and gs_over_count == 0
    ):
        local_deadline = min(
            global_deadline,
            local_start + max(0.0, args.local_search_budget),
        )
        local_search(
            problem,
            pred_for_repair,
            assignment,
            trace,
            pop_load,
            sat_load,
            gs_load,
            deadline=local_deadline,
            max_hops=args.max_hops,
        )
        objective = actual_objective(problem, assignment, trace.item_path_loss)
    local_search_seconds = time.perf_counter() - local_start
    total_seconds = time.perf_counter() - program_start

    return OptimizerResult(
        assignment=assignment,
        trace=trace,
        pop_load=pop_load,
        sat_load=sat_load,
        gs_load=gs_load,
        objective=objective,
        best_lower_bound=best_lb,
        iterations=iterations,
        total_seconds=total_seconds,
        solve_seconds=time.perf_counter() - solve_start,
        main_seconds=main_seconds,
        repair_seconds=repair_seconds,
        local_search_seconds=local_search_seconds,
        dijkstra_seconds=dijkstra_seconds,
        trace_seconds=trace_seconds,
        feasible_seen=feasible_seen,
    )


def print_summary(problem: ProblemData, result: OptimizerResult) -> None:
    """按题目要求打印求解结果。"""

    isl_over_count, pop_over_count, sat_over_count, gs_over_count = count_violations(
        problem, result.trace, result.pop_load, result.sat_load, result.gs_load,
    )
    if math.isfinite(result.objective) and math.isfinite(result.best_lower_bound):
        dual_gap = max(
            0.0,
            (result.objective - result.best_lower_bound)
            / max(abs(result.objective), 1.0)
            * 100.0,
        )
    else:
        dual_gap = math.nan

    avg_iter_ms = result.main_seconds / max(result.iterations, 1) * 1000.0
    dijkstra_pct = result.dijkstra_seconds / max(result.main_seconds, 1.0e-9) * 100.0
    trace_pct = result.trace_seconds / max(result.main_seconds, 1.0e-9) * 100.0
    flow_mean, demand_mean, valid_flows, demand_total = latency_means(
        problem, result.assignment, result.trace.item_path_loss,
    )

    print("\n==================== 求解结果 ====================")
    print(f"数据构建耗时:        {problem.setup_seconds:10.2f} 秒")
    print(f"总耗时:              {result.total_seconds:10.2f} 秒")
    print(f"求解阶段耗时:        {result.solve_seconds:10.2f} 秒")
    print(
        "阶段耗时:            "
        f"main={result.main_seconds:.2f}s, "
        f"repair={result.repair_seconds:.2f}s, "
        f"local={result.local_search_seconds:.2f}s"
    )
    print(f"迭代轮数:            {result.iterations:10d} 轮")
    print(f"最终总损耗:          {result.objective:10.2f}")
    print(f"flow 等权平均 RTT:   {flow_mean:10.2f} ms")
    print(f"demand 加权平均 RTT: {demand_mean:10.2f} ms")
    print(f"有效 flow 数:        {valid_flows:10d}")
    print(f"原始 demand:         {problem.raw_demand_gbps:10.2f} Gbps")
    print(f"可路由 demand:       {demand_total:10.2f} Gbps")
    print(
        "无 ingress 过滤:     "
        f"{problem.dropped_no_ingress_gbps:10.2f} Gbps / "
        f"{problem.dropped_no_ingress_items} items"
    )
    print(f"拉格朗日下界:        {result.best_lower_bound:10.2f}")
    print(f"对偶间隙:            {dual_gap:10.2f}%")
    print()
    print("问题规模:")
    print(f"  卫星节点数:        {problem.n_sats:10d}")
    print(f"  ISL 边数:          {len(problem.edge_cap):10d}")
    print(f"  active cell 数:    {problem.n_cells_total:10d}")
    print(f"  service 数:        {len(problem.service_names):10d}")
    print(f"  commodity 数:      {len(problem.weights):10d}")
    print(f"  PoP 数:            {len(problem.pop_names):10d}")
    print(f"  GS 数:             {len(problem.gs_names):10d}")
    print(f"  egress sat 数:     {len(problem.active_sat_ids):10d}")
    print(f"  候选路径数:        {len(problem.candidate_label):10d}")
    print()
    print("约束满足情况:")
    print(f"  ISL 超载条数:      {isl_over_count:4d} / {len(problem.edge_cap)}")
    print(f"  PoP 超容个数:      {pop_over_count:4d} / {len(problem.pop_names)}")
    print(f"  sat feeder 超载:   {sat_over_count:4d} / {len(problem.active_sat_ids)}")
    print(f"  GS feeder 超载:    {gs_over_count:4d} / {len(problem.gs_names)}")
    print(f"  未成功追踪路径:    {result.trace.invalid_count:4d} / {len(problem.weights)}")
    print()
    print("各 PoP 分配货量:")
    for idx, pop_code in enumerate(problem.pop_names):
        used = float(result.pop_load[idx])
        cap = float(problem.pop_cap[idx])
        util = used / cap * 100.0 if cap > 0 else math.inf
        print(f"  PoP {idx:2d} ({pop_code:>5s}): {used:9.2f} / {cap:9.2f} (容量利用率 {util:6.1f}%)")
    print()
    chosen = result.assignment[result.assignment >= 0]
    if chosen.size:
        unique_candidates = len(np.unique(chosen))
        unique_egress = len(np.unique(problem.candidate_sat[chosen]))
        unique_gs = len(np.unique(problem.candidate_gs_idx[chosen]))
    else:
        unique_candidates = unique_egress = unique_gs = 0
    print("实际使用资源:")
    print(f"  使用候选路径数:    {unique_candidates:10d}")
    print(f"  使用 egress sat:   {unique_egress:10d}")
    print(f"  使用 GS:           {unique_gs:10d}")
    print()
    print(f"平均每轮耗时:        {avg_iter_ms:10.1f} ms")
    print(f"Dijkstra 耗时占比:   {dijkstra_pct:10.1f}%")
    print(f"路径追踪耗时占比:    {trace_pct:10.1f}%")
    print("==================================================")


def print_optimal_certificate(cert: OptimalCertificateResult) -> None:
    """打印 top-k MIP certificate。"""

    print("\n==================== 最优性证书 ====================")
    print(f"候选窗口 top-k:       {cert.top_k:10d}")
    print(f"MIP 变量数:           {cert.n_vars:10d}")
    print(f"约束行数:             {cert.n_rows:10d}")
    print(f"约束非零元:           {cert.nnz:10d}")
    print(f"无候选 commodity:     {cert.skipped_items:10d}")
    print(f"补入 incumbent 候选:  {cert.forced_incumbent_vars:10d}")
    print(f"建模耗时:             {cert.build_seconds:10.2f} 秒")
    print(f"LP 求解耗时:          {cert.lp_seconds:10.2f} 秒")
    print(f"LP 松弛下界:          {cert.lp_fun if cert.lp_fun is not None else math.nan:10.2f}")
    print(f"LP success:           {str(cert.lp_success):>10s}")
    print(f"LP status:            {cert.lp_status:10d}")
    print(f"LP message:           {cert.lp_message}")
    if cert.mip_fun is None:
        print("MILP:                 未运行或未找到整数可行解")
        if cert.mip_status is not None:
            print(f"MILP status:          {cert.mip_status:10d}")
        if cert.mip_message:
            print(f"MILP message:         {cert.mip_message}")
    else:
        print(f"MILP 求解耗时:        {cert.mip_seconds:10.2f} 秒")
        print(f"MILP incumbent:       {cert.mip_fun:10.2f}")
        print(
            f"MILP dual bound:      "
            f"{cert.mip_dual_bound if cert.mip_dual_bound is not None else math.nan:10.2f}"
        )
        print(f"MILP gap:             {cert.mip_gap if cert.mip_gap is not None else math.nan:10.4f}")
        print(f"MILP nodes:           {cert.mip_nodes if cert.mip_nodes is not None else -1:10d}")
        print(f"MILP success:         {str(cert.mip_success):>10s}")
        print(f"MILP status:          {cert.mip_status if cert.mip_status is not None else -1:10d}")
        if cert.violations is not None:
            isl_over, pop_over, sat_over, gs_over = cert.violations
            print(
                "解码校验:             "
                f"ISL={isl_over}, PoP={pop_over}, sat={sat_over}, "
                f"GS={gs_over}, invalid={cert.invalid_count}"
            )
        if cert.mip_message:
            print(f"MILP message:         {cert.mip_message}")
    print("====================================================")


def print_result_brief(name: str, problem: ProblemData, result: OptimizerResult) -> None:
    """打印单个可行解的紧凑指标。"""

    isl_over, pop_over, sat_over, gs_over = count_violations(
        problem, result.trace, result.pop_load, result.sat_load, result.gs_load,
    )
    flow_mean, demand_mean, _valid_flows, _demand_total = latency_means(
        problem, result.assignment, result.trace.item_path_loss,
    )
    print(
        f"{name:>16s}: objective={result.objective:10.2f}, "
        f"flow_mean={flow_mean:7.2f} ms, "
        f"demand_mean={demand_mean:7.2f} ms, "
        f"time={result.solve_seconds:6.2f}s, "
        f"viol=(ISL {isl_over}, PoP {pop_over}, sat {sat_over}, GS {gs_over}), "
        f"invalid={result.trace.invalid_count}"
    )


def print_column_generation(problem: ProblemData, result: ColumnGenerationResult) -> None:
    """打印列生成 certificate。"""

    total_demand = float(problem.weights.astype(np.float64).sum())
    incumbent_demand_mean = result.incumbent_objective / total_demand
    restricted_demand_lb = result.lp_fun / total_demand
    corrected_demand_lb = result.corrected_lower_bound / total_demand
    print("\n==================== 列生成证书 ====================")
    print(f"迭代轮数:             {result.iterations:10d}")
    print(f"是否收敛:             {str(result.converged):>10s}")
    print(f"最终列数:             {result.final_columns:10d}")
    print(f"新增 path 列:         {result.extra_columns:10d}")
    print(f"累计新增列:           {result.added_columns:10d}")
    print(f"incumbent UB:         {result.incumbent_objective:10.2f}")
    print(f"incumbent demand mean:{incumbent_demand_mean:10.2f} ms")
    print(f"restricted LP:        {result.lp_fun:10.2f}")
    print(f"restricted demand LB: {restricted_demand_lb:10.2f} ms")
    print(f"修正 full-LP 下界:    {result.corrected_lower_bound:10.2f}")
    print(f"full-LP demand LB:    {corrected_demand_lb:10.2f} ms")
    print(f"最小 reduced cost:    {result.best_reduced_cost:10.4f}")
    print(f"保守 gap:             {result.gap_percent:10.2f}%")
    print(
        "阶段耗时:             "
        f"build={result.build_seconds:.2f}s, "
        f"LP={result.lp_seconds:.2f}s, "
        f"pricing={result.pricing_seconds:.2f}s, "
        f"total={result.total_seconds:.2f}s"
    )
    print(f"最后 LP success:      {str(result.last_lp_success):>10s}")
    print(f"最后 LP status:       {result.last_lp_status:10d}")
    print(f"最后 LP message:      {result.last_lp_message}")
    print("====================================================")


def main() -> None:
    args = parse_args()
    if not NUMBA_AVAILABLE:
        print(
            "提示: 未安装 numba，路径追踪使用纯 Python fallback。"
            "如需 JIT 加速可运行 `pip install numba`。"
        )
    if not JOBLIB_AVAILABLE:
        print(
            "提示: 未安装 joblib；若 scipy 多源 Dijkstra 单轮超过 20ms，"
            "本脚本不会切换到并行 fallback。"
        )

    problem = build_problem(args)
    print(
        "已读取项目数据: "
        f"{problem.n_sats} 颗卫星, {len(problem.edge_cap)} 条 ISL, "
        f"{problem.n_cells_total} 个 active cells, "
        f"{len(problem.service_names)} 个 services, "
        f"{len(problem.weights)} 个 (cell, service) commodities, "
        f"{len(problem.pop_names)} 个 PoP, "
        f"{len(problem.candidate_label)} 个 (PoP, GS, egress_sat) candidates。"
    )
    static_runner = DijkstraRunner(problem, static_paths=True)
    static_dist, static_pred, _ = static_runner.run(np.zeros(problem.edge_base_loss.shape[0]))
    baseline_start = time.perf_counter()
    baseline = nearest_pop_baseline_solution(
        problem,
        static_pred,
        static_dist,
        deadline=baseline_start + TOTAL_TIME_LIMIT,
        max_hops=args.max_hops,
        path_variants=max(1, int(args.isl_path_variants)),
    )
    print("\n==================== 快速对比 ====================")
    print_result_brief("Nearest baseline", problem, baseline)

    if args.find_optimal:
        solve_start = time.perf_counter()
        result = solve(
            problem,
            args,
            global_deadline=solve_start + TOTAL_TIME_LIMIT,
            program_start=solve_start,
        )
        print_result_brief("Optimizer", problem, result)
        print_summary(problem, result)
        cert = run_optimal_certificate(
            problem,
            args,
            incumbent_assignment=result.assignment,
        )
        print_optimal_certificate(cert)
        return

    solve_start = time.perf_counter()
    result = solve(
        problem,
        args,
        global_deadline=solve_start + TOTAL_TIME_LIMIT,
        program_start=solve_start,
    )
    print_result_brief("Optimizer", problem, result)
    print_summary(problem, result)
    if args.column_generation:
        cg_result = run_column_generation(problem, args, result)
        print_column_generation(problem, cg_result)


if __name__ == "__main__":
    main()
