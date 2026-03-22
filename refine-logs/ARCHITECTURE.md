# Vantage Simulator Architecture

## 1. Design Philosophy

Vantage simulates a centralized TE controller for LEO satellite ISPs. The architecture respects a fundamental tension: we are **simulating** a system that has a control plane and a data plane, but the simulator itself needs its own orchestration layer that is **above** both.

Design traditions drawn from:
- **Network simulator architecture** (ns-3, OMNeT++): clean model/engine separation, pluggable components, deterministic reproducibility
- **Production TE system architecture** (Footprint, Espresso, Edge Fabric): oracle/controller/agent decomposition, clear data contracts

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SIMULATION ENGINE                                   │
│  Orchestrates all subsystems. Drives time. Manages experiments.              │
│                                                                              │
│  ┌─ Epoch Loop ─────────────────────────────────────────────────────────┐    │
│  │                                                                       │    │
│  │  ┌─────────┐     ┌────────────┐     ┌──────────────┐                 │    │
│  │  │  Clock   │────▶│   World    │────▶│  Network     │                 │    │
│  │  │  (time t)│     │   Model    │     │  Snapshot(t) │                 │    │
│  │  └─────────┘     └────────────┘     └──────┬───────┘                 │    │
│  │                                            │                          │    │
│  │                  ┌─────────────────────────┼──────────────────┐       │    │
│  │                  │     CONTROL PLANE       │                  │       │    │
│  │                  │                         ▼                  │       │    │
│  │                  │  ┌──────────────────────────────────────┐  │       │    │
│  │                  │  │  Observation Model                   │  │       │    │
│  │                  │  │  (truth → noisy/delayed view)        │  │       │    │
│  │                  │  └──────────────┬───────────────────────┘  │       │    │
│  │  ┌──────────┐    │                │                           │       │    │
│  │  │ Workload │    │                ▼                           │       │    │
│  │  │ (demand) │───▶│  ┌──────────────────────────────────────┐  │       │    │
│  │  └──────────┘    │  │  TE Controller                       │  │       │    │
│  │                  │  │  (ControllerView, Demand, Capacity)  │  │       │    │
│  │                  │  │  → RoutingIntent                     │  │       │    │
│  │                  │  └──────────────┬───────────────────────┘  │       │    │
│  │                  └────────────────┼──────────────────────────┘       │    │
│  │                                   │                                   │    │
│  │                                   ▼                                   │    │
│  │  ┌────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Forwarding Engine                                             │   │    │
│  │  │  (RoutingIntent × NetworkSnapshot × Demand → RealizedState)   │   │    │
│  │  └───────────────────────────────┬────────────────────────────────┘   │    │
│  │                                  │                                    │    │
│  │                                  ▼                                    │    │
│  │  ┌────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Metrics Analyzer                                              │   │    │
│  │  │  (RealizedState → EpochMetrics)                                │   │    │
│  │  └───────────────────────────────┬────────────────────────────────┘   │    │
│  │                                  │                                    │    │
│  │                           ┌──────▼──────┐                            │    │
│  │                           │  Telemetry  │  (observer hooks)          │    │
│  │                           └─────────────┘                            │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 3. Dependency Model (DAG)

```
engine ──orchestrates──▶ { world, workload, control, forwarding, metrics, telemetry }

control ──reads──▶ world      (via ObservationModel: filtered/delayed/noisy view)
control ──reads──▶ workload   (via DemandEstimator: possibly aggregated demand)

forwarding ──reads──▶ world   (truth: actual topology, actual capacities)
forwarding ──reads──▶ control (RoutingIntent to realize)

metrics ──reads──▶ forwarding (RealizedState to analyze)

telemetry ──observes──▶ all   (passive hooks, injected by engine at setup)

world ──reads──▶ nothing      (pure model of physical reality)
workload ──reads──▶ nothing   (pure traffic generator)
```

**Key rules:**
- No subsystem calls "upward" to engine
- No subsystem calls into telemetry
- Control plane never modifies world state (read-only queries)
- All inter-subsystem data objects are immutable (frozen dataclasses)

## 4. Subsystem Specifications

### 4.1 World Model (`vantage/world/`)

The physical network being simulated. Pure functions of time and configuration. This IS the ground truth — never modified by control or forwarding.

| Component | Input | Output | Notes |
|-----------|-------|--------|-------|
| `ConstellationModel` | time t | `tuple[SatellitePosition, ...]` | SGP4 propagation, position cache in HDF5 |
| `TopologyBuilder` | satellite positions | `ISLGraph` | +Grid pattern, edges carry (delay_ms, capacity_gbps) |
| `GroundNetwork` | config | PoP/GS/peering static data | Loaded once, slowly-varying |
| `GroundDelayModel` | (pop, dest) | delay_ms | Protocol. Implementations: Measurement(L1), Regression(L2), FiberGraph(L3) |
| `SnapshotBuilder` | all above | `NetworkSnapshot` | Frozen aggregate at time t |

```python
@dataclass(frozen=True)
class NetworkSnapshot:
    """Complete physical network state at time t. Single source of truth."""
    epoch: int
    time_s: float
    satellites: tuple[SatellitePosition, ...]
    isl_graph: ISLGraph           # edges carry (delay_ms, capacity_gbps)
    ground_delays: GroundDelayMatrix    # (pop, dest) → delay_ms
    pops: tuple[PoPInfo, ...]     # includes peering capacity
```

**Extension point**: `FailureModel` (Protocol) — masks links/nodes in snapshot. v1 ships `NullFailureModel` (no failures).

### 4.2 Control Plane (`vantage/control/`)

The simulated TE system — this is what we are studying. Mirrors the architecture of Footprint/Espresso/Edge Fabric.

| Component | Real-world Analogue | Responsibility |
|-----------|---------------------|---------------|
| `ObservationModel` | Measurement infrastructure | Transforms `NetworkSnapshot` → `ControllerView`. Configurable: measurement noise (σ_ms), staleness (lag_s), aggregation level. |
| `DemandEstimator` | Traffic counters | Transforms `TrafficMatrix` → controller-visible demand estimate |
| `CapacityEstimator` | Capacity planning | Controller's view of available capacity (may differ from truth) |
| `TEController` | Footprint Mapping / Espresso GC | **Primary extension point.** Takes (view, demand, capacity) → `RoutingIntent` |

```python
class TEController(Protocol):
    """TE algorithm interface. All controllers implement this."""
    def optimize(
        self,
        view: ControllerView,
        demand: DemandEstimate,
        capacity: CapacityEstimate,
    ) -> RoutingIntent: ...
```

**RoutingIntent** — supports weighted multi-path allocation:

```python
@dataclass(frozen=True)
class FlowKey:
    """Identifies a traffic aggregate: (source region, destination prefix)."""
    src_region: str
    dst_prefix: str

@dataclass(frozen=True)
class PathAllocation:
    """One path option for a flow, with a weight indicating traffic split."""
    pop_id: str
    ingress_sat: str               # first satellite in the path
    isl_path: tuple[str, ...]      # ordered satellite IDs (empty for bent-pipe)
    egress_sat: str                # last satellite before GS
    weight: float                  # fraction of demand [0, 1]

@dataclass(frozen=True)
class RoutingIntent:
    """Controller output: how to route all flows this epoch."""
    epoch: int
    allocations: Mapping[FlowKey, tuple[PathAllocation, ...]]
    unserved: frozenset[FlowKey]   # explicitly rejected demand
    computed_at: float             # simulation time when computed
```

**Planned controller implementations:**

| Controller | Strategy | Purpose |
|-----------|----------|---------|
| `NearestPoP` | Nearest PoP geographically | Hot-potato baseline (current Starlink) |
| `SatelliteOnlyTE` | Optimize ISL path, nearest PoP | ISL-only TE baseline |
| `GroundOnlyOracle` | Best ground-delay PoP, no ISL opt | Ground-awareness isolation |
| `BestStaticPoP` | Per-dest static best PoP | Static optimization baseline |
| `LatencyOnlyTE` | Min E2E, ignore capacity | Upper bound on latency gain |
| `VantageGreedy` | Joint PoP+ISL, capacity-aware, greedy | **Primary algorithm** |
| `VantageLP` | LP on reduced topology | Optimality bound |

### 4.3 Forwarding Engine (`vantage/forwarding/`)

Realizes a `RoutingIntent` against network truth. This is "data plane execution" in simulation.

| Component | Responsibility |
|-----------|---------------|
| `FlowRealizer` | Takes (intent, snapshot, demand) → `RealizedState`. Computes actual delays through current topology, resolves capacity contention via proportional fairness, handles split traffic per-path. |
| `ActuationModel` | (Protocol) Delay between intent production and activation. v1: `InstantActuation`. |

```python
@dataclass(frozen=True)
class PathOutcome:
    """Realized outcome for one path of a flow."""
    pop_id: str
    isl_path: tuple[str, ...]
    sat_segment_ms: float
    ground_segment_ms: float
    e2e_delay_ms: float
    demand_served: float
    demand_dropped: float

@dataclass(frozen=True)
class FlowOutcome:
    """Realized outcome for a flow (may have multiple paths)."""
    flow_key: FlowKey
    paths: tuple[PathOutcome, ...]        # per-path outcomes
    total_served: float
    total_dropped: float
    weighted_e2e_ms: float                # demand-weighted average delay

@dataclass(frozen=True)
class RealizedState:
    """Complete realized network state after applying a RoutingIntent."""
    epoch: int
    flow_outcomes: Mapping[FlowKey, FlowOutcome]
    link_utilizations: Mapping[tuple[str, str], float]  # utilization ratio [0, 1+]
    congested_links: frozenset[tuple[str, str]]
    total_served_demand: float
    total_dropped_demand: float
```

**Capacity contention rule**: When multiple flows contend for the same bottleneck link, demand is allocated via **proportional fairness** (each flow gets capacity proportional to its requested share). This is explicit in the `FlowRealizer` contract.

### 4.4 Metrics Analyzer (`vantage/metrics/`)

Pure analytical functions over `RealizedState`. No simulation logic here.

| Component | Responsibility |
|-----------|---------------|
| `LatencyAnalyzer` | CDFs, mean/median/p95/p99, per-region/per-dest breakdown, satellite vs ground decomposition |
| `ThroughputAnalyzer` | Served-demand ratio, congestion-free scale, link utilization distribution |
| `Comparator` | Diff two `RealizedState`s (strategy A vs B), improvement percentages with bootstrap CIs |

**Note**: System-level metrics (controller solve time, memory) are measured by the engine via instrumentation, not by this subsystem. The metrics analyzer only processes `RealizedState`.

### 4.5 Workload (`vantage/workload/`)

| Component | Responsibility |
|-----------|---------------|
| `EndpointPopulation` | User regions (geo-coordinates, prefix sets) + destination definitions. Loaded from config/data files. |
| `TrafficGenerator` | (Protocol) Produces `TrafficMatrix` at each epoch. Implementations: `UniformGenerator`, `GravityModel`, `DiurnalGenerator`, `TraceReplay`. |

### 4.6 Telemetry (`vantage/telemetry/`)

Cross-cutting **observer** subsystem. Injected via hooks at engine setup, never called by other subsystems.

```python
class SimulationObserver(Protocol):
    """Observer hook called by the engine after each epoch."""
    def on_epoch(
        self,
        epoch: int,
        snapshot: NetworkSnapshot,
        intent: RoutingIntent,
        realized: RealizedState,
        metrics: EpochMetrics,
    ) -> None: ...

    def on_tick(
        self,
        tick: int,
        snapshot: NetworkSnapshot,
        realized: RealizedState,
    ) -> None: ...

    def on_complete(self, summary: SimulationSummary) -> None: ...
```

Implementations: `ParquetExporter`, `CSVExporter`, `FlowTraceRecorder`, `ProgressReporter`.

## 5. Time Model

### Two-timescale stepped simulation

| Timescale | What changes | Default period |
|-----------|-------------|----------------|
| **Topology tick** | Satellite positions, ISL delays, link availability | 15s |
| **Control epoch** | TE controller re-optimizes routing | 300s (5 min) |

```python
@dataclass(frozen=True)
class TimeConfig:
    topology_tick_s: float = 15.0
    control_epoch_s: float = 300.0
    simulation_duration_s: float = 86400.0  # 1 day
    seed: int = 42  # deterministic reproducibility
```

### Main Loop (engine/runner.py)

```python
def run(config: ExperimentConfig) -> SimulationSummary:
    # Setup: instantiate all subsystems from config
    world, workload, control, forwarding, metrics, observers = setup(config)

    for epoch in range(num_epochs):
        t_epoch = epoch * config.time.control_epoch_s

        # 1. Build truth snapshot
        snapshot = world.snapshot_at(t_epoch)

        # 2. Generate demand
        demand = workload.demand_at(t_epoch)

        # 3. Controller sees filtered truth (observation model adds noise/lag)
        view = control.observe(snapshot)
        demand_est = control.estimate_demand(demand)
        cap_est = control.estimate_capacity(snapshot)

        # 4. TE optimization (timed for system metrics)
        with timer() as solve_time:
            intent = control.controller.optimize(view, demand_est, cap_est)

        # 5. Realize across topology ticks within this epoch
        tick_results = []
        for tick_t in ticks_in_epoch(t_epoch, config.time):
            tick_snapshot = world.snapshot_at(tick_t)
            realized = forwarding.realize(intent, tick_snapshot, demand)
            tick_metrics = metrics.analyze(realized)
            tick_results.append((tick_t, realized, tick_metrics))
            for obs in observers:
                obs.on_tick(tick_t, tick_snapshot, realized)

        # 6. Aggregate epoch metrics
        epoch_metrics = metrics.aggregate(tick_results, solve_time)
        for obs in observers:
            obs.on_epoch(epoch, snapshot, intent, realized, epoch_metrics)

    summary = metrics.summarize()
    for obs in observers:
        obs.on_complete(summary)
    return summary
```

**Key insight**: The topology moves between control epochs. A policy computed at `t=0` is evaluated against topology at `t=15s, 30s, ...`. This drift is a source of suboptimality the simulator measures.

## 6. Extension Points Summary

All pluggable via `Protocol`:

| Protocol | Package | What it enables |
|----------|---------|----------------|
| `ConstellationModel` | world | Different propagation methods (SGP4, Kepler, cached) |
| `GroundDelayModel` | world | L1/L2/L3 ground delay estimation strategies |
| `FailureModel` | world | Link/node failure injection (v1: no failures) |
| `TEController` | control | **Primary research axis** — 7 implementations |
| `ObservationModel` | control | Measurement noise/lag for robustness study (E6) |
| `TrafficGenerator` | workload | Demand generation strategies |
| `ActuationModel` | forwarding | Policy deployment delay (v1: instant) |
| `SimulationObserver` | telemetry | Output hooks |

## 7. Package Structure

```
vantage/
├── pyproject.toml
├── configs/                         # YAML experiment configurations
│   ├── base.yaml                    # Default parameters
│   ├── small.yaml                   # Fast iteration (100 sats, 50 users)
│   └── full.yaml                    # Full scale (1584 sats, 1000+ users)
│
├── src/vantage/
│   ├── __init__.py
│   ├── cli.py                       # Entry: python -m vantage run configs/full.yaml
│   │
│   ├── model/                       # ── Shared immutable domain objects ──
│   │   ├── __init__.py
│   │   ├── network.py               # SatellitePosition, ISLGraph, ISLEdge, PoPInfo
│   │   ├── flow.py                  # FlowKey, TrafficMatrix, DemandEntry
│   │   ├── policy.py                # RoutingIntent, PathAllocation
│   │   ├── state.py                 # NetworkSnapshot, RealizedState, FlowOutcome, PathOutcome
│   │   └── result.py                # EpochMetrics, LatencySummary, SimulationSummary
│   │
│   ├── engine/                      # ── Simulation Engine (orchestrator) ──
│   │   ├── __init__.py
│   │   ├── runner.py                # Main epoch loop
│   │   ├── clock.py                 # SimulationClock, TimeConfig
│   │   └── config.py                # ExperimentConfig (Pydantic), YAML loader
│   │
│   ├── world/                       # ── World Model (physical truth) ──
│   │   ├── __init__.py
│   │   ├── constellation.py         # ConstellationModel protocol + SGP4 impl
│   │   ├── topology.py              # TopologyBuilder: positions → ISLGraph
│   │   ├── ground_network.py        # Static infra: PoPs, GS, peering links
│   │   ├── ground_delay.py          # GroundDelayModel protocol + L1/L2/L3 impls
│   │   ├── snapshot.py              # SnapshotBuilder: assembles NetworkSnapshot
│   │   └── failures.py              # FailureModel protocol + NullFailureModel
│   │
│   ├── control/                     # ── Control Plane (system under study) ──
│   │   ├── __init__.py
│   │   ├── observation.py           # ObservationModel: snapshot → ControllerView
│   │   ├── estimators.py            # DemandEstimator, CapacityEstimator
│   │   └── controllers/             # TE algorithms (primary extension point)
│   │       ├── __init__.py
│   │       ├── protocol.py          # TEController Protocol definition
│   │       ├── nearest_pop.py       # Hot-potato baseline
│   │       ├── satellite_only.py    # ISL-only TE baseline
│   │       ├── ground_only.py       # Ground-oracle baseline
│   │       ├── static_pop.py        # Best-static-PoP baseline
│   │       ├── latency_only.py      # Unconstrained joint baseline
│   │       ├── greedy.py            # Vantage-Greedy (primary algorithm)
│   │       └── lp.py                # Vantage-LP (optimality bound)
│   │
│   ├── forwarding/                  # ── Forwarding Engine (realization) ──
│   │   ├── __init__.py
│   │   ├── realizer.py              # FlowRealizer: intent × truth → RealizedState
│   │   └── actuation.py             # ActuationModel protocol + InstantActuation
│   │
│   ├── workload/                    # ── Workload Generation ──
│   │   ├── __init__.py
│   │   ├── population.py            # EndpointPopulation (users + destinations)
│   │   └── traffic.py               # TrafficGenerator protocol + implementations
│   │
│   ├── metrics/                     # ── Metrics Analysis (pure functions) ──
│   │   ├── __init__.py
│   │   ├── latency.py               # CDF, percentiles, segment decomposition
│   │   ├── throughput.py            # Served-demand ratio, congestion-free scale
│   │   └── comparator.py           # A vs B diff, bootstrap CIs
│   │
│   ├── telemetry/                   # ── Observer Subsystem (cross-cutting) ──
│   │   ├── __init__.py
│   │   ├── observer.py              # SimulationObserver protocol
│   │   ├── exporters.py             # Parquet, CSV, JSON exporters
│   │   └── recorder.py             # Flow-level trace recorder
│   │
│   └── data/                        # ── Data Loaders (infrastructure) ──
│       ├── __init__.py
│       ├── infrastructure.py        # GS, PoP, IXP, ITU fiber, cable parsers
│       ├── traceroute.py            # Ground truth (traceroute summary) loader
│       └── geolocation.py           # IP → (lat, lon, ASN)
│
├── experiments/                     # Reproducible experiment scripts
│   ├── e1_ground_validation.py
│   ├── e2_satellite_validation.py
│   ├── e3_latency_eval.py
│   ├── e3b_ablation.py
│   ├── e3c_churn.py
│   ├── e4_throughput.py
│   ├── e5_benchmarks.py
│   └── e6_robustness.py
│
├── tests/
│   ├── conftest.py                  # Shared fixtures (toy 24-sat constellation)
│   ├── unit/
│   │   ├── test_topology.py
│   │   ├── test_ground_delay.py
│   │   ├── test_controllers.py
│   │   ├── test_realizer.py
│   │   └── test_metrics.py
│   └── integration/
│       ├── test_epoch_loop.py       # Full epoch: world → control → forward → metrics
│       └── test_controller_swap.py  # Same scenario, different controllers
│
└── artifacts/                       # Generated outputs (gitignored)
    ├── cache/                       # HDF5 caches (positions, delay matrices)
    ├── results/                     # Experiment results (Parquet/CSV)
    └── figures/                     # Generated plots
```

## 8. Anti-Patterns Explicitly Rejected

1. **Packet-level simulation** — Vantage is a flow/prefix-level TE simulator, not ns-3
2. **God network class** — state split across `NetworkSnapshot` (truth), `RoutingIntent` (decision), `RealizedState` (outcome)
3. **Mutable shared state** — all inter-subsystem objects are frozen dataclasses
4. **Controller-evaluator coupling** — forwarding engine is independent from controller; same evaluator scores all controllers
5. **Omniscient controller** — `ObservationModel` interposes between truth and controller, enabling noise/lag simulation
6. **Flat file layout** — packages mirror architectural subsystems exactly
7. **Hardcoded parameters** — all config via YAML + Pydantic validation

## 9. Design Decisions (ADRs)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Stepped discrete-time, not DES | TE operates on flows (5-min epochs), not packets. Simpler, deterministic, reproducible. Same model as Footprint. |
| 2 | Protocol over ABC | Duck typing, more Pythonic. Any class with right methods works. |
| 3 | Immutable policy/state objects | Prevents shared-mutation bugs. Safe to store history. Functional-core design. |
| 4 | ObservationModel separate from Controller | Enables ablation: same controller with perfect vs noisy observations. Key for E6 robustness study. |
| 5 | ForwardingEngine separate from World | Evaluator = ground truth. Must be independent from system under test. |
| 6 | Proportional fairness for contention | Explicit, well-understood rule. Avoids hidden assumptions in controller comparison. |
| 7 | Multi-path weighted allocations | Supports LP solutions with flow splitting. Single-path is just weight=1.0. |
| 8 | System metrics via engine instrumentation | Controller solve time is not derivable from RealizedState. Engine wraps optimization calls with timers. |
| 9 | FailureModel/ActuationModel as v1 no-ops | Clean extension points without over-engineering. NullFailureModel + InstantActuation ship first. |
| 10 | Python-first, Rust only if profiling shows >10min/epoch | Avoid premature optimization. Most computation is graph shortest-path and matrix ops (numpy). |

## 10. How the Architecture Answers the Research Question

The central research question: *Does ground-segment awareness improve satellite TE?*

The architecture answers this through **controller swapping**:

```yaml
# Same world, same workload, same forwarding engine, same metrics
# Only the controller changes — one config line

# Baseline: ignores ground delays
controller: nearest_pop

# vs. Vantage: ground-aware joint optimization
controller: greedy
```

Because `ObservationModel`, `ForwardingEngine`, and `MetricsAnalyzer` are shared across all experiments, the comparison is fair by construction. The ablation study (E3b) works by giving the controller a `GroundDelayModel` that returns zeros — same interface, different implementation.

This is the mark of a well-designed simulator: the research question maps to a config change, not a code change.
