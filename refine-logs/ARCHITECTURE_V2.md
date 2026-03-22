# Vantage Simulator Architecture v2

> Reviewed by GPT-5.4 (xhigh reasoning), 2026-03-21. Final version after user refinement.

## 1. Overview

Vantage is a satellite network E2E traffic engineering simulator. Core thesis: satellite ISPs only optimize ISL routing, ignoring ground segment delay. Vantage jointly optimizes PoP selection + ISL routing.

The simulator is an **epoch-based policy comparison testbed**: same world, same traffic, swap controller — compare results.

## 2. Epoch Loop

```python
for epoch in range(num_epochs):
    t = epoch * epoch_interval_s
    snapshot = world.snapshot_at(t)                        # World: physical truth
    demand   = traffic.generate(t)                         # Traffic: flow demand
    intent   = controller.optimize(snapshot, demand)       # Control plane: routing decision
    result   = forwarder.realize(intent, snapshot, demand)  # Data plane: execution
    metrics.record(epoch, result)                          # Metrics: aggregation
```

## 3. Package Structure

```
src/vantage/
├── model.py                    # All domain types (frozen dataclasses)
├── world/                      # Physical truth
│   ├── satellite/
│   │   ├── constellation.py    # SGP4 position computation
│   │   ├── topology.py         # ISL graph construction
│   │   └── routing.py          # Dijkstra shortest path + predecessor matrix
│   ├── ground.py               # GroundInfrastructure + GroundDelayModel
│   ├── access.py               # AccessModel: ground↔satellite links
│   └── world.py                # WorldModel → snapshot_at(t)
├── traffic.py                  # TrafficGenerator Protocol + implementations
├── control/                    # Control plane
│   ├── controller.py           # TEController Protocol + execution logic
│   └── policy/                 # TE algorithms (primary research axis)
│       ├── nearest_pop.py      # Baseline: hot-potato, nearest PoP
│       ├── satellite_only.py   # Baseline: optimize ISL only
│       ├── ground_only.py      # Baseline: ground-delay oracle
│       ├── static_pop.py       # Baseline: best static PoP per dest
│       ├── latency_only.py     # Baseline: min E2E, ignore capacity
│       ├── greedy.py           # Vantage-Greedy (primary algorithm)
│       └── lp.py               # Vantage-LP (optimality bound)
├── forward.py                  # Data plane: realize intent → Result
├── metrics.py                  # CDF, percentile, comparison tables
├── engine.py                   # Epoch loop orchestrator
└── config/
    └── preprocess.py           # Raw data → processed JSON (offline, run once)

data/
├── raw/                        # Original messy data (GeoJSON, CSV)
├── processed/                  # Clean JSON (checked into git)
│   ├── manifest.json           # Source hashes, schema version, counts
│   ├── ground_stations.json
│   ├── pops.json
│   ├── gs_pop_edges.json       # GS↔PoP bipartite graph (many-to-many)
│   └── fiber_segments.json     # For L3 GroundDelayModel
└── validation/                 # Traceroute ground truth (experiments only)
```

## 4. Module Responsibilities

### model.py — Domain Types

All frozen dataclasses in one file. No logic, no I/O.

```python
# Physical entities
SatellitePosition, ShellConfig, ConstellationConfig
GroundStation, PoP, GSPoPEdge, GeoCoordinate

# Topology
ISLEdge, ISLGraph, AccessLink

# Satellite state (frozen per-epoch snapshot)
SatelliteState:  positions, graph, delay_matrix, predecessor_matrix

# Ground state
InfrastructureView:  pops, ground_stations, gs_pop_edges
GatewayAttachments:  per-GS → top-k visible satellites

# Full snapshot
NetworkSnapshot:  epoch, time_s, satellite, gateway_attachments, infra

# Traffic
FlowKey, TrafficDemand

# Control plane output
RoutingIntent, PathAllocation

# Data plane output
ConcreteRoute, E2EDelay, FlowOutcome, EpochResult
```

### world/ — Physical Truth

**satellite/constellation.py**: ConstellationModel Protocol + XMLConstellationModel (SGP4/Skyfield).
**satellite/topology.py**: TopologyBuilder Protocol + PlusGridTopology. Edges carry delay_s and capacity_gbps.
**satellite/routing.py**: RoutingStrategy Protocol + DijkstraRouting. Returns delay_matrix + predecessor_matrix (compact int32, reconstruct paths on demand).

**ground.py**:
- `GroundInfrastructure`: loads pre-processed JSON (pops, ground stations, GS↔PoP edges). Trivial `json.load()` → dataclass, no parsing logic.
- `GroundDelayModel` Protocol + implementations: L1 (traceroute lookup), L2 (haversine regression), L3 (fiber graph shortest path).

**access.py**: AccessModel Protocol + SphericalAccessModel. Computes slant range, elevation, delay for any ground point ↔ satellite pair. Used for both user uplinks and GS downlinks (with GS-specific min_elevation).

**world.py**: WorldModel class. Composes satellite segment + ground + access into `NetworkSnapshot`. Handles time→timeslot conversion. Shell internalized (not in public API).

### traffic.py — Demand Generation

TrafficGenerator Protocol. `generate(t) → TrafficDemand`.
Implementations: UniformGenerator, GravityModel, DiurnalGenerator.
TrafficDemand is a mapping of FlowKey → demand_gbps.

### control/ — Control Plane

**controller.py**:
```python
class TEController(Protocol):
    def optimize(self, snapshot: NetworkSnapshot, demand: TrafficDemand) -> RoutingIntent: ...
```

Controller sees snapshot directly (no ObservationModel in v1). E6 robustness study can add a NoisyWrapper later.

**policy/**: 7 implementations. Each is a separate file because each algorithm has distinct logic. All implement TEController Protocol.

RoutingIntent specifies PoP selection per flow (and optionally ISL path hints). GS selection and attachment are resolved by forward.py.

### forward.py — Data Plane

```python
def realize(intent: RoutingIntent, snapshot: NetworkSnapshot, demand: TrafficDemand) -> EpochResult
```

For each flow:
1. Resolve user → ingress satellite (from snapshot positions)
2. Look up PoP from intent, find GS serving that PoP (from gs_pop_edges)
3. Resolve GS → egress satellite (from gateway_attachments)
4. Reconstruct ISL path (from predecessor_matrix)
5. Sum delays: uplink + ISL + downlink + backhaul + ground
6. Track link utilization across all layers
7. Apply proportional fairness at bottlenecks

Output: EpochResult with per-flow outcomes, link utilizations, congestion info.

### metrics.py — Analysis

Pure functions over EpochResult sequences.
- Latency: CDF, mean/median/p95/p99, per-region breakdown, segment decomposition
- Throughput: served-demand ratio, congestion-free scale
- Comparison: strategy A vs B diff, improvement percentages

### engine.py — Orchestrator

Epoch loop. Instantiates all modules from config. Drives time. Collects results.

Entry point: `uv run python -m vantage.engine --config configs/experiment.yaml`

## 5. E2E Path Model

```
User → Sat_ingress → (ISL hops) → Sat_egress → GS → (backhaul) → PoP → (ground) → Dest
  ①         ②              ③          ④           ⑤
```

| Segment | Computed by | Data source |
|---------|------------|-------------|
| ① Uplink | forward.py | AccessModel + snapshot.positions |
| ② ISL | forward.py | predecessor_matrix → path → delay |
| ③ Downlink | forward.py | gateway_attachments (top-k per GS) |
| ④ Backhaul | forward.py | gs_pop_edges (static) |
| ⑤ Ground | forward.py | GroundDelayModel (truth oracle) |

## 6. Static Data Pipeline

```
Raw data (GeoJSON/CSV) → config/preprocess.py (run once) → data/processed/ (clean JSON + manifest)
```

Simulator only loads `data/processed/`. Complex parsing lives only in `preprocess.py`.

Manifest tracks source hashes and schema version for reproducibility.

## 7. Design Rules

1. **model.py is pure data** — frozen dataclasses only, no logic
2. **Static data is pre-processed** — no runtime parsing
3. **Controller sees snapshot directly** — v1 has no ObservationModel
4. **RoutingIntent = PoP selection** — GS/attachment resolved by forward.py
5. **GS↔PoP is many-to-many** — static bipartite graph, dynamic GS selection
6. **Gateway attachments are top-k** — not best-1
7. **Compact routing state** — predecessor_matrix (int32), paths reconstructed on demand
8. **Proportional fairness** across all capacity layers
9. **Shell internalized** — not in public API

## 8. Capacity Model

| Link type | Capacity | Direction | Contention |
|-----------|----------|-----------|------------|
| ISL | gbps per edge | Undirected | Shared across flows |
| Gateway (GS↔sat) | gbps per GS | Aggregate | Shared up+down |
| Backhaul (GS↔PoP) | gbps per edge | Undirected | Shared |
| Peering (PoP) | gbps per PoP | Aggregate | Shared across dests |

Bottleneck: flow gets `min(available at each link on its path)`.
