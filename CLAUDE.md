# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync                                          # install deps
uv run python -m vantage.config.preprocess        # regenerate data/processed/ from raw data
uv run pytest tests/                              # all tests
uv run pytest tests/unit/test_ground.py::test_name # single test
uv run pytest tests/ --cov=src/vantage --cov-report=term-missing  # coverage
uv run ruff check src/ tests/                     # lint
uv run mypy src/                                  # type check
```

## Architecture

Vantage is a **satellite network TE ground control system** that jointly optimizes PoP selection + ISL routing. The core thesis: satellite ISPs (like Starlink) only optimize ISL routing, ignoring ground segment delay. Vantage proves that ground-awareness improves E2E latency.

### Epoch Loop (`engine/run.py`)

```
for each epoch:
  snapshot = world.snapshot_at(t)           # physical truth
  demand   = traffic.generate(epoch)        # flow demands
  intent   = controller.optimize(snapshot, demand)  # PoP + path selection
  result   = realize(intent, snapshot, ...)  # compute actual E2E delays
  feedback.observe(result)                  # update ground knowledge
```

### Layer Separation

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **World (truth)** | `world/` | Satellite positions, ISL topology, routing matrices, ground infrastructure, access links |
| **Policy (decision)** | `policy/` | CandidateBasedController + scorers decide path allocations |
| **Forward (execution)** | `forward.py` | Realizes resolved paths → computes actual segment delays |
| **Feedback** | `engine_feedback.py` | Writes realized ground delays back to GroundKnowledge |
| **Analysis** | `analysis/` | Pure functions over EpochResult sequences |
| **Traffic (demand)** | `traffic/` | TrafficGenerator implementations produce per-epoch demands |
| **Domain (types)** | `domain/` | Frozen dataclasses: PathAllocation, FlowKey, NetworkSnapshot, etc. |
| **Common** | `common/` | Physical constants, geographic utilities |

### E2E Delay Decomposition

```
User → (uplink) → Sat_ingress → (ISL hops) → Sat_egress → (downlink) → GS → (backhaul) → PoP → (ground) → Destination
```

All delay values in the system are **RTT** (round-trip time). Never divide observed delays by 2.

### Controller = Research Variable

All controllers implement `TEController` Protocol with `optimize(snapshot, demand) → RoutingIntent`. Controller outputs **fully resolved paths** (`PathAllocation` with pop_code, gs_id, user_sat, egress_sat). The forwarding engine only computes delays — no search.

Available controllers (via `create_controller(name)`):
- `nearest_pop` — baseline: geographically nearest PoP
- `ground_only` — oracle: ground-delay aware, no ISL optimization
- `static_pop` — precomputed best static PoP per destination
- `latency_only` — minimize E2E latency, ignore capacity
- `greedy` — **primary algorithm**: joint E2E optimization (ISL + downlink + backhaul + ground)

### Satellite Subsystem (`world/satellite/`)

Protocol-based composition: `ConstellationModel` → positions, `TopologyBuilder` → ISL graph (+Grid), `RoutingStrategy` → delay_matrix + predecessor_matrix (Dijkstra). `SatelliteSegment` is the facade. `SatelliteDelayCalibration` uses delta method: `calibrated = measured + (est_new - est_old)`.

### Ground Subsystem (`world/ground/`)

- `GroundInfrastructure` — loads pre-processed JSON (validates manifest schema_version)
- `GroundKnowledge` — unified ground delay service: L1 cache + L2/L3 estimator fallback. Single source of truth for all ground delay queries (controllers, forward, engine feedback all depend on this)
- `HaversineDelay` (L2) — distance × detour_factor / fiber_speed
- `FiberGraphDelay` (L3) — Dijkstra on ITU terrestrial + submarine cable graph

### Static Data Pipeline

Raw data (GeoJSON/CSV) → `config/preprocess.py` (run once) → `data/processed/*.json` + `manifest.json`. The system never parses raw data at runtime. Outputs: ground_stations.json, pops.json, gs_pop_edges.json (nearest-1), fiber_graph.json, terminals.json. Orphan PoPs are connected to their nearest GS.

## Key Design Rules

- All domain types are **frozen dataclasses** (`frozen=True, slots=True`) in `domain/`
- **Protocol** over ABC — duck typing, any class with right methods works
- `MappingProxyType` for runtime immutability of dict fields in domain types
- Controller comparison is fair by construction: same world/traffic/forward/metrics, only controller swaps
- Predecessor matrix (`int32`) for memory-efficient path reconstruction — no full path storage
- `SupportsGroundFeedback` Protocol for feedback (engine writes to context.ground_knowledge)
- `CandidateBasedController` base class: enumerate candidates → score → select best
- `CandidateScorer` Protocol: new strategies mainly require a new scorer implementation
- `GroundKnowledge` is the single source of truth for ground delays (L1 cache + L2/L3 estimator)
- Physical constants centralized in `common/constants.py`
- `SatelliteSegment.state_at()` caches results per timeslot
- `RoutingComputer` Protocol allows plugging alternative routing backends
- Design code, name, module from the Starlink's prospective and production system
- Only run affected test unit after modify code.
