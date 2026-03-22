# Implementation Plan

Based on ARCHITECTURE_V2.md. Phases are ordered by dependency — each phase builds on the previous.

## Phase 0: Data Preprocessing + Project Restructure

**Goal**: Pre-process raw data into clean JSON. Restructure files to match v2 layout.

**Tasks**:
1. Write `config/preprocess.py`:
   - Ground stations: raw GeoJSON → `data/processed/ground_stations.json`
   - PoPs: filter folder="PoPs & Backbone", normalize names → `data/processed/pops.json`
   - GS↔PoP edges: derive from raw data → `data/processed/gs_pop_edges.json`
   - Fiber segments: extract from ITU GeoJSON → `data/processed/fiber_segments.json`
   - Write `manifest.json` with source hashes, counts, schema version
2. Move existing code into v2 structure:
   - `world/constellation.py`, `world/topology.py`, `world/routing.py` → `world/satellite/`
   - `world/access.py` stays
   - `data/infrastructure.py` parsing logic → `config/preprocess.py`
   - `model/network.py` → `model.py` (rename + consolidate)
3. Delete old `data/infrastructure.py`, `data/traceroute.py`, `data/_io.py`
4. Write `world/ground.py`:
   - `GroundInfrastructure.__init__(data_dir)`: loads processed JSON with trivial `PoP(**entry)` mapping
   - Properties: `pops`, `ground_stations`, `gs_serving_pop(code)`, `backhaul_delay(gs_id, pop_code)`
5. Update all imports, verify 80 existing tests pass (satellite/topology/routing/access tests unchanged)
6. Write new tests for `GroundInfrastructure` loading processed JSON

**Output**: Clean project structure, static data pipeline working, all existing tests green.

---

## Phase 1: SatelliteSegment Facade + Routing Enhancement

**Goal**: Wrap satellite modules into a single facade. Add predecessor_matrix for path reconstruction.

**Tasks**:
1. Enhance `world/satellite/routing.py`:
   - `DijkstraRouting.compute_all_pairs()` returns `RoutingResult` with both `delay_matrix` AND `predecessor_matrix` (int32)
   - Add `reconstruct_path(predecessor_matrix, src, dst) → tuple[int, ...]` utility
   - Add ISL capacity_gbps to `ISLEdge` (default 20 gbps)
2. Define `SatelliteState` in `model.py`:
   - `timeslot, positions, graph, delay_matrix, predecessor_matrix`
3. Write `world/satellite/__init__.py` with `SatelliteSegment` facade:
   - `__init__(constellation, topology_builder, routing_strategy, shell_id)`
   - `state_at(timeslot) → SatelliteState` (internally: positions → graph → routing)
   - Shell internalized
4. Tests: SatelliteSegment facade tests, predecessor_matrix tests, path reconstruction tests

**Output**: `SatelliteSegment.state_at(t) → SatelliteState` working and tested.

---

## Phase 2: WorldModel + NetworkSnapshot

**Goal**: Compose satellite + ground + access into WorldModel that produces frozen snapshots.

**Tasks**:
1. Define in `model.py`:
   - `GatewayAttachments`: per-GS → top-k AccessLinks
   - `InfrastructureView`: frozen pops, gs, gs_pop_edges
   - `NetworkSnapshot`: epoch, time_s, satellite, gateway_attachments, infra
2. Write `world/world.py`:
   - `WorldModel.__init__(satellite_segment, ground_infra, access_model, ground_delay_model)`
   - `snapshot_at(time_s) → NetworkSnapshot`
   - Internally: compute satellite state, compute gateway attachments (top-k visible sats per GS), freeze infrastructure view
3. Add `GroundDelayModel` Protocol to `world/ground.py`:
   - L1 `MeasurementDelay`: traceroute lookup table
   - L2 `RegressionDelay`: haversine distance regression
   - (L3 `FiberGraphDelay` deferred to later)
4. Tests: WorldModel produces valid snapshots, gateway attachments computed correctly

**Output**: `world.snapshot_at(t) → NetworkSnapshot` working.

---

## Phase 3: Traffic Generation

**Goal**: Generate traffic demand per epoch.

**Tasks**:
1. Define in `model.py`:
   - `FlowKey(src_region, dst_prefix)`
   - `TrafficDemand`: mapping FlowKey → demand_gbps
2. Write `traffic.py`:
   - `TrafficGenerator` Protocol: `generate(t) → TrafficDemand`
   - `UniformGenerator`: uniform demand across all src-dst pairs
   - `GravityModel`: demand proportional to population × distance
3. Define endpoint population from probe data + destination prefixes
4. Tests: generators produce valid demand, correct FlowKey coverage

**Output**: `traffic.generate(t) → TrafficDemand` working.

---

## Phase 4: Control Plane — Baselines

**Goal**: Implement TEController Protocol + first 3 baseline controllers.

**Tasks**:
1. Define in `model.py`:
   - `PathAllocation(pop_code, isl_path_hint, weight)`
   - `RoutingIntent(epoch, allocations: Mapping[FlowKey, tuple[PathAllocation, ...]])`
2. Write `control/controller.py`:
   - `TEController` Protocol: `optimize(snapshot, demand) → RoutingIntent`
   - Factory function: `create_controller(name, config) → TEController`
3. Implement baselines in `control/policy/`:
   - `nearest_pop.py`: for each flow, pick geographically nearest PoP
   - `satellite_only.py`: pick PoP that minimizes ISL delay (ignore ground)
   - `ground_only.py`: pick PoP that minimizes ground delay (oracle)
4. Tests: each controller returns valid RoutingIntent, correct PoP selection logic

**Output**: 3 controllers working, all implementing same Protocol.

---

## Phase 5: Data Plane (forward.py)

**Goal**: Realize RoutingIntent against truth, compute E2E delays and utilization.

**Tasks**:
1. Define in `model.py`:
   - `ConcreteRoute(user_sat, uplink_delay_s, isl_path, isl_delay_s, egress_sat, downlink_delay_s, gs_id, backhaul_delay_s, pop_code, ground_delay_s)`
   - `E2EDelay(uplink_s, isl_s, downlink_s, backhaul_s, ground_s, total_s)`
   - `FlowOutcome(flow_key, route, e2e_delay, demand_served, demand_dropped)`
   - `EpochResult(epoch, flow_outcomes, link_utilizations, total_served, total_dropped)`
2. Write `forward.py`:
   - `realize(intent, snapshot, demand, ground_delay_model) → EpochResult`
   - Per flow: resolve user→sat, GS→sat, reconstruct ISL path, sum delays
   - Track utilization across ISL, gateway, backhaul, peering
   - Proportional fairness at bottlenecks
3. Tests: E2E delay decomposition correctness, capacity contention, proportional fairness

**Output**: Full epoch pipeline working end-to-end.

---

## Phase 6: Metrics + Engine

**Goal**: Wire everything together. Run full experiments.

**Tasks**:
1. Write `metrics.py`:
   - Latency CDF, mean/median/p95/p99
   - Per-region and per-destination breakdown
   - Segment decomposition (satellite vs ground)
   - Controller A vs B comparison with improvement %
2. Write `engine.py`:
   - Epoch loop: world → traffic → controller → forwarder → metrics
   - Config loading (YAML → Pydantic)
   - Entry point: `uv run python -m vantage.engine --config ...`
3. Integration test: full epoch loop with toy constellation + uniform traffic + nearest_pop controller
4. First experiment: E1 ground delay validation (compare L1/L2 against traceroute ground truth)

**Output**: Simulator runs end-to-end. First experiment results.

---

## Phase 7: Remaining Controllers + Experiments

**Tasks**:
1. Implement remaining controllers:
   - `static_pop.py`, `latency_only.py`
   - `greedy.py` (Vantage-Greedy, primary algorithm)
   - `lp.py` (Vantage-LP, optimality bound)
2. Run experiments E2-E6
3. Generate paper figures

---

## Summary

| Phase | Modules | Depends on | Key deliverable |
|-------|---------|-----------|-----------------|
| 0 | config/, model.py, world/ground.py | — | Clean structure + static data |
| 1 | world/satellite/ | Phase 0 | SatelliteSegment facade |
| 2 | world/world.py, model.py | Phase 0+1 | NetworkSnapshot |
| 3 | traffic.py | Phase 0 | TrafficDemand |
| 4 | control/ (3 baselines) | Phase 2 | RoutingIntent |
| 5 | forward.py | Phase 2+3+4 | EpochResult |
| 6 | metrics.py, engine.py | Phase 5 | End-to-end pipeline |
| 7 | control/policy/ (remaining) | Phase 6 | Full experiments |
