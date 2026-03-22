# Vantage Experiment Plan

## Claim-to-Experiment Mapping

| Claim | Experiment | Priority |
|-------|-----------|----------|
| C1: Ground segment dominates E2E latency | E0: Data analysis (already done in notebook) | Done |
| C2: Simulator ground delay matches ground truth | E1: Ground delay model validation | P0 |
| C3: Simulator satellite delay is accurate | E2: Satellite delay model validation | P0 |
| C4: Vantage reduces E2E latency | E3: Latency optimization evaluation | P0 |
| C4b: Ground awareness is the key driver | E3b: Ablation study (component isolation) | P0 |
| C4c: Assignments are stable across epochs | E3c: Temporal stability & churn analysis | P1 |
| C5: Vantage improves throughput | E4: Throughput/capacity evaluation | P0 |
| C6: System scales to realistic sizes | E5: System performance benchmarks | P1 |
| C7: Vantage is robust to model errors | E6: Robustness & sensitivity study | P1 |

---

## Phase 1: Simulator Foundation (Weeks 1-4)

### Module 1: Data Layer (`vantage/simulator/data/`)

**Goal**: Load, parse, and index all infrastructure data.

**Tasks:**
1. **Infrastructure loader**
   - Parse Starlink GS locations (`starlink_gs.json`)
   - Parse PoP locations and IP prefixes (`starlink_pops.csv`, `starlink_pop.geojson`)
   - Parse peering data (`starlink_peeringdb.json`, `google_peeringdb.json`, `wiki_peeringdb.json`)
   - Parse ITU fiber network (`ground_fibre.json`) — build NetworkX graph
   - Parse IXP locations (`exchanges.json`, `ixp.json`)
   - Parse submarine cables (`cable-geo.json`, `landing-point-geo.json`)

2. **Traceroute ground truth loader**
   - Parse summary files (`google_summary.json`, `facebook_summary.json`, `wikipedia_summary.json`)
   - Extract (probe_id, pop, destination) → {starlink_segment, ground_segment, total_rtt}
   - Build lookup table: (pop_code, destination) → median_ground_delay_ms

3. **Geolocation service**
   - Load MaxMind GeoLite2 databases
   - Load IP geolocation CSV
   - Provide IP → (lat, lon, ASN) lookup

**Validation gate**: All data loads correctly, traceroute lookup returns expected values for known probes.

### Module 2: Satellite Network Model (`vantage/simulator/space/`)

**Goal**: Model Starlink constellation, compute satellite positions, ISL delays, and user-satellite-PoP paths.

**Tasks:**
1. **Constellation model**
   - SGP4 orbital propagation (borrow approach from StarryNet's `sn_observer.py`)
   - Starlink shell 1: 550km altitude, 53° inclination, 72 planes × 22 sats = 1584 sats
   - Configurable: allow smaller constellations for fast testing
   - Cache satellite positions per timeslot in HDF5

2. **ISL topology**
   - +Grid connectivity: 4 ISLs per satellite (2 intra-orbit, 2 inter-orbit)
   - ISL delay: `distance_km / c_vacuum * 1000` (ms)
   - Delay matrix per timeslot, stored in HDF5
   - Exception: no inter-orbit ISL at high latitudes (>75°)

3. **Ground-to-Space links (GSL)**
   - User → nearest visible satellite (elevation angle ≥ 25°)
   - Satellite → Ground Station (GS) → PoP mapping
   - Access delay: 3D slant range / c_vacuum
   - Support multiple GS per PoP

4. **Satellite segment delay calculator**
   - Given (user_location, pop): compute RTT_sat
   - Path: user → sat_up → ISL shortest path → sat_down → GS → PoP
   - Dijkstra on ISL delay graph
   - Also support bent-pipe (no ISL): user → sat → GS → PoP directly

**Validation gate (E2)**: Compare RTT_sat predictions against:
- StarPerf's XML/TLE delay estimates (cross-validation)
- Measured `avg_starlink_segment` from traceroute summaries (ground truth)
- Target: correlation R² > 0.8, MAPE < 25%

### Module 3: Ground Delay Model (`vantage/simulator/ground/`)

**Goal**: Estimate RTT_ground(pop, destination) for any (pop, destination) pair.

**Tasks:**
1. **L1: Ground truth lookup**
   - For known (pop, dest) pairs: return median from traceroute data
   - Coverage: ~49 PoPs × 3 destinations ≈ 147 pairs (some missing)

2. **L2: Regression model**
   - Features: haversine_distance(pop, dest), continent_match, IXP_proximity
   - Train on traceroute ground truth, evaluate with leave-one-out CV
   - Fallback for missing (pop, dest) pairs

3. **L3: Fiber graph model** (if L2 insufficient)
   - Build weighted graph from ITU fiber data
   - Edge weight: fiber distance × (c/c_fiber) where c_fiber ≈ 2c/3
   - Add routing inflation factor (calibrated from ground truth)
   - Optional: incorporate CAIDA BGP AS-path data for more realistic routing

4. **Ground delay update mechanism**
   - Ground delays change slowly (hours/days), not per-timeslot
   - Controller queries Ground Delay Oracle at configurable intervals
   - Oracle returns RTT_ground(pop, dest) matrix

**Validation gate (E1)**:
- Leave-one-out cross-validation on traceroute ground truth
- L1 accuracy: exact match (by definition)
- L2 accuracy target: MAPE < 20%, R² > 0.7
- Compare prediction vs actual for each (pop, dest) pair

---

## Phase 2: Traffic & Optimization (Weeks 5-8)

### Module 4: Flow Generator (`vantage/simulator/traffic/`)

**Goal**: Generate realistic traffic demand matrices.

**Tasks:**
1. **User distribution model**
   - Use Starlink coverage map + probe locations as seeds
   - Distribute users proportional to Starlink subscriber density by country
   - Configurable: number of user groups, geographic distribution

2. **Destination model**
   - Major destinations: Google, Facebook, Wikipedia (traceroute data available)
   - CDN edge locations from cloud service data (`all_cloud_geo.json`)
   - Extensible to more destinations

3. **Traffic demand generation**
   - Per-user traffic volume: configurable distribution (Pareto/log-normal)
   - Session model: arrival rate, session lifetime CDF (follow Footprint's approach)
   - Time-of-day variation: diurnal traffic pattern
   - Output: demand matrix D[user_group][destination] per epoch

4. **Epoch management**
   - Epoch length: configurable (default 5 minutes, matching Footprint)
   - Constellation state updates: satellite positions refresh per timeslot (15s)
   - Ground delay oracle refresh: per epoch (5 min)

### Module 5: TE Controller (`vantage/simulator/controller/`)

**Goal**: Jointly optimize PoP selection and ISL routing.

**Tasks:**
1. **Input assembly**
   - RTT_sat(u, p) matrix from satellite model (per timeslot, averaged per epoch)
   - RTT_ground(p, d) matrix from ground delay oracle
   - Demand matrix D(u, d) from flow generator
   - Capacity vector C(link) for ISL and PoP peering links

2. **Baseline strategies**
   - **Hot-Potato**: assign each user to nearest PoP (current Starlink behavior)
   - **ISL-Only TE**: optimize ISL routing, PoP = nearest
   - **Ground-Only Oracle**: choose PoP with min RTT_ground(p,d), no ISL awareness
   - **Best-Static-PoP**: per-destination best PoP (static, no per-user adaptation)
   - **Joint-Latency-Only**: minimize RTT_e2e without capacity constraints (upper bound on latency reduction)

3. **Vantage-Greedy algorithm**
   - For each (user, destination) pair:
     - Enumerate feasible PoPs (reachable via satellite)
     - Score: RTT_sat(u, p) + RTT_ground(p, d)
     - Assign to best PoP if capacity allows, else next best
   - Greedy order: highest demand flows first (like Espresso GC)
   - Update capacity constraints after each assignment
   - Re-run ISL routing (Dijkstra) after PoP assignments change

4. **Vantage-LP algorithm**
   - Role: provides optimality bounds on reduced topology; Greedy is the practical deployment strategy
   - Formulation following Footprint's approach:
     ```
     min  Σ_{u,d} demand(u,d) · Σ_p w_{u,p,d} · RTT_e2e(u,p,d)
          + λ · Σ_link P(μ_link)
     s.t. Σ_p w_{u,p,d} = 1  ∀(u,d)
          μ_link ≤ 1  ∀link
          w_{u,p,d} ≥ 0
     ```
   - Solve with scipy.optimize (free) or MOSEK (academic license)
   - Session stickiness: model old sessions persisting on previous PoP (Footprint's temporal dynamics)
   - Scalability note: LP on full topology (1584 sats) may be intractable; use reduced topology (aggregate ISL links into corridors) or solve per-region subproblems. Report gap between LP bound and Greedy solution

5. **Control loop**
   - Every epoch:
     1. Update constellation state
     2. Refresh RTT_sat matrix
     3. Refresh RTT_ground matrix (from oracle)
     4. Collect demand matrix
     5. Run optimization
     6. Output assignment: (user, dest) → (pop, ISL_path)

### Module 6: Evaluation Engine (`vantage/simulator/eval/`)

**Goal**: Compute and visualize all evaluation metrics.

**Tasks:**
1. **Latency metrics**
   - E2E RTT CDF across all flows
   - Summary statistics: mean, median, p95, p99
   - RTT breakdown: satellite segment vs ground segment
   - Per-region latency (NA, EU, Asia, Africa, etc.)
   - Per-destination latency (Google, Wikipedia, Facebook)
   - Latency improvement vs each baseline (absolute ms and %)
   - 95% confidence intervals via bootstrap (≥5 random seeds per configuration)

2. **Throughput/bandwidth metrics**
   - Total traffic served without congestion (congestion-free scale)
   - Served-demand ratio at each traffic scale
   - Maximum demand that can be served under capacity constraints
   - Throughput improvement vs baselines

3. **System performance metrics**
   - Optimization computation time (wall clock)
   - Memory usage
   - Scalability: time vs constellation size, number of users, number of destinations

---

## Phase 3: Validation & Evaluation (Weeks 9-14)

### E1: Ground Delay Model Validation

**Method:**
1. Load all traceroute ground truth: (pop, dest) → ground_delay_ms
2. Leave-one-out cross-validation:
   - For each (pop, dest) pair, train L2 model on remaining data
   - Predict ground delay for held-out pair
   - Compare prediction vs actual
3. Compute: MAE, MAPE, R², per-region breakdown
4. Plot: predicted vs actual scatter plot, error CDF

**Pass criteria:** MAPE < 20%, most errors within ±10ms for nearby destinations

### E2: Satellite Delay Model Validation

**Method:**
1. For each probe in traceroute data:
   - Extract measured `avg_starlink_segment` (from summary JSON)
   - Compute predicted RTT_sat(probe_location, probe_pop) from simulator
2. Compare predicted vs measured satellite delay
3. Cross-validate against StarPerf's three delay models (XML_real, XML_geo, TLE)

**Pass criteria:** R² > 0.8, MAPE < 25%, similar or better accuracy than StarPerf models

### E3: Latency Optimization Evaluation

**Method:**
1. Generate traffic demand matrix (1000+ user groups, 3+ destinations)
2. Run all strategies: Hot-Potato, ISL-Only, Ground-Only Oracle, Best-Static-PoP, Joint-Latency-Only, Vantage-Greedy, Vantage-LP
3. Compute E2E latency CDF for each strategy (mean, median, p95, p99)
4. Break down by region and destination
5. Run ≥5 random seeds per configuration, report 95% confidence intervals
6. Sensitivity analysis: vary traffic load (0.1x to 2x), constellation size, number of PoPs

**Key plots:**
- CDF of E2E latency for all strategies (main result figure)
- Bar chart of average latency improvement per region (with error bars)
- Heatmap: latency savings by (user_region, destination)
- Sensitivity curves
- LP vs Greedy gap analysis (on reduced topology)

### E3b: Ablation Study (Component Isolation)

**Goal:** Prove that ground-segment awareness — not just having more PoP choices — drives Vantage's improvement.

**Method:**
1. Start from Vantage-Greedy as the full system
2. Ablate components one at a time:
   - **−Ground**: replace RTT_ground(p,d) with uniform constant → isolates ground awareness value
   - **−Capacity**: remove capacity constraints → shows cost of capacity-aware routing
   - **−ISL routing**: force bent-pipe only → shows ISL routing contribution
3. For each ablation, compute E2E latency CDF and compare to full Vantage
4. Report: absolute and relative degradation per ablated component

**Key plots:**
- Bar chart: latency increase when each component is removed
- CDF overlay: full Vantage vs each ablation

**Pass criteria:** Removing ground awareness causes the largest degradation (confirms dominant contribution)

### E3c: Temporal Stability & Churn Analysis

**Goal:** Show that Vantage assignments are stable across epochs (low churn), important for session continuity.

**Method:**
1. Run Vantage-Greedy over 24 consecutive epochs (2 hours simulated time)
2. Track PoP assignment changes per (user, destination) pair across epochs
3. Compute:
   - Churn rate: fraction of flows that switch PoP per epoch
   - Session disruption: fraction of active sessions affected
   - Latency variation: std dev of E2E RTT across epochs for same flow
4. Compare churn under different session stickiness weights

**Key plots:**
- Churn rate over time (line plot)
- Latency stability CDF (std dev of per-flow RTT)

### E4: Throughput/Capacity Evaluation

**Method:**
1. Scale traffic demand from 0.1x to 5x baseline
2. For each scale, compute:
   - Maximum congestion-free throughput
   - Number of overloaded links
   - Average link utilization
3. Compare Vantage vs baselines

**Key plots:**
- Congestion-free scale bar chart (like Footprint Fig 10-11)
- Link utilization CDF at different traffic scales

### E5: System Performance Benchmarks

**Method:**
1. Vary constellation size: 100, 500, 1584, 4000 satellites
2. Vary user groups: 100, 500, 1000, 5000
3. Vary destinations: 3, 10, 50
4. Measure optimization time, memory, convergence

**Key plots:**
- Computation time vs problem size
- Scalability curves

### E6: Robustness & Sensitivity Study

**Goal:** Show Vantage degrades gracefully when ground delay predictions are imperfect.

**Method:**
1. Inject noise into RTT_ground predictions:
   - Gaussian noise: σ = {5, 10, 20, 50} ms
   - Systematic bias: +{5, 10, 20} ms on random 20% of (pop, dest) pairs
   - Stale data: use ground delay from T-1h, T-6h, T-24h
2. Re-run Vantage-Greedy with noisy inputs
3. Measure degradation in E2E latency vs clean predictions
4. Sensitivity sweep on key assumptions:
   - ISL capacity: {10, 20, 40, 100} Gbps per link
   - Number of active PoPs: {20, 30, 49, 70}
   - User density scaling: {0.5x, 1x, 2x, 5x}

**Key plots:**
- Latency degradation vs noise level (line plot with error bars)
- Heatmap: sensitivity of improvement % to (ISL capacity × PoP count)

**Pass criteria:** Vantage still outperforms Hot-Potato by >5% even with σ=20ms noise

---

## Phase 4: Paper Writing (Weeks 15-20)

### Paper Sections → Experiments Mapping

| Section | Content | Data Source |
|---------|---------|------------|
| §1 Introduction | Motivation + contributions | E0 data analysis |
| §2 Motivation | Ground segment heterogeneity analysis | E0 + traceroute data |
| §3 System Design | Simulator architecture, optimization formulation | Design doc |
| §4 Ground Delay Modeling | L1/L2/L3 approach, validation | E1 |
| §5 Evaluation | Main results + ablation + robustness | E2, E3, E3b, E3c, E4, E5, E6 |
| §6 Related Work | Comparison with Footprint/Edge Fabric/Espresso | Literature |
| §7 Conclusion | Summary + future work | All |

---

## Run Order & Decision Gates

```
Week 1-2: Module 1 (Data Layer) + Module 2 start
  → Gate: All data loads, basic constellation renders correctly

Week 3-4: Module 2 (Satellite Model) + Module 3 (Ground Model L1/L2)
  → Gate: E1 passes (ground delay MAPE < 20%)
  → Gate: E2 passes (satellite delay R² > 0.8)
  → Decision: If L2 insufficient, invest in L3 (fiber graph model)

Week 5-6: Module 4 (Flow Generator) + Module 5 start (Controller)
  → Gate: Traffic patterns look realistic, no off-by-one in demand matrix

Week 7-8: Module 5 (Controller complete) + Module 6 (Eval Engine)
  → Gate: Baselines produce expected behavior (hot-potato ≈ current Starlink)
  → Gate: Optimization converges, results make physical sense

Week 9-10: E3 (Latency evaluation) + E3b (Ablation study)
  → Decision: If improvements < 10%, investigate why, adjust model
  → Gate: Ablation confirms ground awareness is dominant driver

Week 11-12: E4 (Throughput) + E5 (System benchmarks) + E3c (Churn analysis)
  → Gate: Results consistent across different traffic scales

Week 13-14: E6 (Robustness study) + sensitivity analysis + figure generation
  → Gate: All claims supported by data with confidence intervals

Week 15-20: Paper writing, revision, submission preparation
```

**Performance note:** All modules implemented in Python first. If any computation (ISL shortest path, constellation propagation) takes >10min per epoch, consider Rust acceleration via pyo3 at that point — not before.

---

## Budget & Resources

- **Compute**: Local machine for development, university cluster for large-scale runs
- **External data**: CAIDA BGP data (free for researchers), RIPE Atlas (existing probes)
- **Libraries**: Python (networkx, numpy, scipy, skyfield, h5py, matplotlib)
- **Solver**: scipy.optimize (free) or MOSEK (academic license)
- **Performance**: Python-first; Rust (pyo3) only if profiling shows bottleneck >10min per epoch
