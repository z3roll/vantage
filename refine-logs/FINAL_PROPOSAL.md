# Vantage: End-to-End Traffic Engineering for Satellite Networks

## Problem Anchor

Satellite ISPs like Starlink optimize only the satellite segment (user → PoP), treating the terrestrial network as a black box. Existing TE research focuses exclusively on ISL routing. However, empirical analysis of 60万+ traceroutes shows:

1. **Most traffic is bent-pipe** (no ISL), making ISL-centric TE ineffective
2. **Ground segment dominates E2E latency** — up to 80% for some user-destination pairs
3. **Ground latency is highly heterogeneous** — the same user reaches Google in 20ms but Wikipedia in 100ms via the same PoP, because destination infrastructure density varies dramatically
4. **No single PoP is universally optimal** — the best egress depends on the destination (Pearson r = 0.48 between Google and Wikipedia ground ratios)

Content providers (Google/Espresso, Meta/Edge Fabric, Microsoft/Footprint) solve this via cold-potato routing from their side. But Starlink, as a satellite ISP with centralized control and ISL capability, is uniquely positioned to do end-to-end TE at lower cost than traditional ground ISPs.

## Method Thesis (One Sentence)

Vantage is a simulator and TE system for satellite ISPs that jointly optimizes PoP selection and ISL routing by incorporating ground-segment latency awareness, reducing end-to-end latency and improving throughput compared to ISL-only baselines.

## Dominant Contribution

**The first end-to-end TE simulator for satellite networks that models both space and ground segments**, enabling joint optimization that neither ISL-centric satellite TE nor ground-only content provider TE can achieve alone.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Vantage Simulator                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │  Flow Generator  │    │   Ground Delay Oracle            │   │
│  │  - User locations│    │   - L1: Traceroute ground truth  │   │
│  │  - Destinations  │    │   - L2: Haversine regression     │   │
│  │  - Traffic volume│    │   - L3: ITU fiber graph model    │   │
│  │  - Session model │    │   - Calibrated against probes    │   │
│  └────────┬────────┘    └──────────────┬───────────────────┘   │
│           │                             │                       │
│           ▼                             ▼                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Satellite Network Model                     │   │
│  │  - Constellation: SGP4 orbital propagation               │   │
│  │  - ISL topology: +Grid connectivity                      │   │
│  │  - ISL delay: Distance/c_vacuum per timeslot             │   │
│  │  - GSL: User→Satellite→GroundStation→PoP                │   │
│  │  - Capacity: ISL bandwidth, PoP peering capacity         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   TE Controller                          │   │
│  │  Input:                                                  │   │
│  │    - RTT_sat(user, pop) for all feasible (user, pop)     │   │
│  │    - RTT_ground(pop, dest) from Ground Delay Oracle      │   │
│  │    - Traffic demand matrix D(user, dest)                  │   │
│  │    - Link capacities C(link)                             │   │
│  │                                                          │   │
│  │  Output:                                                 │   │
│  │    - PoP assignment: user → pop (per destination)        │   │
│  │    - ISL routing: satellite path for each flow           │   │
│  │                                                          │   │
│  │  Objective:                                              │   │
│  │    min Σ RTT_e2e(flow) subject to capacity constraints   │   │
│  │    where RTT_e2e = RTT_sat(u,p) + RTT_ground(p,d)       │   │
│  │                                                          │   │
│  │  Algorithms:                                             │   │
│  │    - Greedy (fast, like Espresso GC)                     │   │
│  │    - LP relaxation (optimal, like Footprint)             │   │
│  │    - Heuristic baselines for comparison                  │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Evaluation Engine                          │   │
│  │  - E2E latency (CDF, per-region, per-destination)        │   │
│  │  - Throughput / goodput                                  │   │
│  │  - Link utilization distribution                         │   │
│  │  - Congestion-free scale                                 │   │
│  │  - Optimizer computation time                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Optimization Formulation (Sketch)

**Decision variables:**
- `w_{u,p,d}` ∈ [0,1]: fraction of user u's traffic to destination d routed via PoP p
- `r_{path}`: ISL routing path selection

**Objective:**
```
min  Σ_{u,d} demand(u,d) · Σ_p w_{u,p,d} · [RTT_sat(u,p) + RTT_ground(p,d)]
     + λ · Σ_link P(utilization(link))
```
where P(μ) is a convex penalty function on link utilization (following Footprint's approach).

**Constraints:**
- `Σ_p w_{u,p,d} = 1` for all (u, d)  — all traffic must be served
- `utilization(link) ≤ 1` for all links — capacity constraint
- ISL link capacity constraints
- PoP peering capacity constraints
- Visibility constraints (user can only reach satellites within elevation angle)

## Baselines

1. **Hot-Potato (Current Starlink)**: Route to nearest PoP, ignore ground delay
2. **ISL-Only TE**: Optimize ISL routing but fix PoP assignment to nearest
3. **Oracle Ground-Only**: Optimize PoP selection using ground truth delays but no ISL awareness
4. **Vantage-Greedy**: Joint optimization with greedy algorithm
5. **Vantage-LP**: Joint optimization with LP solver

## Key Claims

1. **Ground segment dominates**: For >60% of user-destination pairs, ground segment accounts for >50% of E2E latency
2. **Simulator accuracy**: Ground delay predictions match traceroute ground truth within MAPE <20%
3. **Latency reduction**: Vantage reduces average E2E latency by X% compared to hot-potato baseline
4. **Throughput improvement**: Vantage improves congestion-free scale by X% compared to ISL-only TE
5. **Scalability**: Optimization completes within seconds for realistic constellation sizes

## Complexity Intentionally Rejected

- **Full Docker emulation** (StarryNet approach) — unnecessary for evaluation at scale
- **Application-layer QoE modeling** (Espresso's MTBR) — can be added later if needed
- **Real-time BGP simulation** — we model ground delay empirically, not via BGP convergence
- **Multi-shell constellation** — start with single-shell Starlink, extend later

## Remaining Risks

1. **Ground delay prediction accuracy** for unseen (PoP, destination) pairs — mitigated by iterative modeling (L1→L2→L3)
2. **Traffic demand model realism** — mitigated by using published Starlink user distribution data and standard traffic models
3. **ISL capacity assumptions** — Starlink ISL specs are not public; use reasonable estimates from literature
