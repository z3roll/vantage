# Refinement Report

## Starting Point
- Research idea: End-to-end TE for satellite networks (Vantage)
- Preliminary validation: POP reassignment analysis in StarPerf experiment notebook
- Reference systems: Footprint (Microsoft), Edge Fabric (Meta), Espresso (Google)
- Available data: 313 traceroutes, Starlink infrastructure (49+ PoPs, GS), ITU fiber, peering data

## Key Refinements Made

### 1. From "POP-CDN coupling analysis" to "End-to-end TE system"
The earlier framing (StarRoute) focused on diagnosing the problem. Vantage focuses on solving it — designing a system that Starlink could deploy to jointly optimize satellite and ground segments.

### 2. From analysis notebook to simulator architecture
The StarPerf experiment notebook does POP reassignment as a static optimization. Vantage needs:
- Dynamic constellation model (satellites move every 15s)
- Flow-level traffic generation
- Capacity-constrained optimization
- Temporal evaluation (multiple epochs)

### 3. Ground delay modeling strategy
Decided on iterative approach rather than building a perfect model upfront:
- L1: Direct traceroute ground truth (313 traces) + haversine regression for missing pairs
- L2: ITU fiber graph shortest path with calibration
- L3 (if needed): CAIDA BGP topology + routing inflation model

### 4. Evaluation metrics scoped
Prioritized based on paper comparison:
- **Tier 1 (must-have)**: E2E latency, throughput/bandwidth, system performance
- **Tier 2 (nice-to-have)**: Link utilization, congestion-free scale, detour analysis
- **Tier 3 (future work)**: Application QoE (MTBR, goodput), failure recovery

### 5. Technical stack
- Python for main framework
- Rust for compute-intensive hotspots (ISL shortest path, constellation propagation)
- Data: existing vantage/data/ infrastructure files
