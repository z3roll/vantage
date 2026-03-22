# Pipeline Summary

**Problem**: Satellite ISPs (Starlink) optimize only satellite segment routing, ignoring ground segment latency heterogeneity, which accounts for up to 80% of end-to-end delay for many user-destination pairs.

**Final Method Thesis**: Vantage is a simulator and TE system that jointly optimizes PoP selection and ISL routing by incorporating ground-segment latency awareness, enabling end-to-end traffic engineering for satellite networks.

**Final Verdict**: READY (proceed to implementation)

**Date**: 2026-03-20

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- **Architecture**: `refine-logs/ARCHITECTURE.md`

## Contribution Snapshot
- **Dominant contribution**: First end-to-end TE simulator for satellite networks that models both space and ground segments, with validated ground delay predictions
- **Optional supporting contribution**: LP formulation for joint PoP selection + ISL routing under capacity constraints
- **Explicitly rejected complexity**: Docker emulation, real-time BGP simulation, application-layer QoE modeling

## Must-Prove Claims
1. Ground segment dominates E2E latency for >60% of user-destination pairs (E0 ✅)
2. Simulator ground delay predictions match traceroute ground truth within MAPE <20% (E1)
3. Simulator satellite delay predictions are accurate (R² >0.8) (E2)
4. Vantage reduces average E2E latency by >10% vs hot-potato baseline (E3)
5. Ground awareness is the dominant driver of improvement (E3b ablation)
6. Vantage improves congestion-free throughput scale vs ISL-only TE (E4)
7. Vantage is robust to ground delay prediction noise (E6)

## First Runs to Launch
1. **M1: Data Layer** — load all infrastructure data, build ground truth lookup tables
2. **M2: Satellite Model** — implement constellation, ISL delay matrix, validate against StarPerf
3. **M3: Ground Delay L1/L2** — implement ground truth lookup + regression model, run E1 validation

## Main Risks
- **Ground delay prediction accuracy for unseen pairs**: Mitigation — iterative L1→L2→L3 approach, CAIDA BGP data available
- **ISL capacity assumptions**: Mitigation — sensitivity analysis with range of reasonable values
- **Traffic demand realism**: Mitigation — use Starlink subscriber density data, standard session models from Footprint

## Architecture

See `refine-logs/ARCHITECTURE.md` for the full architecture design document (reviewed by Codex, 2 rounds).

**Core design**: 3-tier architecture with 2 cross-cutting concerns:

```
Engine (orchestrator)
├── World Model (physical truth: constellation, ISL, ground delays)
├── Control Plane (system under study: observation → controller → RoutingIntent)
├── Forwarding Engine (realization: intent × truth → RealizedState)
├── Workload (traffic generation)
├── Metrics Analyzer (pure analytical functions over RealizedState)
└── Telemetry (observer hooks, cross-cutting)
```

**Key property**: Comparing controllers = swapping one config line. Same world, same forwarding, same metrics.

## Package Structure

```
/Users/zerol/PhD/vantage/
├── data/                         # Existing data (infrastructure, traces, papers)
├── refine-logs/                  # Pipeline output + architecture doc
├── configs/                      # YAML experiment configurations
├── experiments/                  # Reproducible experiment scripts (E1-E6)
├── artifacts/                    # Generated outputs (gitignored)
│   ├── cache/                    # HDF5 caches
│   ├── results/                  # Parquet/CSV results
│   └── figures/                  # Plots
│
└── src/vantage/                  # Python package
    ├── cli.py                    # Entry: python -m vantage run configs/full.yaml
    ├── model/                    # Shared immutable domain objects
    │   ├── network.py            #   SatellitePosition, ISLGraph, PoPInfo
    │   ├── flow.py               #   FlowKey, TrafficMatrix
    │   ├── policy.py             #   RoutingIntent, PathAllocation
    │   ├── state.py              #   NetworkSnapshot, RealizedState, FlowOutcome
    │   └── result.py             #   EpochMetrics, SimulationSummary
    ├── engine/                   # Simulation Engine (orchestrator)
    │   ├── runner.py             #   Main epoch loop
    │   ├── clock.py              #   SimulationClock, TimeConfig
    │   └── config.py             #   ExperimentConfig (Pydantic + YAML)
    ├── world/                    # World Model (physical truth)
    │   ├── constellation.py      #   SGP4 propagation
    │   ├── topology.py           #   TopologyBuilder: positions → ISLGraph
    │   ├── ground_network.py     #   Static infra: PoPs, GS, peering
    │   ├── ground_delay.py       #   GroundDelayModel: L1/L2/L3
    │   ├── snapshot.py           #   SnapshotBuilder → NetworkSnapshot
    │   └── failures.py           #   FailureModel (v1: no-op)
    ├── control/                  # Control Plane (system under study)
    │   ├── observation.py        #   ObservationModel: truth → noisy view
    │   ├── estimators.py         #   DemandEstimator, CapacityEstimator
    │   └── controllers/          #   TE algorithms (7 implementations)
    │       ├── protocol.py       #     TEController Protocol
    │       ├── nearest_pop.py    #     Hot-potato baseline
    │       ├── satellite_only.py #     ISL-only baseline
    │       ├── ground_only.py    #     Ground-oracle baseline
    │       ├── static_pop.py     #     Best-static-PoP baseline
    │       ├── latency_only.py   #     Unconstrained joint baseline
    │       ├── greedy.py         #     Vantage-Greedy (primary)
    │       └── lp.py             #     Vantage-LP (optimality bound)
    ├── forwarding/               # Forwarding Engine (realization)
    │   ├── realizer.py           #   intent × truth → RealizedState
    │   └── actuation.py          #   ActuationModel (v1: instant)
    ├── workload/                 # Workload Generation
    │   ├── population.py         #   EndpointPopulation
    │   └── traffic.py            #   TrafficGenerator protocol + impls
    ├── metrics/                  # Metrics Analysis (pure functions)
    │   ├── latency.py            #   CDF, percentiles, decomposition
    │   ├── throughput.py         #   Served-demand ratio, congestion-free scale
    │   └── comparator.py         #   A vs B diff, bootstrap CIs
    ├── telemetry/                # Observer Subsystem
    │   ├── observer.py           #   SimulationObserver protocol
    │   ├── exporters.py          #   Parquet, CSV exporters
    │   └── recorder.py           #   Flow-level traces
    └── data/                     # Data Loaders
        ├── infrastructure.py     #   GS, PoP, IXP, fiber, cable parsers
        ├── traceroute.py         #   Ground truth loader
        └── geolocation.py        #   IP → (lat, lon, ASN)
```

## Next Action
- Proceed to implementation: start with `model/` (domain types) + `data/` (loaders) + `world/` (constellation + topology)
- Use `/run-experiment` skill when ready to execute experiments
