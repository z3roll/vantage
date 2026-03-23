# Bifrost

**Steering Satellite Traffic with End-to-End Awareness.**

## Key Idea

Current satellite networks optimize each segment in isolation — ISL routing minimizes satellite hop delay, but is oblivious to what happens after traffic exits the constellation. This local optimization produces globally suboptimal end-to-end paths.

Bifrost takes a holistic view: by jointly reasoning about satellite and terrestrial segments, the system steers traffic toward exit points that minimize *total* path delay, not just the space segment. The core insight is that a moderately longer satellite path can dramatically shorten the overall journey if it delivers traffic to a better terrestrial handoff point.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          SIMULATION ENGINE                               │
│  Orchestrates all subsystems. Drives time. Manages feedback loop.       │
│                                                                          │
│  ┌─ Epoch Loop ───────────────────────────────────────────────────────┐  │
│  │                                                                     │  │
│  │  ┌─────────┐     ┌────────────┐     ┌──────────────┐               │  │
│  │  │  Clock   │────▶│   World    │────▶│  Network     │               │  │
│  │  │  (time t)│     │   Model    │     │  Snapshot(t) │               │  │
│  │  └─────────┘     └────────────┘     └──────┬───────┘               │  │
│  │                                            │                        │  │
│  │                  ┌─────────────────────────┼────────────────┐       │  │
│  │                  │     CONTROL PLANE       │                │       │  │
│  │                  │                         ▼                │       │  │
│  │  ┌──────────┐    │  ┌───────────────────────────────────┐   │       │  │
│  │  │ Traffic  │    │  │  TE Controller                     │   │       │  │
│  │  │ (demand) │───▶│  │  (CandidateBasedController)       │   │       │  │
│  │  └──────────┘    │  │  enumerate candidates → score →   │   │       │  │
│  │                  │  │  select best → RoutingIntent       │   │       │  │
│  │  ┌──────────┐    │  └──────────────┬────────────────────┘   │       │  │
│  │  │  Ground  │───▶│                 │                        │       │  │
│  │  │Knowledge │    │  Strategies:    │                        │       │  │
│  │  │(L1+L2/L3)│◀ ─ ┤  nearest_pop | ground_only |           │       │  │
│  │  └──────────┘    │  static_pop  | greedy                   │       │  │
│  │       ▲          └─────────────┼───────────────────────────┘       │  │
│  │       │                        │                                    │  │
│  │       │                        ▼                                    │  │
│  │       │  ┌──────────────────────────────────────────────────────┐   │  │
│  │       │  │  Forwarding Engine (forward.py)                      │   │  │
│  │       │  │  RoutingIntent × Snapshot × Demand → EpochResult    │   │  │
│  │       │  │  Computes actual E2E delays along resolved paths     │   │  │
│  │       │  └──────────────────────┬───────────────────────────────┘   │  │
│  │       │                         │                                   │  │
│  │       │  ┌──────────────────────▼───────────────────────────────┐   │  │
│  │       │  │  Feedback Observer (engine_feedback.py)               │   │  │
│  │       └──│  EpochResult → ground_knowledge.put(pop, dest, rtt) │   │  │
│  │          └──────────────────────┬───────────────────────────────┘   │  │
│  │                                 │                                   │  │
│  │                                 ▼                                   │  │
│  │          ┌──────────────────────────────────────────────────────┐   │  │
│  │          │  Analysis (analysis/metrics.py)                      │   │  │
│  │          │  EpochResult → latency stats, controller comparison │   │  │
│  │          └──────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Dependency Model

```
engine ──orchestrates──▶ { world, traffic, control, forward, feedback, analysis }

control  ──reads──▶ world            (NetworkSnapshot: satellite positions, ISL graph)
control  ──reads──▶ ground_knowledge (GroundKnowledge: cached + estimated ground delays)
control  ──reads──▶ traffic          (TrafficDemand: flow demands per epoch)

forward  ──reads──▶ world            (truth: actual topology, actual delays)
forward  ──reads──▶ control          (RoutingIntent to realize)
forward  ──reads──▶ ground_knowledge (ground delay for E2E computation)

feedback ──writes─▶ ground_knowledge (observed ground delays from forwarding results)

analysis ──reads──▶ forward          (EpochResult to analyze, offline only)

world    ──reads──▶ nothing          (pure model of physical reality)
traffic  ──reads──▶ nothing          (pure demand generator)
```

### Key Design Rules

- All domain types are **frozen dataclasses** (`frozen=True, slots=True`)
- **Protocol** over ABC — duck typing, any class with right methods works
- No subsystem calls "upward" to engine
- Control plane never modifies world state (read-only)
- `GroundKnowledge` is the **single source of truth** for ground delays
- New strategies require only a new `CandidateScorer` implementation

## E2E Delay Decomposition

```
User → (uplink) → Sat_ingress → (ISL hops) → Sat_egress → (downlink) → GS → (backhaul) → PoP → (ground) → Destination
         ~2ms          variable (~5-60ms)          ~2ms        ~1ms              variable (~1-105ms)
```

All delay values in the system are **RTT** (round-trip time).

## Quick Start

```bash
uv sync                                    # install deps
uv run python -m vantage.config.preprocess  # preprocess raw data (run once)
uv run python -m vantage.main              # run experiment
uv run pytest tests/                        # run tests
```

## Project Structure

```
src/vantage/
    engine/              # Epoch loop orchestrator + RunContext
    domain/              # Frozen dataclasses (pure data models)
    world/
        ground/          # GroundKnowledge, GroundInfrastructure, delay models
        satellite/       # SatelliteSegment, TVG routing (scipy), constellation
    control/
        controller.py    # TEController Protocol + factory
        policy/
            common/      # CandidateBasedController, candidate, scoring, utils
            nearest_pop.py
            ground_only.py
            static_pop.py
            greedy.py
    forward.py           # Realize intent → compute actual delays
    engine_feedback.py   # GroundDelayFeedback observer
    traffic/             # EndpointPopulation, generators (Uniform / Gravity)
    analysis/            # Metrics, controller comparison
    common/              # Physical constants, geographic utilities
    config/              # Preprocessed data + preprocess script
```
