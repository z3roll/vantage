# Vantage

**End-to-end-aware traffic steering for Starlink-class satellite networks.**

Vantage is an event-driven simulator that compares two PoP-selection strategies
for user-to-service flows that traverse the Starlink constellation:

- **Baseline (nearest-PoP):** every cell is pinned to its geographically
  closest PoP — the current Starlink default.
- **Progressive (cascade):** a central controller ranks PoPs per
  `(cell, destination)` by predicted end-to-end RTT and emits a ranked
  cascade; the data plane walks the cascade and picks the first egress
  whose Ka-feeder still has capacity.

Both controllers run in lockstep against bit-identical traffic
(~6.4M users, 7 destination services, ~50 PoPs, 1-second epochs). Every
flow's latency is decomposed into propagation, queuing, transmission and
ground components, and results stream live to a browser dashboard.

## Quick Start

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.13+. Everything
else (constellation XML, geography, configs) is bundled in the repo.

```bash
uv sync                 # one-time dependency install
uv run python run.py    # runs 60-epoch demo, auto-opens dashboard
```

The script starts a local HTTP server on `http://localhost:8000/` and opens
it in your default browser. The dashboard polls every 3 seconds, so you can
watch results fill in while the simulation is still running.

### Options

```bash
uv run python run.py --epochs 600        # longer run (~10 min)
uv run python run.py --user-scale 1.0    # use real ~6.4M user count
uv run python run.py --port 9000         # alternative HTTP port
uv run python run.py --no-browser        # headless (server still up)
```

When the simulation finishes the server keeps running so you can keep
browsing — `Ctrl-C` to exit.

## Project Structure

```
run.py                          # entry point: runs sim + serves dashboard
dashboard/index.html            # live dashboard (polls JSON every 3s)
src/vantage/
    domain/                     # frozen dataclasses (Cell, FIB, CapacityView, ...)
    world/
        ground/                 # ground infra, PoPs, GroundKnowledge cache
        satellite/              # constellation, ISL topology, visibility
    control/
        policy/
            nearest_pop.py      # Baseline controller
            greedy.py           # Progressive (cascade) controller
            common/             # shared FIB / candidate-scorer infrastructure
    traffic/                    # EndpointPopulation, FlowLevelGenerator
    engine/                     # RunContext, GroundDelayFeedback
    forward.py                  # data plane: realize intent -> per-flow RTT
    analysis/                   # offline metrics helpers
    common/                     # physical constants, geo utilities
    config/                     # bundled inputs (Starlink.xml, geojson, JSONs)
```

All domain types are immutable (`@dataclass(frozen=True, slots=True)`); all
latencies are in **ms**; all capacities in **Gbps**; all coordinates in
**degrees**.
