# Vantage

**End-to-end-aware traffic steering for Starlink-class satellite networks.**

Vantage is an event-driven simulator that compares PoP-selection strategies
for user-to-service flows that traverse a Starlink-class constellation:

- **Baseline (nearest-PoP):** every cell is pinned to its geographically
  closest PoP and spills through the nearest cascade under pressure.
- **Progressive (cascade):** a central controller ranks PoPs per
  `(cell, destination)` by predicted end-to-end RTT and emits a ranked
  cascade; the data plane walks the cascade and picks the first egress
  whose Ka-feeder still has capacity.
- **LP rounding / MILP:** optimization references for the same
  `(cell, destination) -> PoP` assignment problem, used to compare the
  greedy controller against relaxation and integer-optimal baselines.

All controllers run in lockstep against bit-identical traffic
(12 destination services, 48 PoPs, 1-second epochs). Every flow's latency is
decomposed into propagation, queuing, transmission, satellite, and ground
components, and results stream live to a browser dashboard.

## Quick Start

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.11+. Everything
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
uv run python run.py --seed 12345        # reproducible traffic/truth/ingress
uv run python run.py --port 9000         # alternative HTTP port
uv run python run.py --no-browser        # headless (server still up)
uv run python run.py --no-serve          # no dashboard server, just write JSON
uv run python run.py --max-gs-per-pop 2  # cap GS attachments per PoP
```

When the simulation finishes the server keeps running so you can keep
browsing — `Ctrl-C` to exit.

## Project Structure

```
run.py                          # entry point: runs sim + serves dashboard
dashboard/index.html            # live dashboard (polls JSON every 3s)
src/vantage/
    model/                      # physical network model
        coverage.py             # cells, H3 land coverage, endpoint mapping
        network.py              # WorldModel and NetworkSnapshot
        ground/                 # GS/PoP infrastructure and ground latency truth
        satellite/              # constellation, shell state, topology, routing
    control/
        plane.py                # routing-plane artifacts consumed by forward
        knowledge.py            # learned ground RTT cache and eviction helpers
        feedback.py             # folds realized ground truth into knowledge
        costing.py              # shared ground-cost lookup for policies
        evaluation.py           # control-plan objective and predicted RTT stats
        policy/                 # routing-plane builders
            nearest_pop.py      # Baseline controller
            greedy.py           # Progressive (cascade) controller
            lpround.py          # LP relaxation + rounding controller
            milp.py             # integer optimum reference controller
            common/             # shared routing-plane construction helpers
    forward/                    # satellite-side traffic engineering layer
        strategy/               # path choice and RoutingPlaneForward
        execution/              # two-pass realize loop and measurement
        resources/              # capacity views and usage accounting
        results/                # per-flow and per-epoch outcomes
    traffic/                    # EndpointPopulation, FlowLevelGenerator
    sim/                        # CLI config, runtime build, metrics, dashboard I/O
    common/                     # physical constants, geo utilities
    config/                     # bundled inputs (Starlink.xml, geojson, JSONs)
```

Most model and routing-plane types are immutable dataclasses; all latencies are
in **ms**; all capacities in **Gbps**; all coordinates in **degrees**.
