# Experiment Tracker

## Status Legend
- ⬜ Not started
- 🔄 In progress
- ✅ Complete
- ❌ Blocked
- 🔀 Decision needed

---

## Module Implementation

| Module | Status | Notes |
|--------|--------|-------|
| M1: Data Layer | ⬜ | Infrastructure loaders, traceroute GT, geolocation |
| M2: Satellite Network Model | ⬜ | SGP4 constellation, ISL topology, GSL, delay calc |
| M3: Ground Delay Model | ⬜ | L1 ground truth, L2 regression, L3 fiber graph |
| M4: Flow Generator | ⬜ | User distribution, destinations, demand matrix |
| M5: TE Controller | ⬜ | Hot-potato, ISL-only, Ground-only, Static-PoP, Latency-only, Greedy, LP |
| M6: Evaluation Engine | ⬜ | Latency, throughput, system metrics |

## Experiments

| Experiment | Status | Pass Criteria | Result |
|-----------|--------|---------------|--------|
| E0: Motivation data analysis | ✅ | Ground ratio >50% for many pairs | Confirmed (up to 80%) |
| E1: Ground delay validation | ⬜ | MAPE <20%, R² >0.7 | — |
| E2: Satellite delay validation | ⬜ | R² >0.8, MAPE <25% | — |
| E3: Latency optimization | ⬜ | Improvement >10% vs hot-potato (with 95% CI) | — |
| E3b: Ablation study | ⬜ | Ground awareness is dominant contributor | — |
| E3c: Temporal stability | ⬜ | Low churn rate across epochs | — |
| E4: Throughput evaluation | ⬜ | Congestion-free scale improvement | — |
| E5: System benchmarks | ⬜ | Optimization <10s for 1584 sats | — |
| E6: Robustness study | ⬜ | >5% improvement even with σ=20ms noise | — |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-20 | Use iterative ground delay model (L1→L2→L3) | Pragmatic; full BGP sim out of scope |
| 2026-03-20 | Prioritize latency, throughput, system metrics | User preference; other metrics later |
| 2026-03-20 | Python main + optional Rust for hotspots | Balance development speed and performance |
| 2026-03-20 | Target NSDI 2027 (September deadline) | 5 months available |
| 2026-03-21 | Added ablation (E3b), churn (E3c), robustness (E6) | Codex review: needed to isolate contributions and prove robustness |
| 2026-03-21 | Added Best-Static-PoP and Joint-Latency-Only baselines | Stronger comparison; isolates per-user adaptation and capacity-awareness |
| 2026-03-21 | LP positioned as optimality bound, Greedy as primary | LP may not scale to full topology; report gap |
| 2026-03-21 | Rust deprioritized; Python-first, Rust only if >10min/epoch | User preference; avoid premature optimization |
| 2026-03-21 | Enhanced metrics: p95/p99, 95% CI, served-demand ratio | Statistical rigor for NSDI-level evaluation |
