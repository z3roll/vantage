# Review Summary

## Review Round 1: Problem Validation

**Verdict**: The problem is well-motivated. Ground segment latency heterogeneity is a real and underexplored issue.

**Strengths:**
- Strong empirical motivation (60万+ traceroutes, ground ratio up to 80%)
- Clear gap: ISL-centric TE provides limited benefit when most traffic is bent-pipe
- Unique positioning: satellite ISP doing TE (vs content provider doing TE)

**Concerns addressed:**
- Q: "Is this just ISL routing with extra steps?" → A: No. The core insight is that PoP selection matters more than ISL routing for most traffic. ISL is the mechanism to reach alternative PoPs, but the optimization target is ground segment delay.
- Q: "Why can't CDN providers solve this?" → A: They can partially (Espresso, Edge Fabric), but they optimize from their side only. Starlink controls both satellite and ground egress, enabling joint optimization.

## Review Round 2: Method Design

**Verdict**: System architecture is sound. Modular design with clear separation of concerns.

**Key decisions:**
1. **Simulator over emulator**: Chose computation-based simulation (like Footprint) over Docker-based emulation (like StarryNet). Reason: need to evaluate at scale (49+ PoPs, 1000s of users, multiple destinations).
2. **Iterative ground delay model**: Start with traceroute ground truth + regression, iterate to fiber graph model. Reason: pragmatic; full BGP simulation is out of scope.
3. **LP + Greedy dual approach**: LP for optimal bounds, greedy for practical deployment (following Espresso's choice of greedy over LP for operational simplicity).

## Review Round 3: Experiment Design

**Verdict**: Must prove simulator accuracy before claiming optimization results.

**Critical experiment order:**
1. First validate ground delay model against traceroute ground truth
2. Then validate satellite delay model against StarPerf/measured data
3. Only then run optimization experiments
4. Sensitivity analysis on key assumptions (ISL capacity, traffic model)
