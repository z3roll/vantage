# Vantage: Request to Google (Ground-Aware PoP Selection)

Vantage TE adds ground delay knowledge to the routing decision. Traffic exits at the PoP nearest to the **destination**, trading longer satellite paths for much shorter ground paths.

```mermaid
sequenceDiagram
    autonumber
    participant VC as Vantage Controller<br/>(centralized)
    participant UT as User Terminal<br/>(Berlin, Germany)
    participant IS as Ingress Satellite<br/>(overhead Berlin)
    participant ISL as ISL Network<br/>(hop-by-hop)
    participant ES as Egress Satellite<br/>(over SJC GS)
    participant GS as Ground Station<br/>(San Jose, CA)
    participant POP as PoP San Jose<br/>(nearest to GOOGLE)
    participant G as Google<br/>(Mountain View, CA)

    rect rgb(230, 245, 255)
        Note over VC: PERIODIC (every epoch):<br/>Reads: sat positions + ISL topology<br/>+ ground delay knowledge
        VC->>VC: Compute optimal PoP per destination<br/>google -> PoP_SJC (min E2E)
        Note over VC: OUTPUT: routing policy<br/>{google: pop_sjc, amazon: pop_sea, ...}
        VC->>IS: Push UPDATED forwarding tables<br/>(route google traffic to GS_SJC)
        VC->>ES: Push UPDATED forwarding tables
        Note over VC: NEW INFO ADDED:<br/>1. Ground delay knowledge (L1+L2)<br/>2. Optimized ISL forwarding tables
    end

    Note over UT: HAS: satellite ephemeris<br/>(unchanged from baseline)
    Note over IS: HAS: UPDATED forwarding table<br/>(from Vantage, NOT default shortest)
    Note over GS: HAS: fixed fiber backhaul<br/>to bound PoP (San Jose)

    UT->>IS: Uplink packet (same satellite selection)
    Note right of UT: Terminal behavior<br/>UNCHANGED

    IS->>ISL: Forward to OPTIMAL GS (SJC)<br/>(longer ISL path, ~60ms)
    Note over IS,ISL: ISL path is LONGER than baseline<br/>(Berlin -> California vs Berlin -> Frankfurt)
    ISL->>ES: Hop-by-hop to SJC egress<br/>(~55ms ISL delay)

    ES->>GS: Downlink to San Jose GS<br/>(~2ms downlink)

    GS->>POP: Fiber backhaul to PoP SJC<br/>(~0.5ms backhaul)

    rect rgb(230, 255, 230)
        Note over POP,G: OPTIMIZED GROUND PATH<br/>San Jose -> Mountain View = ~1ms RTT
        POP->>G: Short terrestrial path<br/>(~1ms ground RTT)
    end

    G-->>POP: Response
    POP-->>GS: Reverse path
    GS-->>ES: Uplink
    ES-->>ISL: ISL return
    ISL-->>IS: Hop-by-hop
    IS-->>UT: Downlink to terminal

    Note over UT,G: TOTAL E2E RTT ~ 67ms (vs 121ms baseline)<br/>Satellite: ~66ms (longer path) | Ground: ~1ms (SHORT path)<br/>Trade: +50ms satellite for -104ms ground = NET -54ms
```

## What Vantage Adds (Delta from Baseline)

| Node | Baseline Info | Vantage Adds |
|------|-------------|-------------|
| Vantage Controller | (does not exist) | **NEW**: sat positions + ISL topology + ground delay knowledge → routing policy |
| Ingress Satellite | Default forwarding table (nearest GS) | **UPDATED** forwarding table (optimal GS per destination) |
| User Terminal | Ephemeris + antenna data | (unchanged — no terminal modification needed) |
| Ground Station | Fixed backhaul | (unchanged) |
| PoP | BGP table | (unchanged) |

## Key Insight

The **only infrastructure change** is updating satellite forwarding tables. No terminal changes, no GS changes, no PoP changes. Vantage is a **software-only control plane upgrade**.
