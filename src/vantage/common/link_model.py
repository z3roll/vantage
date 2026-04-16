"""Analytical link performance models for flow-level simulation.

M/M/1/K queuing model: finite-buffer queue producing queuing delay,
transmission (serialization) delay, and packet loss probability.

Path-level aggregation: end-to-end loss, bottleneck capacity.

All delays in ms, bandwidth in Gbps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_AVG_PACKET_BYTES",
    "DEFAULT_BUFFER_PACKETS",
    "LinkPerformance",
    "bottleneck_capacity",
    "link_performance",
    "path_loss",
    "pftk_throughput",
]

DEFAULT_AVG_PACKET_BYTES: int = 1500
"""Ethernet MTU — average packet size for serialization delay."""

DEFAULT_BUFFER_PACKETS: int = 1000
"""Typical router/switch buffer depth in packets."""


@dataclass(frozen=True, slots=True)
class LinkPerformance:
    """Performance metrics for a single link under a given load."""

    propagation_ms: float
    queuing_ms: float
    transmission_ms: float
    total_delay_ms: float
    loss_probability: float
    utilization: float


def link_performance(
    propagation_ms: float,
    capacity_gbps: float,
    load_gbps: float,
    buffer_packets: int = DEFAULT_BUFFER_PACKETS,
    avg_packet_bytes: int = DEFAULT_AVG_PACKET_BYTES,
) -> LinkPerformance:
    """Compute per-link delay and loss using the M/M/1/K model.

    Parameters
    ----------
    propagation_ms:
        One-way propagation delay (distance / speed of light).
    capacity_gbps:
        Link capacity in Gbps.
    load_gbps:
        Offered traffic load in Gbps.
    buffer_packets:
        Queue depth in packets (K in M/M/1/K).
    avg_packet_bytes:
        Average packet size for serialization / service time.

    Returns
    -------
    LinkPerformance with all delay components, loss, and utilization.
    """
    if capacity_gbps <= 0:
        raise ValueError(f"capacity must be positive, got {capacity_gbps}")
    if load_gbps < 0:
        raise ValueError(f"load must be non-negative, got {load_gbps}")

    rho = load_gbps / capacity_gbps  # utilization

    # Transmission (serialization) delay: time to put one packet on the wire.
    # capacity_gbps * 1e6 = bits per ms
    bits_per_ms = capacity_gbps * 1e6
    service_time_ms = (avg_packet_bytes * 8) / bits_per_ms

    transmission_ms = service_time_ms

    # M/M/1/K loss probability.
    loss = _mm1k_loss(rho, buffer_packets)

    # M/M/1/K mean queue length → queuing delay via Little's Law.
    # L = ρ/(1-ρ) - (K+1)ρ^{K+1}/(1-ρ^{K+1})  for ρ ≠ 1
    # L = K/2                                      for ρ = 1
    # Effective arrival rate: λ_eff = λ(1 - P_loss)
    # Mean system time: T = L / λ_eff
    # Queuing delay: W = T - service_time
    avg_queue_len = _mm1k_queue_length(rho, buffer_packets)
    effective_rho = rho * (1.0 - loss)  # effective utilization after loss
    if effective_rho > 1e-12 and avg_queue_len > 0:
        mean_system_time = (avg_queue_len / effective_rho) * service_time_ms
        queuing_ms = max(0.0, mean_system_time - service_time_ms)
        # Cap at buffer_packets × service_time (physical upper bound).
        queuing_ms = min(queuing_ms, buffer_packets * service_time_ms)
    else:
        queuing_ms = 0.0

    total_delay_ms = propagation_ms + queuing_ms + transmission_ms

    return LinkPerformance(
        propagation_ms=propagation_ms,
        queuing_ms=queuing_ms,
        transmission_ms=transmission_ms,
        total_delay_ms=total_delay_ms,
        loss_probability=loss,
        utilization=rho,
    )


def _mm1k_loss(rho: float, k: int) -> float:
    """Blocking probability for an M/M/1/K queue.

    P_K = (1-ρ) ρ^K / (1 - ρ^{K+1})   for ρ < 1
    P_K = 1 / (K+1)                     for ρ = 1
    P_K = (ρ-1) ρ^K / (ρ^{K+1} - 1)   for ρ > 1
    """
    if k <= 0:
        return 1.0 if rho > 0 else 0.0
    if abs(rho - 1.0) < 1e-10:
        return 1.0 / (k + 1)
    if rho == 0.0:
        return 0.0

    try:
        if rho < 1.0:
            log_num = math.log(1.0 - rho) + k * math.log(rho)
            log_den = math.log(1.0 - rho ** (k + 1))
        else:
            # ρ > 1: use the algebraically equivalent form with positive terms.
            log_num = math.log(rho - 1.0) + k * math.log(rho)
            log_den = math.log(rho ** (k + 1) - 1.0)
        return min(1.0, max(0.0, math.exp(log_num - log_den)))
    except (ValueError, OverflowError):
        # Extreme K with ρ > 1: ρ^(K+1) overflows → loss ≈ 1 - 1/ρ.
        if rho > 1.0:
            return min(1.0, 1.0 - 1.0 / rho)
        return 0.0


def _mm1k_queue_length(rho: float, k: int) -> float:
    """Mean number of customers in an M/M/1/K system.

    L = ρ/(1-ρ) - (K+1)ρ^{K+1}/(1-ρ^{K+1})  for ρ ≠ 1
    L = K/2                                      for ρ = 1
    """
    if k <= 0:
        return 0.0
    if abs(rho - 1.0) < 1e-10:
        return k / 2.0
    if rho == 0.0:
        return 0.0

    first_term = rho / (1.0 - rho)
    try:
        rho_k1 = rho ** (k + 1)
        second_term = (k + 1) * rho_k1 / (1.0 - rho_k1)
    except OverflowError:
        # ρ^(K+1) overflows: for ρ > 1, second_term ≈ -(K+1); for ρ < 1, ≈ 0
        second_term = -(k + 1) if rho > 1.0 else 0.0

    return max(0.0, first_term - second_term)


# ---------------------------------------------------------------------------
# Path-level aggregation
# ---------------------------------------------------------------------------


def path_loss(per_link_losses: Sequence[float]) -> float:
    """End-to-end loss probability: 1 - ∏(1 - p_i).

    Assumes independent loss across links (standard assumption).
    """
    survival = 1.0
    for p in per_link_losses:
        survival *= 1.0 - p
    return 1.0 - survival


def bottleneck_capacity(per_link_capacities: Sequence[float]) -> float:
    """Bottleneck (minimum) capacity along a path."""
    if not per_link_capacities:
        return 0.0
    return min(per_link_capacities)


# ---------------------------------------------------------------------------
# TCP throughput model (PFTK — Padhye/Firoiu/Towsley/Kurose)
# ---------------------------------------------------------------------------

DEFAULT_MSS_BYTES: int = 1460
"""Maximum Segment Size (bytes), typical for Ethernet MTU - headers."""

DEFAULT_MAX_WINDOW_BYTES: int = 65535 * 64
"""Receiver window with window scaling factor 6 (~4 MB)."""


def pftk_throughput(
    rtt_ms: float,
    loss: float,
    *,
    mss_bytes: int = DEFAULT_MSS_BYTES,
    rto_ms: float | None = None,
    max_window_bytes: int = DEFAULT_MAX_WINDOW_BYTES,
    b: int = 2,
) -> float:
    """PFTK steady-state TCP throughput in **Gbps**.

    Models TCP NewReno throughput as a function of round-trip time and
    segment loss probability, accounting for both fast-retransmit (3
    dup-ACK) and RTO timeout recovery.

    ::

                            MSS
        T = min(W/R, ────────────────────────────────────────)
                      R√(2bp/3) + RTO·min(1,3√(3bp/8))·p·(1+32p²)

    Reference: Padhye et al., "Modeling TCP Reno Performance: A Simple
    Model and Its Empirical Validation", IEEE/ACM ToN, 2000.

    Parameters
    ----------
    rtt_ms : Round-trip time in ms.
    loss : Segment loss probability in [0, 1].
    mss_bytes : Maximum Segment Size.
    rto_ms : Retransmission timeout; defaults to max(2×RTT, 200ms).
    max_window_bytes : Receiver window (caps throughput).
    b : Segments acknowledged per ACK (2 = delayed ACK).

    Returns
    -------
    Steady-state TCP throughput in Gbps.
    """
    if rtt_ms <= 0:
        return 0.0
    if loss <= 0:
        # No loss → window-limited throughput.
        rtt_s = rtt_ms / 1000.0
        return max_window_bytes * 8 / rtt_s / 1e9

    rtt_s = rtt_ms / 1000.0
    if rto_ms is None:
        rto_ms = max(2.0 * rtt_ms, 200.0)
    rto_s = rto_ms / 1000.0

    # Fast-retransmit term: 1 / (RTT × √(2bp/3))
    fr_term = rtt_s * math.sqrt(2.0 * b * loss / 3.0)

    # RTO timeout term
    rto_term = (
        rto_s
        * min(1.0, 3.0 * math.sqrt(3.0 * b * loss / 8.0))
        * loss
        * (1.0 + 32.0 * loss * loss)
    )

    t_pftk = 1.0 / (fr_term + rto_term)  # packets / sec

    # Window-limited rate
    t_window = max_window_bytes / (mss_bytes * rtt_s)  # packets / sec

    t = min(t_pftk, t_window)

    return t * mss_bytes * 8 / 1e9  # → Gbps
