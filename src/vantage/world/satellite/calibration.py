"""Satellite delay calibration using observed traffic baselines.

Adjusts propagation-based satellite delay estimates using observed
baselines from terminal telemetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vantage.domain import Endpoint, NetworkSnapshot
    from vantage.world.satellite.visibility import SphericalAccessModel


class SatelliteDelayCalibration:
    """Calibrates satellite segment delay using observed baselines.

    Each terminal has an observed satellite segment RTT for its current PoP.
    When the controller redirects to a different PoP, the satellite
    segment changes. We estimate:

        new_sat_delay = observed_sat_delay + (est_new - est_old)
    """

    def __init__(self) -> None:
        # terminal_id → (pop_code, observed_sat_rtt, estimated_sat_rtt)
        self._baseline: dict[str, tuple[str, float, float]] = {}

    def set_baseline(
        self,
        terminal_id: str,
        pop_code: str,
        observed_sat_rtt: float,
        estimated_sat_rtt: float,
    ) -> None:
        """Record a terminal's observed satellite delay at its baseline PoP."""
        self._baseline[terminal_id] = (
            pop_code,
            observed_sat_rtt,
            estimated_sat_rtt,
        )

    def calibrate(
        self, terminal_id: str, new_estimated_sat_rtt: float
    ) -> float:
        """Calibrate a new satellite segment estimate using baseline truth."""
        baseline = self._baseline.get(terminal_id)
        if baseline is None:
            return new_estimated_sat_rtt

        _, observed, est_old = baseline
        calibrated = observed + (new_estimated_sat_rtt - est_old)
        return max(calibrated, 0.0)

    def terminals(self) -> list[tuple[str, str, float]]:
        """Return (terminal_id, pop_code, observed_sat_rtt) for all baselines."""
        return [
            (name, pop, obs)
            for name, (pop, obs, _) in self._baseline.items()
        ]

    def fill_estimates(
        self,
        snapshot: NetworkSnapshot,
        endpoints: dict[str, Endpoint],
        access: SphericalAccessModel,
    ) -> None:
        """Compute propagation-based satellite estimate for each terminal's baseline PoP.

        Must be called after baselines are set (via set_baseline) and before
        calibrate() is used. Fills in the estimated_sat_rtt for each terminal.
        Access model is used only for nearest-satellite visibility check.
        """
        sat = snapshot.satellite

        for terminal_id, pop_code, _ in self.terminals():
            ep = endpoints.get(terminal_id)
            if ep is None:
                continue

            # Find ingress satellite (visibility check)
            uplink = access.nearest_satellite(
                ep.lat_deg, ep.lon_deg, 0.0, sat.positions, 25.0
            )
            if uplink is None:
                continue

            best_est = float("inf")
            for gs_id, _ in snapshot.infra.pop_gs_edges(pop_code):
                gs = snapshot.infra.gs_by_id(gs_id)
                if gs is None:
                    continue
                gs_links = snapshot.satellite.gateway_attachments.attachments.get(gs_id)
                if not gs_links:
                    continue
                for downlink in gs_links:
                    est = sat.compute_satellite_rtt(
                        uplink.sat_id, downlink.sat_id,
                        ep.lat_deg, ep.lon_deg,
                        gs.lat_deg, gs.lon_deg,
                    )
                    if est < best_est:
                        best_est = est

            if best_est < float("inf"):
                _, observed, _ = self._baseline[terminal_id]
                self._baseline[terminal_id] = (pop_code, observed, best_est)

