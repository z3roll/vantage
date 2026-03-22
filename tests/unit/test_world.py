"""Tests for WorldModel, NetworkSnapshot, and GroundDelayModel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vantage.domain import NetworkSnapshot
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.ground import FiberGraphDelay, GroundInfrastructure, HaversineDelay
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.world import WorldModel


# ---------------------------------------------------------------------------
# GroundDelayModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHaversineDelay:
    """Test haversine-based ground delay estimation."""

    def test_same_location_zero_delay(self) -> None:
        model = HaversineDelay()
        delay = model.estimate(0.0, 0.0, 0.0, 0.0)
        assert delay == 0.0

    def test_delay_increases_with_distance(self) -> None:
        model = HaversineDelay()
        d1 = model.estimate(0.0, 0.0, 1.0, 0.0)  # ~111 km
        d2 = model.estimate(0.0, 0.0, 10.0, 0.0)  # ~1111 km
        assert d2 > d1

    def test_delay_is_positive(self) -> None:
        model = HaversineDelay()
        delay = model.estimate(40.0, -74.0, 37.0, -122.0)  # NY → SF
        assert delay > 0

    def test_ny_to_sf_realistic(self) -> None:
        """NY → SF ~3900 km, fiber ~29 ms one-way with 1.5x detour."""
        model = HaversineDelay()
        delay = model.estimate(40.7, -74.0, 37.8, -122.4)
        # 3900 * 1.5 / 200000 * 1000 ≈ 29 ms
        assert 20 < delay < 40

    def test_custom_fiber_speed(self) -> None:
        fast = HaversineDelay(c_fiber_km_s=300_000.0, detour_factor=1.0)
        slow = HaversineDelay(c_fiber_km_s=150_000.0, detour_factor=1.0)
        d_fast = fast.estimate(0.0, 0.0, 10.0, 0.0)
        d_slow = slow.estimate(0.0, 0.0, 10.0, 0.0)
        assert d_slow > d_fast


@pytest.mark.integration
class TestFiberGraphDelay:
    """Test fiber-graph-based ground delay."""

    @pytest.fixture
    def model(self) -> FiberGraphDelay:
        path = Path(__file__).resolve().parents[2] / "data/processed/fiber_graph.json"
        return FiberGraphDelay(path)

    def test_same_location_near_zero(self, model: FiberGraphDelay) -> None:
        delay = model.estimate(0.0, 0.0, 0.01, 0.01)
        assert delay < 1.0  # < 1 ms

    def test_delay_positive(self, model: FiberGraphDelay) -> None:
        # NY → SF
        delay = model.estimate(40.7, -74.0, 37.8, -122.4)
        assert delay > 0

    def test_delay_increases_with_distance(self, model: FiberGraphDelay) -> None:
        d_short = model.estimate(51.5, -0.1, 48.9, 2.3)   # London → Paris
        d_long = model.estimate(51.5, -0.1, 35.7, 139.7)   # London → Tokyo
        assert d_long > d_short

    def test_fiber_longer_than_haversine(self, model: FiberGraphDelay) -> None:
        """Fiber path should be longer than straight-line haversine."""
        haversine = HaversineDelay(c_fiber_km_s=200_000.0, detour_factor=1.0)
        fiber_delay = model.estimate(51.5, -0.1, 48.9, 2.3)
        haversine_delay = haversine.estimate(51.5, -0.1, 48.9, 2.3)
        # Fiber graph path >= haversine (usually longer due to routing)
        assert fiber_delay >= haversine_delay * 0.9  # allow 10% tolerance


# ---------------------------------------------------------------------------
# WorldModel + NetworkSnapshot
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWorldModel:
    """Test WorldModel snapshot generation with real data."""

    @pytest.fixture
    def world(self, starlink_xml_path: str) -> WorldModel:
        constellation = XMLConstellationModel(starlink_xml_path, dt_s=15.0)
        ground = GroundInfrastructure(
            Path(__file__).resolve().parents[2] / "data" / "processed"
        )
        satellite = SatelliteSegment(
            constellation=constellation,
            topology_builder=PlusGridTopology(),
            shell_id=1,
            ground_stations=ground.ground_stations,
            visibility=SphericalAccessModel(),
        )
        return WorldModel(satellite, ground)

    def test_snapshot_returns_network_snapshot(self, world: WorldModel) -> None:
        snapshot = world.snapshot_at(epoch=0, time_s=0.0)
        assert isinstance(snapshot, NetworkSnapshot)

    def test_snapshot_epoch_and_time(self, world: WorldModel) -> None:
        snapshot = world.snapshot_at(epoch=5, time_s=75.0)
        assert snapshot.epoch == 5
        assert snapshot.time_s == 75.0

    def test_snapshot_has_satellite_state(self, world: WorldModel) -> None:
        snapshot = world.snapshot_at(epoch=0, time_s=0.0)
        assert snapshot.satellite.num_sats > 0
        assert snapshot.satellite.delay_matrix.shape[0] == snapshot.satellite.num_sats

    def test_snapshot_has_gateway_attachments(self, world: WorldModel) -> None:
        snapshot = world.snapshot_at(epoch=0, time_s=0.0)
        attachments = snapshot.satellite.gateway_attachments.attachments
        # At least some GS should have visible satellites
        assert len(attachments) > 0
        # Each attached GS has at most top-k links
        for _gs_id, links in attachments.items():
            assert len(links) <= 5  # default top_k
            assert all(link.elevation_deg > 0 for link in links)

    def test_snapshot_has_infrastructure(self, world: WorldModel) -> None:
        snapshot = world.snapshot_at(epoch=0, time_s=0.0)
        assert len(snapshot.infra.pops) == 49
        assert len(snapshot.infra.ground_stations) == 165
        assert len(snapshot.infra.gs_pop_edges) > 0

    def test_snapshot_is_frozen(self, world: WorldModel) -> None:
        snapshot = world.snapshot_at(epoch=0, time_s=0.0)
        with pytest.raises(AttributeError):
            snapshot.epoch = 99  # type: ignore[misc]

    def test_different_times_different_positions(self, world: WorldModel) -> None:
        s0 = world.snapshot_at(epoch=0, time_s=0.0)
        s1 = world.snapshot_at(epoch=1, time_s=300.0)
        # Satellite positions should differ between epochs
        assert not np.array_equal(s0.satellite.positions, s1.satellite.positions)
