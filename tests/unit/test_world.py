"""Tests for WorldModel, NetworkSnapshot, and MeasuredGroundDelay."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from vantage.domain import NetworkSnapshot
from vantage.world.ground import (
    DEFAULT_MEASURED_SERVICES,
    GroundInfrastructure,
    MeasuredGroundDelay,
)
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.constellation import XMLConstellationModel
from vantage.world.satellite.topology import PlusGridTopology
from vantage.world.satellite.visibility import SphericalAccessModel
from vantage.world.world import WorldModel

TRACEROUTE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "probe_trace" / "traceroute"
)


# ---------------------------------------------------------------------------
# MeasuredGroundDelay
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMeasuredGroundDelayInMemory:
    """Direct construction + strict lookup semantics."""

    def test_lookup_returns_measured_value(self) -> None:
        model = MeasuredGroundDelay(
            one_way_rtt_ms={("fra", "google"): 12.5, ("fra", "facebook"): 9.75},
        )
        assert model.estimate("fra", "google") == 12.5
        assert model.estimate("fra", "facebook") == 9.75

    def test_unknown_pair_raises(self) -> None:
        model = MeasuredGroundDelay(one_way_rtt_ms={("fra", "google"): 12.5})
        with pytest.raises(KeyError, match="no measured ground RTT"):
            model.estimate("fra", "wikipedia")
        with pytest.raises(KeyError, match="no measured ground RTT"):
            model.estimate("lhr", "google")

    def test_empty_raises_on_every_lookup(self) -> None:
        model = MeasuredGroundDelay.empty()
        assert len(model) == 0
        with pytest.raises(KeyError):
            model.estimate("fra", "google")

    def test_has_and_pops_and_destinations(self) -> None:
        model = MeasuredGroundDelay(
            one_way_rtt_ms={
                ("fra", "google"): 1.0,
                ("fra", "facebook"): 2.0,
                ("lhr", "google"): 3.0,
            }
        )
        assert model.has("fra", "google")
        assert not model.has("lhr", "wikipedia")
        assert model.pops() == frozenset({"fra", "lhr"})
        assert model.destinations() == frozenset({"google", "facebook"})
        assert len(model) == 3

    def test_direct_dict_is_frozen_on_construction(self) -> None:
        live: dict[tuple[str, str], float] = {("fra", "google"): 10.0}
        model = MeasuredGroundDelay(one_way_rtt_ms=live)
        # Tampering with the original dict must not change the model.
        live[("fra", "google")] = 0.0
        live[("hacker", "google")] = 9999.0
        assert model.estimate("fra", "google") == 10.0
        assert not model.has("hacker", "google")

    def test_frozen_dataclass(self) -> None:
        model = MeasuredGroundDelay(one_way_rtt_ms={("fra", "google"): 10.0})
        with pytest.raises(AttributeError):
            model.one_way_rtt_ms = MappingProxyType({})  # type: ignore[misc]


@pytest.mark.unit
class TestMeasuredGroundDelayFromTraceroute:
    """Loader semantics over the real traceroute summaries in data/."""

    @pytest.fixture
    def model(self) -> MeasuredGroundDelay:
        return MeasuredGroundDelay.from_traceroute_dir(TRACEROUTE_DIR)

    def test_covers_29_pops_times_3_services(
        self, model: MeasuredGroundDelay
    ) -> None:
        """Expected shape: 29 PoPs × 3 services = 87 measured pairs."""
        assert len(model) == 29 * len(DEFAULT_MEASURED_SERVICES)
        assert len(model.pops()) == 29
        assert model.destinations() == frozenset(DEFAULT_MEASURED_SERVICES)

    def test_every_measured_pair_is_positive(
        self, model: MeasuredGroundDelay
    ) -> None:
        for pop in model.pops():
            for svc in DEFAULT_MEASURED_SERVICES:
                assert model.estimate(pop, svc) > 0

    def test_values_are_one_way_not_round_trip(
        self, model: MeasuredGroundDelay
    ) -> None:
        """Loader divides round-trip measurements by 2.

        We can't verify the exact source value here, but a sanity
        bound is: every one-way RTT should be below the slowest
        plausible trans-Pacific round-trip (~300 ms one-way = 600 ms
        RTT), well above zero.
        """
        for pop in model.pops():
            for svc in DEFAULT_MEASURED_SERVICES:
                v = model.estimate(pop, svc)
                assert 0 < v < 300

    def test_unknown_pair_still_raises(
        self, model: MeasuredGroundDelay
    ) -> None:
        with pytest.raises(KeyError):
            model.estimate("fra", "tencent")  # tencent is not measured
        with pytest.raises(KeyError):
            model.estimate("narita_fake", "google")

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MeasuredGroundDelay.from_traceroute_dir(tmp_path / "nope")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "traceroute").mkdir()
        with pytest.raises(FileNotFoundError, match="no traceroute summary"):
            MeasuredGroundDelay.from_traceroute_dir(tmp_path / "traceroute")

    def test_partial_directory_raises_by_default(self, tmp_path: Path) -> None:
        """With the default ``require_all_services=True``, a directory
        containing only a subset of the requested service files must
        raise so experiments can't silently degrade to partial data."""
        import shutil

        sub = tmp_path / "traceroute"
        sub.mkdir()
        # Copy only google's file, leave facebook/wikipedia missing.
        shutil.copy(TRACEROUTE_DIR / "google_summary.json", sub / "google_summary.json")
        with pytest.raises(FileNotFoundError, match="require_all_services"):
            MeasuredGroundDelay.from_traceroute_dir(sub)

    def test_partial_directory_accepted_when_explicit(self, tmp_path: Path) -> None:
        """Opting in via ``require_all_services=False`` returns the
        partial table — this is the tooling/test escape hatch."""
        import shutil

        sub = tmp_path / "traceroute"
        sub.mkdir()
        shutil.copy(TRACEROUTE_DIR / "google_summary.json", sub / "google_summary.json")
        model = MeasuredGroundDelay.from_traceroute_dir(
            sub, require_all_services=False
        )
        # Only google should be present.
        assert model.destinations() == frozenset({"google"})
        assert len(model) == len(model.pops())  # 1 service × 29 PoPs


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
