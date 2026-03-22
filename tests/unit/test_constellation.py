"""Tests for constellation model."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.world.satellite.constellation import XMLConstellationModel, parse_xml_config


STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")


@pytest.mark.unit
class TestParseXMLConfig:
    """Test XML constellation config parsing."""

    def test_parse_starlink(self) -> None:
        config = parse_xml_config(STARPERF_XML)
        assert config.name == "Starlink"
        assert len(config.shells) == 6

    def test_shell1_params(self) -> None:
        config = parse_xml_config(STARPERF_XML)
        shell1 = config.shells[0]
        assert shell1.altitude_km == 476.0
        assert shell1.orbit_cycle_s == 5638.0
        assert shell1.inclination_deg == 53.0
        assert shell1.num_orbits == 72
        assert shell1.sats_per_orbit == 32
        assert shell1.total_sats == 2304

    def test_polar_shell_detected(self) -> None:
        config = parse_xml_config(STARPERF_XML)
        # Shell 6: 97.5° inclination — polar
        shell6 = config.shells[5]
        assert shell6.is_polar is True
        # Shell 1: 53° — not polar
        assert config.shells[0].is_polar is False

    def test_total_sats(self) -> None:
        config = parse_xml_config(STARPERF_XML)
        # Sum of all shells
        assert config.total_sats > 0


@pytest.mark.unit
class TestXMLConstellationModel:
    """Test XML-based constellation position generation."""

    @pytest.fixture
    def small_model(self) -> XMLConstellationModel:
        """Create model with a small custom XML for fast testing."""
        # Use shell 6 (smallest: 6 orbits × 58 sats = 348)
        # from the real Starlink XML — still reasonably fast
        return XMLConstellationModel(
            xml_path=STARPERF_XML,
            dt_s=60.0,  # larger step for speed
        )

    def test_num_timeslots(self, small_model: XMLConstellationModel) -> None:
        # Shell 1: orbit_cycle = 5638s, dt = 60s → 93 timeslots
        assert small_model.num_timeslots_for_shell(1) == 5638 // 60

    def test_positions_at_returns_correct_count(
        self, small_model: XMLConstellationModel
    ) -> None:
        positions = small_model.positions_at(timeslot=0, shell_id=1)
        expected = 72 * 32  # shell 1: 72 orbits × 32 sats
        assert len(positions) == expected

    def test_positions_have_valid_coordinates(
        self, small_model: XMLConstellationModel
    ) -> None:
        positions = small_model.positions_at(timeslot=0, shell_id=1)
        for pos in positions[:10]:  # spot check
            assert -90.0 <= pos.lat_deg <= 90.0
            assert -180.0 <= pos.lon_deg <= 180.0
            assert pos.alt_km > 400.0  # should be near 476km
            assert pos.alt_km < 600.0

    def test_positions_change_over_time(
        self, small_model: XMLConstellationModel
    ) -> None:
        pos_t0 = small_model.positions_at(timeslot=0, shell_id=1)
        pos_t1 = small_model.positions_at(timeslot=1, shell_id=1)
        # At least some positions should differ
        diffs = sum(
            1 for a, b in zip(pos_t0, pos_t1) if a.lat_deg != b.lat_deg
        )
        assert diffs > 0

    def test_invalid_timeslot_raises(
        self, small_model: XMLConstellationModel
    ) -> None:
        with pytest.raises(ValueError, match="out of range"):
            small_model.positions_at(timeslot=99999, shell_id=1)

    def test_positions_array_at_shape(
        self, small_model: XMLConstellationModel
    ) -> None:
        arr = small_model.positions_array_at(timeslot=0, shell_id=1)
        assert arr.shape == (72 * 32, 3)
