"""Tests for :mod:`vantage.control.policy.common.utils`.

Focus: ``find_ingress_satellite`` RNG decoupling (Bug 9 from the
2026-04-17 audit). Pre-fix, the function consumed a process-wide
module-level RNG on every call regardless of ``top_prob`` — which
meant the controller's deterministic ``top_prob=1.0`` path silently
perturbed ``forward.realize``'s stochastic 80/20 selection in the
data plane. Post-fix:

* ``top_prob >= 1.0`` short-circuits to the top-elevation sat with
  zero RNG consumption.
* Callers can pass an explicit ``rng`` to keep their stochastic
  decisions independent of any other module's RNG usage.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from vantage.control.policy.common import utils
from vantage.control.policy.common.utils import find_ingress_satellite
from vantage.domain import AccessLink, Endpoint


@pytest.mark.unit
class TestFindIngressSatelliteRNG:

    def _src(self) -> Endpoint:
        return Endpoint(name="t", lat_deg=0.0, lon_deg=0.0)

    def _sats(self) -> np.ndarray:
        # One zenith-overhead sat — visible regardless of the access
        # model's elevation cutoff.
        return np.array([[0.0, 0.0, 550.0]], dtype=np.float64)

    def _visible(self, n: int = 1) -> list[AccessLink]:
        return [
            AccessLink(sat_id=i, elevation_deg=89.0,
                       slant_range_km=550.0, delay=1.83)
            for i in range(n)
        ]

    def test_top_prob_one_does_not_consume_module_rng(self) -> None:
        """Deterministic call (``top_prob=1.0``) must not advance the
        fallback module RNG."""
        before = utils._RNG.getstate()
        result = find_ingress_satellite(
            self._src(), self._sats(),
            top_prob=1.0, _visible=self._visible(),
        )
        after = utils._RNG.getstate()

        assert result is not None and result.sat_id == 0
        assert before == after, (
            "top_prob=1.0 must short-circuit without touching module RNG; "
            "Bug 9 not fixed"
        )

    def test_explicit_rng_param_does_not_touch_module_rng(self) -> None:
        """When the caller passes its own RNG, the module-level
        ``_RNG`` is untouched — caller controls reproducibility
        independently of any other module's RNG usage."""
        before = utils._RNG.getstate()
        own_rng = random.Random(42)
        find_ingress_satellite(
            self._src(), self._sats(),
            top_prob=0.5, _visible=self._visible(n=3), rng=own_rng,
        )
        after = utils._RNG.getstate()

        assert before == after, (
            "with explicit rng, module-level _RNG must not be touched"
        )

    def test_no_visible_sats_returns_none_without_touching_rng(self) -> None:
        """Empty visibility list returns None without consuming RNG."""
        before = utils._RNG.getstate()
        result = find_ingress_satellite(
            self._src(), self._sats(), _visible=[],
        )
        after = utils._RNG.getstate()

        assert result is None
        assert before == after

    def test_fallback_path_does_consume_module_rng(self) -> None:
        """Inverse of test 1: when no explicit ``rng`` is passed and
        ``top_prob < 1.0``, the module RNG IS advanced — guards
        against a future regression that accidentally bypasses the
        fallback path entirely."""
        before = utils._RNG.getstate()
        find_ingress_satellite(
            self._src(), self._sats(),
            top_prob=0.5, _visible=self._visible(n=2),
        )
        after = utils._RNG.getstate()
        assert before != after, (
            "fallback path should advance module _RNG; if this fails the "
            "module RNG is being silently bypassed"
        )

    def test_explicit_rng_gives_reproducible_choice(self) -> None:
        """Two independent RNGs seeded the same way produce the same
        stochastic decision sequence — basic reproducibility sanity
        check for the new ``rng`` parameter."""
        visible = self._visible(n=3)
        sats = self._sats()
        src = self._src()

        rng_a = random.Random(7)
        rng_b = random.Random(7)
        choices_a = [
            find_ingress_satellite(
                src, sats, top_prob=0.0, _visible=visible, rng=rng_a,
            ).sat_id
            for _ in range(20)
        ]
        choices_b = [
            find_ingress_satellite(
                src, sats, top_prob=0.0, _visible=visible, rng=rng_b,
            ).sat_id
            for _ in range(20)
        ]
        assert choices_a == choices_b
