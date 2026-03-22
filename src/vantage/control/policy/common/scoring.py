"""Scoring framework for candidate path selection.

To add a new strategy: implement the CandidateScorer Protocol and wire
it into a controller (or use CandidateBasedController directly).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from vantage.control.policy.common.candidate import PathCandidate


class CandidateScorer(Protocol):
    """Protocol for scoring candidate paths.

    Returns a score (lower is better) or None to skip the candidate.
    """

    def score(self, candidate: PathCandidate) -> float | None: ...


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------


class SatelliteCostScorer:
    """Score by satellite segment cost only.

    Used for within-PoP path selection by controllers that choose
    PoP via a separate strategy (NearestPoP, GroundOnly, StaticPoP).
    """

    def score(self, candidate: PathCandidate) -> float:
        return candidate.satellite_rtt


class E2EScorer:
    """Score by full end-to-end delay (satellite + ground).

    Candidates without ground_rtt are skipped.
    Used by VantageGreedy for joint optimization.
    """

    def score(self, candidate: PathCandidate) -> float | None:
        if candidate.ground_rtt is None:
            return None
        return candidate.satellite_rtt + candidate.ground_rtt


# Module-level singleton instances for convenience.
satellite_cost_scorer = SatelliteCostScorer()
e2e_scorer_instance = E2EScorer()


def select_best(
    candidates: Iterable[PathCandidate],
    scorer: CandidateScorer,
) -> PathCandidate | None:
    """Select the candidate with the lowest score.

    Returns None if no candidate is viable (all scored None or empty).
    """
    best: PathCandidate | None = None
    best_score = float("inf")
    for c in candidates:
        s = scorer.score(c)
        if s is not None and s < best_score:
            best_score = s
            best = c
    return best
