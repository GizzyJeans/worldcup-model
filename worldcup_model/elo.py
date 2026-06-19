"""World Football Elo ratings.

A self-correcting team-strength rating updated chronologically after every
match. Update magnitude scales with the result, the goal margin, and the
importance of the fixture (a World Cup final moves ratings more than a
friendly). We also learn, in the same pass, how an Elo gap maps to an
expected goal supremacy so Elo can emit its own scoreline expectation that we
blend with the Dixon-Coles model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# K-factor by fixture importance (classic eloratings.net scheme).
TOURNAMENT_K = {
    "fifa world cup": 60.0,
    "copa américa": 50.0,
    "uefa euro": 50.0,
    "african cup of nations": 50.0,
    "afc asian cup": 50.0,
    "gold cup": 50.0,
    "confederations cup": 45.0,
    "fifa world cup qualification": 40.0,
    "uefa nations league": 40.0,
}
_DEFAULT_K = 30.0
_FRIENDLY_K = 20.0


def _k_factor(tournament: str) -> float:
    t = str(tournament).lower()
    if t == "friendly":
        return _FRIENDLY_K
    for key, k in TOURNAMENT_K.items():
        if key in t:
            return k
    if "qualification" in t or "qualifier" in t:
        return 40.0
    return _DEFAULT_K


def _goal_diff_multiplier(goal_diff: int) -> float:
    """Margin-of-victory weighting: bigger wins move ratings more."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


class EloModel:
    def __init__(self, base: float = 1500.0, home_adv: float = 70.0):
        self.base = base
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}
        # Learned Elo-gap -> goal-supremacy slope and league average goals.
        self.supremacy_slope: float = 1.0 / 250.0
        self.avg_total_goals: float = 2.6

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def fit(self, played: pd.DataFrame) -> "EloModel":
        ratings: dict[str, float] = {}
        gaps: list[float] = []
        margins: list[float] = []

        for row in played.itertuples(index=False):
            h, a = row.home_team, row.away_team
            rh = ratings.get(h, self.base)
            ra = ratings.get(a, self.base)
            ha = 0.0 if row.neutral else self.home_adv
            dr = (rh + ha) - ra

            # Expected score for the home side, then realised result.
            we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
            gd = int(row.home_score) - int(row.away_score)
            w = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)

            k = _k_factor(row.tournament) * _goal_diff_multiplier(gd)
            delta = k * (w - we)
            ratings[h] = rh + delta
            ratings[a] = ra - delta

            gaps.append(dr)
            margins.append(gd)

        self.ratings = ratings

        # Calibrate Elo gap -> expected goal supremacy (slope through origin)
        # and the league-average total goals, used by `expected_goals`.
        gaps_arr = np.asarray(gaps)
        margins_arr = np.asarray(margins)
        denom = float(np.dot(gaps_arr, gaps_arr))
        if denom > 0:
            self.supremacy_slope = float(np.dot(gaps_arr, margins_arr) / denom)
        self.avg_total_goals = float(
            (played["home_score"] + played["away_score"]).mean()
        )
        return self

    def expected_goals(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float]:
        """Elo-implied (home, away) expected goals via supremacy + avg total."""
        ha = 0.0 if neutral else self.home_adv
        dr = (self.rating(home) + ha) - self.rating(away)
        supremacy = self.supremacy_slope * dr
        total = self.avg_total_goals
        lam = max(0.05, (total + supremacy) / 2.0)
        mu = max(0.05, (total - supremacy) / 2.0)
        return lam, mu

    def top(self, n: int = 20) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)[:n]
