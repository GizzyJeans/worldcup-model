"""World Cup host-nation advantage.

The generic home-advantage term under-rates a World Cup *host*: the crowd,
familiarity, travel and refereeing edges of hosting a home World Cup are much
larger than an ordinary home game. We estimate that extra effect from the full
history of World Cup host games (how much hosts beat their rating + generic
home advantage) and apply it as a supremacy bonus on top of the normal model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import _goal_diff_multiplier, _k_factor

# 2026 hosts. A host game = one of these playing a (non-neutral) home WC match.
HOSTS_2026 = {"United States", "Canada", "Mexico"}


def is_host_game(home: str, neutral: bool) -> bool:
    return (home in HOSTS_2026) and (not neutral)


def estimate_host_advantage(
    played: pd.DataFrame, base: float = 1500.0, home_adv: float = 70.0
) -> tuple[float, float, int]:
    """Estimate the WC host supremacy bonus (goals) from full history.

    Replays Elo chronologically and, on World Cup host home games, measures the
    residual between actual and rating-expected goal difference (where the
    rating expectation already includes the *generic* home advantage). The mean
    residual is the extra host effect. Returns (bonus_goals, std_error, n)."""
    ratings: dict[str, float] = {}
    gaps, margins = [], []
    host_gaps, host_margins = [], []
    for r in played.itertuples(index=False):
        rh = ratings.get(r.home_team, base)
        ra = ratings.get(r.away_team, base)
        dr = (rh + (0.0 if r.neutral else home_adv)) - ra
        gd = int(r.home_score) - int(r.away_score)
        gaps.append(dr)
        margins.append(gd)
        if str(r.tournament) == "FIFA World Cup" and not r.neutral:
            host_gaps.append(dr)
            host_margins.append(gd)
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        w = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        d = _k_factor(r.tournament) * _goal_diff_multiplier(gd) * (w - we)
        ratings[r.home_team] = rh + d
        ratings[r.away_team] = ra - d

    g, m = np.array(gaps), np.array(margins)
    slope = float((g @ m) / (g @ g))  # goals per Elo point
    hg, hm = np.array(host_gaps), np.array(host_margins)
    resid = hm - slope * hg
    n = len(resid)
    se = float(resid.std() / np.sqrt(n)) if n else 0.0
    return float(resid.mean()), se, n
