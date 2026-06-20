"""Turn (lambda, mu, rho) into a scoreline matrix and betting-market probs."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .dixon_coles import _tau


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Joint P(home=i, away=j) with the Dixon-Coles low-score correction."""
    h = np.arange(max_goals + 1)
    ph = poisson.pmf(h, lam)
    pa = poisson.pmf(h, mu)
    mat = np.outer(ph, pa)

    # Apply the rho correction to the 2x2 low-score block.
    hh = np.array([0, 0, 1, 1])
    aa = np.array([0, 1, 0, 1])
    corr = _tau(hh, aa, np.full(4, lam), np.full(4, mu), rho)
    mat[0, 0] *= corr[0]
    mat[0, 1] *= corr[1]
    mat[1, 0] *= corr[2]
    mat[1, 1] *= corr[3]

    s = mat.sum()
    if s > 0:
        mat /= s
    return mat


def markets(lam: float, mu: float, rho: float, ou_lines=(2.5,), max_goals: int = 10) -> dict:
    """All supported market probabilities from a single scoreline matrix."""
    mat = score_matrix(lam, mu, rho, max_goals)
    n = mat.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]

    home_win = float(mat[i > j].sum())
    draw = float(np.trace(mat))
    away_win = float(mat[i < j].sum())

    out: dict[str, object] = {
        "expected_goals": {"home": round(lam, 3), "away": round(mu, 3)},
        "1x2": {"home": home_win, "draw": draw, "away": away_win},
        "double_chance": {
            "1X": home_win + draw,
            "12": home_win + away_win,
            "X2": draw + away_win,
        },
        "btts": {
            "yes": float(mat[1:, 1:].sum()),
            "no": float(mat[0, :].sum() + mat[:, 0].sum() - mat[0, 0]),
        },
    }

    total = i + j
    ou = {}
    for line in ou_lines:
        over = float(mat[total > line].sum())
        ou[str(line)] = {"over": over, "under": 1.0 - over}
    out["over_under"] = ou

    # Most likely correct scores.
    flat = [((r, c), float(mat[r, c])) for r in range(n) for c in range(n)]
    flat.sort(key=lambda kv: kv[1], reverse=True)
    out["correct_score"] = {f"{r}-{c}": p for (r, c), p in flat[:6]}
    return out


def asian_handicap(lam: float, mu: float, rho: float, home_line: float,
                   max_goals: int = 10) -> dict[str, float]:
    """Push-excluded cover probabilities for a home-perspective handicap line.

    Returns {"home": P(home covers `home_line`), "away": P(away covers the
    mirror `-home_line`)}, normalised over the non-push outcomes (a refunded
    push leaves an even-money bet, so it drops out of the cover probability).
    Quarter lines (e.g. -1.25) settle as two half-bets and are averaged."""
    if round(home_line * 4) % 2 == 1:  # quarter line -> mean of adjacent lines
        lo = asian_handicap(lam, mu, rho, home_line - 0.25, max_goals)
        hi = asian_handicap(lam, mu, rho, home_line + 0.25, max_goals)
        return {k: (lo[k] + hi[k]) / 2 for k in lo}
    mat = score_matrix(lam, mu, rho, max_goals)
    n = mat.shape[0]
    margin = np.subtract.outer(np.arange(n), np.arange(n)) + home_line
    win = float(mat[margin > 1e-9].sum())
    loss = float(mat[margin < -1e-9].sum())
    s = win + loss
    if s <= 0:
        return {"home": 0.5, "away": 0.5}
    return {"home": win / s, "away": loss / s}
