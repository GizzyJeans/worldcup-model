"""Does the model carry independent signal beyond the market? Find the blend.

The model disagrees with the market on some teams' base strength (it under-
rated host USA, and Turkey). The principled question is not "hand-tune those
ratings" but: *when the model deviates from the market, is it ever right?*

We answer it with forecast combination. For each international with Pinnacle
closing odds we have walk-forward model probabilities and the de-vigged market
probabilities. We blend them in log space,

    p_blend ∝ p_model^(1-w) · p_market^w ,

and fit the weight w that minimises out-of-sample log-loss:
    w -> 0  the model is all you need;
    w -> 1  the market already contains everything the model knows.

The gap between the best blend and market-only is the model's added value.

    python market_blend.py [--refit-days 180]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import optimize

from roi_backtest import build_odds_table
from worldcup_model import data as datamod
from worldcup_model.markets import markets
from worldcup_model.model import ExpertModel

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())


def blend(pm: np.ndarray, pk: np.ndarray, w: float) -> np.ndarray:
    lp = (1 - w) * np.log(np.clip(pm, 1e-9, 1)) + w * np.log(np.clip(pk, 1e-9, 1))
    e = np.exp(lp - lp.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit-days", type=int, default=180)
    args = ap.parse_args()

    odds = build_odds_table("pinnacle-close")
    played = datamod.load(since="2014-01-01").played

    pm_list, pk_list, ys = [], [], []
    refit, model = None, None
    for r in odds.itertuples(index=False):
        if model is None or r.date >= refit:
            train = played[played["date"] < r.date]
            model = ExpertModel().fit(
                datamod.MatchData(played=train, upcoming=train.iloc[0:0]), asof=r.date)
            refit = r.date + pd.Timedelta(days=args.refit_days)
        if r.home not in model.dc.attack or r.away not in model.dc.attack:
            continue
        host = (str(r.tournament) == "FIFA World Cup") and not bool(r.neutral)
        lam, mu = model.expected_goals(r.home, r.away, bool(r.neutral), host)
        mp = markets(lam, mu, model.dc.rho)["1x2"]
        raw = np.array([1 / r.oh, 1 / r.od, 1 / r.oa])
        pm_list.append([mp["home"], mp["draw"], mp["away"]])
        pk_list.append(raw / raw.sum())              # de-vigged market probs
        ys.append(0 if r.gh > r.ga else (2 if r.gh < r.ga else 1))

    pm, pk, y = np.array(pm_list), np.array(pk_list), np.array(ys)
    n = len(y)
    res = optimize.minimize_scalar(lambda w: logloss(blend(pm, pk, w), y),
                                   bounds=(0, 1), method="bounded")
    w = float(res.x)

    print(f"Matches with model + Pinnacle-closing probs: {n}\n")
    print(f"  {'forecaster':<22}{'log-loss':>10}{'Brier':>9}")
    print("  " + "-" * 41)
    for name, p in [("Model only (w=0)", pm),
                    ("Market only (w=1)", pk),
                    (f"Best blend (w={w:.2f})", blend(pm, pk, w))]:
        oh = np.eye(3)[y]
        print(f"  {name:<22}{logloss(p, y):>10.4f}{((p-oh)**2).sum(1).mean():>9.4f}")

    gain = logloss(pk, y) - logloss(blend(pm, pk, w), y)
    print(f"\n  Optimal market weight w* = {w:.2f}")
    print(f"  Blend beats market-only by {gain:+.4f} log-loss "
          f"({'model adds signal' if gain > 1e-3 else 'model adds ~nothing'})")
    print("\n  Interpretation: w* near 1 => trust the price; the model's job is\n"
          "  analysis, not betting against the close.")


if __name__ == "__main__":
    main()
