"""Profitability backtest: bet the model's edges into Pinnacle CLOSING odds.

This is the real test. Pinnacle's closing line is the sharpest, lowest-margin
price in the market; if the model cannot profit betting INTO the close, its
"edges" are not real. We walk forward with no look-ahead (refit the model only
on matches before each fixture's date), place a bet whenever the model sees
value, settle on the actual result, and report yield with a bootstrap CI.

Odds source: eatpizzanot/soccer-dataset (Pinnacle closing, API-Football).
Joined to martj42 results for the neutral-venue flag and team vocabulary.

    python roi_backtest.py [--refit-days 180] [--min-ev 0.03] [--max-edge 0.12]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from worldcup_model import data as datamod
from worldcup_model.markets import markets
from worldcup_model.model import ExpertModel
from worldcup_model.paths import ODDS_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

INTL_LEAGUES = {78, 80, 82, 84, 85, 86, 87, 88}
NAME_FIX = {"Cape Verde Islands": "Cape Verde", "Congo DR": "DR Congo", "USA": "United States"}
SOFT_BOOKS = ["William Hill", "Bet365", "1xBet", "Marathon Bet"]  # recreational books


def _select_odds(od: pd.DataFrame, fxi_ids, book: str) -> pd.DataFrame:
    """Return one (fixture_id, oh, od, oa) row per fixture for the chosen book.

    pinnacle-close: Pinnacle's closing line (sharpest benchmark).
    william-hill:   a single soft/recreational book.
    best-soft:      line-shopping — best available price per outcome across the
                    recreational books (what a value bettor actually gets)."""
    o = od[od["fixture_id"].isin(fxi_ids)].copy()
    if book == "pinnacle-close":
        s = o[(o["source"] == "API-Football-closing") & (o["bookmaker"] == "Pinnacle")]
        s = s.drop_duplicates("fixture_id")
    elif book == "william-hill":
        s = o[o["bookmaker"] == "William Hill"].groupby("fixture_id", as_index=False).mean(
            numeric_only=True)
    elif book == "best-soft":
        s = o[o["bookmaker"].isin(SOFT_BOOKS)].groupby("fixture_id", as_index=False).agg(
            home_win=("home_win", "max"), draw=("draw", "max"), away_win=("away_win", "max"))
    else:
        raise ValueError(book)
    return s[["fixture_id", "home_win", "draw", "away_win"]]


def build_odds_table(book: str = "pinnacle-close") -> pd.DataFrame:
    """Merge chosen-book odds with martj42 results (for neutral + names)."""
    fx = pd.read_csv(os.path.join(ODDS_DIR, "fixtures.csv"))
    fx["date"] = pd.to_datetime(fx["date"], utc=True).dt.tz_localize(None).dt.normalize()
    od = pd.read_csv(os.path.join(ODDS_DIR, "odds.csv"))
    tm = pd.read_csv(os.path.join(ODDS_DIR, "teams.csv"))[["id", "name"]]
    name = dict(zip(tm.id, tm.name))

    fxi = fx[fx["league_id"].isin(INTL_LEAGUES) & fx["goals_home"].notna()].copy()
    sel = _select_odds(od, set(fxi["id"]), book)
    m = fxi.merge(sel, left_on="id", right_on="fixture_id")
    m["home"] = m["home_team_id"].map(name).replace(NAME_FIX)
    m["away"] = m["away_team_id"].map(name).replace(NAME_FIX)
    m = m[["date", "home", "away", "goals_home", "goals_away", "home_win", "draw", "away_win"]]
    m.columns = ["date", "home", "away", "gh", "ga", "oh", "od", "oa"]

    # Attach neutral-venue flag from martj42 by (home, away) within +/-1 day.
    mart = datamod.load(since="2014-01-01").played[
        ["date", "home_team", "away_team", "neutral", "tournament"]]
    j = m.merge(mart, left_on=["home", "away"], right_on=["home_team", "away_team"],
                suffixes=("", "_m"))
    j = j[(j["date"] - j["date_m"]).abs() <= pd.Timedelta(days=2)]
    j = j.drop_duplicates(subset=["date", "home", "away"])
    return j[["date", "home", "away", "neutral", "tournament", "gh", "ga", "oh", "od", "oa"]].sort_values("date").reset_index(drop=True)


def bootstrap_ci(profits: np.ndarray, stakes: np.ndarray, n: int = 5000) -> tuple[float, float]:
    """95% bootstrap CI on yield = sum(profit)/sum(stake)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(profits), size=(n, len(profits)))
    ys = profits[idx].sum(1) / stakes[idx].sum(1)
    return float(np.percentile(ys, 2.5)), float(np.percentile(ys, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit-days", type=int, default=180)
    ap.add_argument("--min-ev", type=float, default=0.03)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--max-edge", type=float, default=0.12, help="game-level sanity skip")
    ap.add_argument("--no-sanity", action="store_true", help="disable the max-edge filter")
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--book", default="pinnacle-close",
                    choices=["pinnacle-close", "william-hill", "best-soft"])
    args = ap.parse_args()

    odds = build_odds_table(args.book)
    played = datamod.load(since="2014-01-01").played
    print(f"Book: {args.book}  |  Test set: {len(odds)} internationals, "
          f"{odds['date'].min().date()}..{odds['date'].max().date()}")
    over = np.array([1 / odds["oh"], 1 / odds["od"], 1 / odds["oa"]]).sum(0).mean() - 1
    print(f"Avg overround (vig): {over:.1%}\n")

    bets = []          # (profit_per_unit, edge, won)
    bankroll, peak, maxdd = 100.0, 100.0, 0.0
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
        p = markets(lam, mu, model.dc.rho)["1x2"]
        decs = {"H": r.oh, "D": r.od, "A": r.oa}
        probs = {"H": p["home"], "D": p["draw"], "A": p["away"]}
        result = "H" if r.gh > r.ga else ("A" if r.gh < r.ga else "D")

        # Sanity filter: if model grossly disagrees with the close on any
        # outcome, distrust the whole match.
        if not args.no_sanity:
            if max(abs(probs[s] - 1 / decs[s]) for s in "HDA") > args.max_edge:
                continue

        # Bet the single best-EV value selection.
        best = max("HDA", key=lambda s: probs[s] * decs[s] - 1)
        ev = probs[best] * decs[best] - 1
        edge = probs[best] - 1 / decs[best]
        if ev < args.min_ev or edge < args.min_edge:
            continue

        won = result == best
        profit = (decs[best] - 1) if won else -1.0
        bets.append((profit, edge, won))

        f = min(max(0.0, ev / (decs[best] - 1)) * args.kelly, 0.05)
        bankroll *= 1 + f * (decs[best] - 1) if won else 1 - f
        peak = max(peak, bankroll)
        maxdd = max(maxdd, (peak - bankroll) / peak)

    if not bets:
        print("No value bets were placed.")
        return
    profit = np.array([b[0] for b in bets])
    won = np.array([b[2] for b in bets])
    stakes = np.ones_like(profit)
    yld = profit.sum() / stakes.sum()
    lo, hi = bootstrap_ci(profit, stakes)

    tag = "OFF" if args.no_sanity else f"ON (max-edge {args.max_edge:.0%})"
    print(f"Sanity filter: {tag}")
    print(f"Bets placed:        {len(bets)}  of {len(odds)} matches")
    print(f"Win rate:           {won.mean():.1%}")
    print(f"Avg odds taken:     {(profit[won] + 1).mean():.2f}")
    print(f"Flat-stake YIELD:   {yld:+.1%}   (95% CI {lo:+.1%} .. {hi:+.1%})")
    print(f"Profit (flat 1u):   {profit.sum():+.1f}u on {len(bets)}u staked")
    print(f"{args.kelly:g}x-Kelly bankroll: 100 -> {bankroll:.1f}   (max drawdown {maxdd:.1%})")
    print()
    verdict = ("EDGE vs the closing line (but check the CI / sample size)" if lo > 0
               else "NO proven edge vs the closing line — CI includes 0 / negative")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
