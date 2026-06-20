"""Walk-forward backtest of the picking method on ALREADY-PLAYED matches.

For every match in the test window we refit the expert model on results strictly
earlier than that match's date (no look-ahead), then score the forecasts it would
have made against what actually happened:

    1X2     does the model's favourite win? calibration, Brier, log-loss.
    O/U 2.5 does the model's over/under lean match the actual goal total?
    AH      does the model's favoured side cover standard handicap lines, and
            does the underdog cover the mirror lines?

What this answers: did the method's DIRECTION beat a coin-flip / the baseline,
and was it calibrated? Hit rate ("過盤率") is real. PROFIT, however, needs the
prices we'd have actually taken — this results-only dataset has none for the WC,
so the ROI columns are an explicit proxy: bets struck at the model's own fair
odds (1/p), i.e. a perfectly efficient market. That isolates calibration P&L;
a real book's margin makes it worse, never better. For a true betting/CLV test,
log live prices with `picks.py scan` and `settle` them once the lines close.

    python wc_backtest.py [--tournament "FIFA World Cup"] [--from 2026-06-01]
                          [--since 2014-01-01] [--ah-price 1.91]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from worldcup_model import data as datamod
from worldcup_model.markets import score_matrix
from worldcup_model.model import ExpertModel
from worldcup_model.paths import MODEL_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ---- handicap helpers -----------------------------------------------------
def margin_dist(mat: np.ndarray) -> dict[int, float]:
    """P(home_goals - away_goals = k) from a scoreline matrix."""
    d: dict[int, float] = defaultdict(float)
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            d[i - j] += mat[i, j]
    return d


def ah_settle(margin: float, line: float) -> float:
    """Asian-handicap result for a side with goal `margin` and handicap `line`.

    1.0 win, 0.0 loss, 0.5 push (stake refunded); quarter lines split half/half."""
    q = round(line * 4) % 4
    if q in (0, 2):  # integer or half line
        v = margin + line
        return 1.0 if v > 1e-9 else (0.5 if abs(v) < 1e-9 else 0.0)
    return (ah_settle(margin, line - 0.25) + ah_settle(margin, line + 0.25)) / 2


def ah_cover_prob(md: dict[int, float], line: float) -> float:
    """Model P(side covers | not push) for a side getting `line`, push excluded."""
    win = loss = 0.0
    for m, p in md.items():
        v = m + line
        if v > 1e-9:
            win += p
        elif v < -1e-9:
            loss += p
    return win / (win + loss) if (win + loss) > 0 else 0.0


# ---- walk-forward ---------------------------------------------------------
def walk_forward(played: pd.DataFrame, test: pd.DataFrame, cfg: dict,
                 neutral: bool) -> pd.DataFrame:
    rows = []
    for dt, grp in test.groupby("date"):
        train = played[played["date"] < dt]
        if len(train) < 200:
            continue
        m = ExpertModel(dc_weight=cfg["dc_weight"],
                        half_life_days=cfg["half_life_days"],
                        host_adv=cfg["host_adv"])
        m.fit(datamod.MatchData(played=train, upcoming=played.iloc[0:0]),
              asof=pd.Timestamp(dt))
        for _, r in grp.iterrows():
            h, a = r["home_team"], r["away_team"]
            try:
                lam, mu = m.expected_goals(h, a, neutral=neutral)
            except (KeyError, ValueError):
                continue  # a team with no fitted rating yet
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            pred = m.predict(h, a, neutral=neutral, ou_lines=(2.5,))
            p1, pou = pred["1x2"], pred["over_under"]["2.5"]
            mat = score_matrix(lam, mu, m.dc.rho, max_goals=12)
            md = margin_dist(mat)
            res = "home" if hs > as_ else ("away" if as_ > hs else "draw")
            fav = max(p1, key=p1.get)
            fav_home = p1["home"] >= p1["away"]
            md_fav = md if fav_home else {-k: v for k, v in md.items()}
            ou_pick = "over" if pou["over"] > 0.5 else "under"
            ou_res = "over" if hs + as_ > 2.5 else "under"
            rows.append(dict(
                date=dt.date(), home=h, away=a, score=f"{hs}-{as_}", res=res,
                p_home=p1["home"], p_draw=p1["draw"], p_away=p1["away"],
                fav=fav, fav_p=p1[fav], fav_hit=int(fav == res),
                p_over=pou["over"], ou_pick=ou_pick, ou_hit=int(ou_pick == ou_res),
                total=hs + as_, fav_margin=(hs - as_) if fav_home else (as_ - hs),
                md_fav=md_fav))
    return pd.DataFrame(rows)


# ---- reporting ------------------------------------------------------------
def fair_roi(hit: np.ndarray, prob: np.ndarray) -> float:
    """ROI of flat bets struck at the model's own fair odds 1/p (push-free)."""
    price = 1.0 / np.clip(prob, 1e-9, 1.0)
    return float((hit * (price - 1) - (1 - hit)).mean())


def report(df: pd.DataFrame, ah_price: float) -> None:
    n = len(df)
    print(f"\n=== Walk-forward backtest: {n} matches "
          f"(each refit on prior results only) ===")

    print("\n-- 1X2 (match result) --")
    fav_hit = df["fav_hit"].mean()
    print(f"  model favourite hit rate : {fav_hit:.1%}  ({df['fav_hit'].sum()}/{n})"
          f"   break-even odds {1/fav_hit:.2f}" if fav_hit else "  n/a")
    print(f"  avg favourite prob       : {df['fav_p'].mean():.1%} "
          f"(>{fav_hit:.0%} actual => overconfident)")
    print(f"  baseline 'always home'   : {(df['res']=='home').mean():.1%}")
    y = df["res"].map({"home": 0, "draw": 1, "away": 2}).values
    P = df[["p_home", "p_draw", "p_away"]].values
    P = P / P.sum(1, keepdims=True)
    onehot = np.eye(3)[y]
    brier = ((P - onehot) ** 2).sum(1).mean()
    logloss = -np.log(np.clip(P[np.arange(n), y], 1e-9, 1)).mean()
    print(f"  Brier {brier:.4f}   LogLoss {logloss:.4f}")
    print(f"  fair-odds ROI (favourite): {fair_roi(df['fav_hit'].values, df['fav_p'].values):+.1%}")

    print("\n-- Over/Under 2.5 --")
    print(f"  model direction hit rate : {df['ou_hit'].mean():.1%} "
          f"({df['ou_hit'].sum()}/{n})")
    print(f"  actual over-2.5 rate     : {(df['total']>2.5).mean():.1%}"
          f"   avg goals {df['total'].mean():.2f}")
    p_pick = np.where(df["ou_pick"] == "over", df["p_over"], 1 - df["p_over"])
    print(f"  fair-odds ROI (O/U lean) : {fair_roi(df['ou_hit'].values, p_pick):+.1%}")

    print("\n-- Calibration (favourite prob bucket -> actual) --")
    for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
        s = df[(df["fav_p"] >= lo) & (df["fav_p"] < hi)]
        if len(s):
            print(f"  {lo:.0%}-{hi:.0%}: n={len(s):2d}  "
                  f"predicted~{s['fav_p'].mean():.0%}  actual {s['fav_hit'].mean():.0%}")

    print(f"\n-- Asian handicap (cover rate; ROI struck at {ah_price:.2f}) --")
    print(f"  {'line':<16}{'cover':>7}{'model':>7}{'ROI@'+format(ah_price,'.2f'):>9}"
          f"{'W/L/P':>10}")
    def ah_roi(s: np.ndarray) -> float:
        return float(np.array([(ah_price - 1) if x > 0.5
                               else (0.0 if np.isclose(x, 0.5) else -1.0)
                               for x in s]).mean())

    for L in (-0.5, -1.0, -1.5, -2.0):
        s = np.array([ah_settle(m, L) for m in df["fav_margin"]])
        pr = np.array([ah_cover_prob(md, L) for md in df["md_fav"]])
        win, loss = int((s > 0.5).sum()), int((s < 0.5).sum())
        push = int(np.isclose(s, 0.5).sum())
        cover = win / (win + loss) if (win + loss) else 0.0
        print(f"  {'favourite '+format(L,'+g'):<16}{cover:>7.0%}{pr.mean():>7.0%}"
              f"{ah_roi(s):>+9.1%}{f'{win}/{loss}/{push}':>10}")
    for L in (0.5, 1.0, 1.5):
        s = np.array([ah_settle(-m, L) for m in df["fav_margin"]])  # underdog margin
        win, loss = int((s > 0.5).sum()), int((s < 0.5).sum())
        push = int(np.isclose(s, 0.5).sum())
        cover = win / (win + loss) if (win + loss) else 0.0
        print(f"  {'underdog '+format(L,'+g'):<16}{cover:>7.0%}{'—':>7}"
              f"{ah_roi(s):>+9.1%}{f'{win}/{loss}/{push}':>10}")

    print("\nNOTE: ROI columns are a proxy (fair odds / a fixed AH price), not real "
          "prices.\n      Hit/cover rates are real. Log live odds with picks.py for a "
          "true CLV test.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tournament", default="FIFA World Cup",
                    help="substring filter on the tournament column")
    ap.add_argument("--from", dest="from_date", default="2026-06-01",
                    help="test-window start (ISO date)")
    ap.add_argument("--to", dest="to_date", default=None,
                    help="test-window end (ISO date), inclusive")
    ap.add_argument("--since", default="2014-01-01",
                    help="earliest history to fit on (model recency floor)")
    ap.add_argument("--ah-price", type=float, default=1.91,
                    help="assumed decimal price for the handicap ROI column")
    ap.add_argument("--venue", choices=["neutral", "home"], default="neutral",
                    help="treat fixtures as neutral-venue (World Cup) or home/away")
    ap.add_argument("--detail", action="store_true",
                    help="print the per-match prediction table")
    args = ap.parse_args()

    cfg = json.load(open(MODEL_PATH))
    played = datamod.load(since=args.since).played
    test = played[played["tournament"].str.contains(args.tournament, case=False,
                                                     na=False)]
    test = test[test["date"] >= pd.Timestamp(args.from_date)]
    if args.to_date:
        test = test[test["date"] <= pd.Timestamp(args.to_date)]
    test = test.sort_values("date").reset_index(drop=True)
    if test.empty:
        raise SystemExit(f"No played '{args.tournament}' matches in the window.")

    df = walk_forward(played, test, cfg, neutral=(args.venue == "neutral"))
    if df.empty:
        raise SystemExit("No matches scored (teams lacked fitted ratings).")
    if args.detail:
        cols = ["date", "home", "away", "score", "res", "fav", "fav_p",
                "fav_hit", "ou_pick", "ou_hit"]
        pd.set_option("display.width", 200)
        print(df[cols].to_string(index=False))
    report(df, args.ah_price)


if __name__ == "__main__":
    main()
