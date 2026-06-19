"""Tune the model's blend / recency hyperparameters on the walk-forward backtest.

The expert model has two free hyperparameters that were never fit from data:

    dc_weight       Dixon-Coles vs Elo weight at the expected-goals level
    half_life_days  exponential recency half-life of the Dixon-Coles fit

This script grid-searches them with the same no-look-ahead, walk-forward refit
used by `backtest.py`, scoring 1X2 forecasts with proper metrics (log-loss,
Brier, RPS). To avoid tuning on the data we then report on, it splits the test
window in two: hyperparameters are chosen on the earlier *tune* slice and the
winner is confirmed on the held-out *validation* slice against the current
defaults.

Cheap by design: `dc_weight` is only a blend coefficient applied at predict
time, so for each `half_life` we refit once per window, cache each match's
Dixon-Coles and Elo goal expectations, and then sweep every `dc_weight` in
memory for free.

    python tune.py [--since 2014-01-01] [--test-start 2018-01-01]
                   [--val-start 2023-01-01] [--refit-days 180]
                   [--half-lives 365,540,730,1095]
                   [--dc-weights 0.5,0.6,0.7,0.8,0.9,1.0]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from worldcup_model import data as datamod
from worldcup_model.dixon_coles import DixonColes
from worldcup_model.elo import EloModel
from worldcup_model.markets import markets

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

DEFAULT_DC_WEIGHT = 0.7
DEFAULT_HALF_LIFE = 540.0
HOST_ADV = 0.39  # World Cup host goal-supremacy bonus (see worldcup_model/host.py)


def result_1x2(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (2 if hs < as_ else 1)


def score(probs: np.ndarray, y: np.ndarray) -> dict:
    """probs: (N,3) in H,D,A order; y: (N,) class index. Lower is better
    except accuracy."""
    eps = 1e-15
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[y]
    logloss = float(-np.log(p[np.arange(len(y)), y]).mean())
    brier = float(((p - onehot) ** 2).sum(axis=1).mean())
    cp, cy = np.cumsum(p, axis=1), np.cumsum(onehot, axis=1)
    rps = float((((cp - cy) ** 2)[:, :2].sum(axis=1) / 2).mean())
    acc = float((p.argmax(axis=1) == y).mean())
    return {"logloss": logloss, "brier": brier, "rps": rps, "acc": acc, "n": len(y)}


def walk_forward(played: pd.DataFrame, test_start: pd.Timestamp,
                 refit_days: int, half_life: float) -> list[dict]:
    """Refit every `refit_days` and cache each test match's goal components.

    Returns one record per scored match with the Dixon-Coles goals (ld, md),
    the Elo goals (le, me), the fit's rho, the host flag, the 1X2 outcome and
    the match date — enough to reconstruct any-dc_weight blended probabilities
    without refitting.
    """
    empty = played.iloc[0:0]
    rows: list[dict] = []
    refit = test_start
    max_date = played["date"].max()
    while refit <= max_date:
        train = played[played["date"] < refit]
        elo = EloModel().fit(train)
        w = datamod.time_weights(train["date"], refit, half_life)
        dc = DixonColes().fit(train, w)
        chunk = played[(played["date"] >= refit) &
                       (played["date"] < refit + pd.Timedelta(days=refit_days))]
        for r in chunk.itertuples(index=False):
            h, a = r.home_team, r.away_team
            if h not in dc.attack or a not in dc.attack:
                continue
            neu = bool(r.neutral)
            host = (str(r.tournament) == "FIFA World Cup") and not neu
            ld, md = dc.expected_goals(h, a, neu)
            le, me = elo.expected_goals(h, a, neu)
            rows.append({
                "date": r.date, "ld": ld, "md": md, "le": le, "me": me,
                "rho": dc.rho, "host": host,
                "y": result_1x2(int(r.home_score), int(r.away_score)),
            })
        refit += pd.Timedelta(days=refit_days)
    return rows


def blended_probs(rows: list[dict], dc_weight: float) -> np.ndarray:
    """1X2 probabilities for every cached match at a given dc_weight."""
    out = np.empty((len(rows), 3))
    s = HOST_ADV / 2.0
    for k, r in enumerate(rows):
        lam = dc_weight * r["ld"] + (1 - dc_weight) * r["le"]
        mu = dc_weight * r["md"] + (1 - dc_weight) * r["me"]
        if r["host"]:
            lam, mu = lam + s, mu - s
        lam, mu = max(0.05, lam), max(0.05, mu)
        m = markets(lam, mu, r["rho"])["1x2"]
        out[k] = (m["home"], m["draw"], m["away"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2014-01-01")
    ap.add_argument("--test-start", default="2018-01-01")
    ap.add_argument("--val-start", default="2023-01-01",
                    help="matches on/after this date are the held-out validation slice")
    ap.add_argument("--refit-days", type=int, default=180)
    ap.add_argument("--half-lives", default="365,540,730,1095")
    ap.add_argument("--dc-weights", default="0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--metric", default="rps", choices=["rps", "logloss", "brier"],
                    help="selection metric on the tune slice (lower is better)")
    args = ap.parse_args()

    half_lives = [float(x) for x in args.half_lives.split(",")]
    dc_weights = [float(x) for x in args.dc_weights.split(",")]
    test_start = pd.Timestamp(args.test_start)
    val_start = pd.Timestamp(args.val_start)

    md = datamod.load(since=args.since)
    played = md.played
    print(f"Loaded {len(played):,} played matches "
          f"({played['date'].min().date()}..{played['date'].max().date()}).")
    print(f"Tune slice  : {test_start.date()} .. {(val_start - pd.Timedelta(days=1)).date()}")
    print(f"Validation  : {val_start.date()} .. {played['date'].max().date()}")
    print(f"Grid        : half_life {half_lives}  x  dc_weight {dc_weights}\n")

    # Always evaluate the current default so the comparison has a baseline,
    # even when the refined grid does not include it.
    if DEFAULT_HALF_LIFE not in half_lives:
        half_lives = half_lives + [DEFAULT_HALF_LIFE]
    if DEFAULT_DC_WEIGHT not in dc_weights:
        dc_weights = dc_weights + [DEFAULT_DC_WEIGHT]

    # Cache walk-forward goal components once per half_life, then sweep
    # dc_weight in memory. Split the cached rows into tune vs validation.
    results = []  # (half_life, dc_weight, tune_metrics, val_metrics)
    cache: dict[float, list[dict]] = {}
    for hl in half_lives:
        rows = walk_forward(played, test_start, args.refit_days, hl)
        cache[hl] = rows
        dates = np.array([r["date"] for r in rows])
        y = np.array([r["y"] for r in rows])
        tune_mask = dates < val_start
        val_mask = ~tune_mask
        for w in dc_weights:
            P = blended_probs(rows, w)
            results.append((hl, w,
                            score(P[tune_mask], y[tune_mask]),
                            score(P[val_mask], y[val_mask])))

    # ---- ranking on the tune slice ------------------------------------
    print(f"  {'half_life':>10}{'dc_weight':>11}{'  | tune ':>0}"
          f"{'logloss':>10}{'brier':>9}{'rps':>9}{'acc':>8}")
    print("  " + "-" * 57)
    ranked = sorted(results, key=lambda r: r[2][args.metric])
    for hl, w, tm, _ in ranked:
        flag = ""
        if hl == DEFAULT_HALF_LIFE and w == DEFAULT_DC_WEIGHT:
            flag = "  <- current default"
        print(f"  {hl:>10.0f}{w:>11.2f}  {tm['logloss']:>9.4f}{tm['brier']:>9.4f}"
              f"{tm['rps']:>9.4f}{tm['acc']:>8.1%}{flag}")

    best = ranked[0]
    base = next(r for r in results
                if r[0] == DEFAULT_HALF_LIFE and r[1] == DEFAULT_DC_WEIGHT)

    # ---- confirm on held-out validation -------------------------------
    print(f"\nHeld-out validation ({best[3]['n']:,} matches), selection metric = {args.metric}:")
    print(f"  {'config':<28}{'logloss':>10}{'brier':>9}{'rps':>9}{'acc':>8}")
    print("  " + "-" * 64)
    bl = f"current  hl={base[0]:.0f} w={base[1]:.2f}"
    bw = f"tuned    hl={best[0]:.0f} w={best[1]:.2f}"
    for name, r in ((bl, base), (bw, best)):
        vm = r[3]
        print(f"  {name:<28}{vm['logloss']:>10.4f}{vm['brier']:>9.4f}"
              f"{vm['rps']:>9.4f}{vm['acc']:>8.1%}")

    d_ll = best[3]["logloss"] - base[3]["logloss"]
    d_rps = best[3]["rps"] - base[3]["rps"]
    print(f"\n  delta (tuned - current):  log-loss {d_ll:+.4f}   rps {d_rps:+.4f}")
    if best[0] == base[0] and best[1] == base[1]:
        print("  Current defaults are already optimal on the tune slice.")
    elif d_ll < 0 and d_rps < 0:
        print(f"  -> Recommend half_life_days={best[0]:.0f}, dc_weight={best[1]:.2f} "
              "(improves both metrics out of sample).")
    else:
        print("  -> Tuned config wins in-sample but does NOT clearly beat the "
              "default out of sample; keep the current defaults.")


if __name__ == "__main__":
    main()
