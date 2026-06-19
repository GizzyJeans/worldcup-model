"""Walk-forward backtest of the expert model's forecast skill.

For each match in the test window we predict using a model fit ONLY on matches
that happened strictly earlier (the model is refit every `--refit-days`), so
there is no look-ahead. We then score the 1X2 (and Over/Under 2.5) forecasts
with proper metrics and compare against baselines.

What this answers: are the model's probabilities well-calibrated and better
than naive baselines? (A true betting-ROI / closing-line-value backtest needs
historical odds, which this results-only dataset lacks — see --odds-file.)

    python backtest.py [--since 2014-01-01] [--test-start 2018-01-01]
                       [--refit-days 180] [--out predictions.csv]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from worldcup_model import data as datamod
from worldcup_model.markets import markets
from worldcup_model.model import ExpertModel

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

LABELS = ("H", "D", "A")  # ordered for RPS


def result_1x2(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (2 if hs < as_ else 1)


def probs_from_goals(lam: float, mu: float, rho: float) -> np.ndarray:
    m = markets(lam, mu, rho)["1x2"]
    return np.array([m["home"], m["draw"], m["away"]])


# ---- metrics --------------------------------------------------------------
def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
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


def calibration_table(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> str:
    """Reliability of all 3 class probabilities pooled into deciles."""
    onehot = np.eye(3)[y]
    pp, hit = probs.ravel(), onehot.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pp, edges) - 1, 0, bins - 1)
    lines = ["  pred-bin    n   predicted  observed"]
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        lines.append(f"  {edges[b]:.1f}-{edges[b+1]:.1f}  {m.sum():5d}    "
                     f"{pp[m].mean():6.1%}    {hit[m].mean():6.1%}")
    return "\n".join(lines)


def brier_binary(p_over: np.ndarray, y_over: np.ndarray) -> dict:
    p = np.clip(p_over, 1e-15, 1 - 1e-15)
    ll = float(-(y_over * np.log(p) + (1 - y_over) * np.log(1 - p)).mean())
    return {"logloss": ll, "brier": float(((p - y_over) ** 2).mean()), "n": len(y_over)}


# ---- walk-forward ---------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2014-01-01")
    ap.add_argument("--test-start", default="2018-01-01")
    ap.add_argument("--refit-days", type=int, default=180)
    ap.add_argument("--half-life", type=float, default=540.0)
    ap.add_argument("--dc-weight", type=float, default=0.7)
    ap.add_argument("--out", default=None, help="optional CSV of per-match predictions")
    args = ap.parse_args()

    md = datamod.load(since=args.since)
    played = md.played
    test_start = pd.Timestamp(args.test_start)
    empty = played.iloc[0:0]

    # Fixed baseline = outcome base rates from the pre-test training data.
    pre = played[played["date"] < test_start]
    base = np.array([
        (pre["home_score"] > pre["away_score"]).mean(),
        (pre["home_score"] == pre["away_score"]).mean(),
        (pre["home_score"] < pre["away_score"]).mean(),
    ])
    print(f"Train {pre['date'].min().date()}..{(test_start - pd.Timedelta(days=1)).date()} "
          f"({len(pre):,} matches).  Base rate H/D/A = "
          f"{base[0]:.1%}/{base[1]:.1%}/{base[2]:.1%}")

    rows = []
    refit = test_start
    max_date = played["date"].max()
    n_refits = 0
    while refit <= max_date:
        train = played[played["date"] < refit]
        model = ExpertModel(dc_weight=args.dc_weight, half_life_days=args.half_life)
        model.fit(datamod.MatchData(played=train, upcoming=empty), asof=refit)
        n_refits += 1

        chunk = played[(played["date"] >= refit) &
                       (played["date"] < refit + pd.Timedelta(days=args.refit_days))]
        for r in chunk.itertuples(index=False):
            h, a = r.home_team, r.away_team
            if h not in model.dc.attack or a not in model.dc.attack:
                continue
            neu = bool(r.neutral)
            host = (str(r.tournament) == "FIFA World Cup") and not neu
            ld, mdc = model.dc.expected_goals(h, a, neu)
            le, me = model.elo.expected_goals(h, a, neu)
            lb, mb = model.expected_goals(h, a, neu, host)
            y = result_1x2(int(r.home_score), int(r.away_score))
            over = int(int(r.home_score) + int(r.away_score) > 2.5)
            mk = markets(lb, mb, model.dc.rho)
            rows.append({
                "date": r.date, "home": h, "away": a, "y": y, "over25": over,
                "blend": probs_from_goals(lb, mb, model.dc.rho),
                "dc": probs_from_goals(ld, mdc, model.dc.rho),
                "elo": probs_from_goals(le, me, 0.0),
                "p_over": mk["over_under"]["2.5"]["over"],
            })
        refit += pd.Timedelta(days=args.refit_days)

    if not rows:
        print("No test matches evaluated.")
        return

    y = np.array([r["y"] for r in rows])
    over = np.array([r["over25"] for r in rows])
    models = {
        "Model (blend)": np.vstack([r["blend"] for r in rows]),
        "Dixon-Coles":   np.vstack([r["dc"] for r in rows]),
        "Elo only":      np.vstack([r["elo"] for r in rows]),
        "Base rate":     np.tile(base, (len(y), 1)),
    }

    print(f"\nTested {len(y):,} matches "
          f"({rows[0]['date'].date()}..{rows[-1]['date'].date()}), {n_refits} refits.\n")
    print(f"  {'forecaster':<16}{'log-loss':>10}{'Brier':>9}{'RPS':>9}{'accuracy':>10}")
    print("  " + "-" * 52)
    for name, P in models.items():
        m = evaluate(P, y)
        print(f"  {name:<16}{m['logloss']:>10.4f}{m['brier']:>9.4f}"
              f"{m['rps']:>9.4f}{m['acc']:>10.1%}")
    print("  (lower log-loss / Brier / RPS is better; higher accuracy is better)")

    ou = brier_binary(np.array([r["p_over"] for r in rows]), over)
    print(f"\n  Over/Under 2.5  ->  log-loss {ou['logloss']:.4f}, Brier {ou['brier']:.4f}, "
          f"base-rate over = {over.mean():.1%}")

    print("\nCalibration (Model blend, all 1X2 probabilities pooled):")
    print(calibration_table(models["Model (blend)"], y))

    if args.out:
        df = pd.DataFrame([{
            "date": r["date"], "home": r["home"], "away": r["away"],
            "result": LABELS[r["y"]],
            "p_home": r["blend"][0], "p_draw": r["blend"][1], "p_away": r["blend"][2],
        } for r in rows])
        df.to_csv(args.out, index=False)
        print(f"\nWrote {len(df):,} predictions -> {args.out}")


if __name__ == "__main__":
    main()
