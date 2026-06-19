"""Predict matches and find value bets with the trained expert model.

Examples
--------
  # Market probabilities for a neutral-venue match
  python predict.py predict --home Spain --away England --neutral

  # Find value vs bookmaker odds (decimal), with bankroll + staking
  python predict.py value --home Brazil --away Croatia --neutral \\
      --odds "1x2:home=1.95,draw=3.5,away=4.2;over_under:over=2.0,under=1.85" \\
      --bankroll 1000 --kelly 0.25

  # Predict every upcoming World Cup fixture in the dataset
  python predict.py fixtures --tournament "FIFA World Cup"

  # Show the rating tables
  python predict.py ratings --n 25
"""

from __future__ import annotations

import argparse
import sys

from worldcup_model import data as datamod
from worldcup_model.host import is_host_game
from worldcup_model.model import ExpertModel
from worldcup_model.paths import MODEL_PATH

# Team names carry accents (Curaçao, Türkiye); avoid Windows codepage crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def _load_model(path: str) -> ExpertModel:
    import os
    if not os.path.exists(path):
        raise SystemExit(f"No model at {path!r}. Run `python train.py` first.")
    return ExpertModel.load(path)


def _parse_odds(spec: str) -> dict[str, dict[str, float]]:
    """'1x2:home=1.95,draw=3.5,away=4.2;over_under:over=2.0,under=1.85' -> dict."""
    out: dict[str, dict[str, float]] = {}
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        market, sels = block.split(":", 1)
        d: dict[str, float] = {}
        for pair in sels.split(","):
            sel, val = pair.split("=")
            d[sel.strip().lower()] = float(val)
        out[market.strip().lower()] = d
    return out


def _fmt_pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def _print_prediction(pred: dict) -> None:
    fx = pred["fixture"]
    venue = "neutral" if fx["neutral"] else f"{fx['home']} home"
    eg = pred["expected_goals"]
    print(f"\n{fx['home']} vs {fx['away']}  ({venue})")
    print(f"  Elo: {pred['elo']['home']} vs {pred['elo']['away']}")
    print(f"  Expected goals: {eg['home']} - {eg['away']}")
    x = pred["1x2"]
    print(f"  1X2:   home {_fmt_pct(x['home'])} | draw {_fmt_pct(x['draw'])} "
          f"| away {_fmt_pct(x['away'])}")
    for line, d in pred["over_under"].items():
        print(f"  O/U {line}: over {_fmt_pct(d['over'])} | under {_fmt_pct(d['under'])}")
    b = pred["btts"]
    print(f"  BTTS:  yes {_fmt_pct(b['yes'])} | no {_fmt_pct(b['no'])}")
    cs = ", ".join(f"{k} {_fmt_pct(v)}" for k, v in pred["correct_score"].items())
    print(f"  Top scores: {cs}")


def _outs(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def cmd_predict(args) -> None:
    model = _load_model(args.model)
    host = is_host_game(args.home, args.neutral)
    ho, ao = _outs(args.home_out), _outs(args.away_out)
    pred = model.predict(args.home, args.away, neutral=args.neutral, host=host,
                         home_out=ho, away_out=ao)
    if ho or ao:
        print(f"(out — {args.home}: {ho or 'none'} | {args.away}: {ao or 'none'})")
    _print_prediction(pred)


def cmd_value(args) -> None:
    model = _load_model(args.model)
    odds = _parse_odds(args.odds)
    host = is_host_game(args.home, args.neutral)
    ho, ao = _outs(args.home_out), _outs(args.away_out)
    if ho or ao:
        print(f"(out — {args.home}: {ho or 'none'} | {args.away}: {ao or 'none'})")
    _print_prediction(model.predict(args.home, args.away, neutral=args.neutral, host=host,
                                    home_out=ho, away_out=ao))
    bets = model.find_value(
        args.home, args.away, odds, neutral=args.neutral,
        bankroll=args.bankroll, kelly=args.kelly, min_edge=args.min_edge, host=host,
        home_out=ho, away_out=ao,
    )
    print(f"\nValue bets (bankroll {args.bankroll:.0f}, {args.kelly:g}x Kelly, "
          f"min edge {args.min_edge:.0%}):")
    if not bets:
        print("  none — no selection beats its price by the edge threshold.")
        return
    print(f"  {'market':<13}{'pick':<7}{'odds':>6}{'model':>8}{'fair':>8}"
          f"{'edge':>8}{'EV':>8}{'stake':>9}")
    for b in bets:
        print(f"  {b.market:<13}{b.selection:<7}{b.odds:>6.2f}"
              f"{_fmt_pct(b.model_prob):>8}{_fmt_pct(b.fair_prob):>8}"
              f"{b.edge:>+8.1%}{b.ev:>+8.1%}{b.stake:>9.2f}")


def cmd_fixtures(args) -> None:
    model = _load_model(args.model)
    md = datamod.load(since="2014-01-01")
    up = md.upcoming
    if args.tournament:
        up = up[up["tournament"].str.contains(args.tournament, case=False, na=False)]
    up = up.head(args.limit)
    if up.empty:
        print("No matching upcoming fixtures in the dataset.")
        return
    for row in up.itertuples(index=False):
        try:
            host = is_host_game(row.home_team, bool(row.neutral))
            pred = model.predict(row.home_team, row.away_team, neutral=bool(row.neutral), host=host)
        except KeyError as e:
            print(f"\n{row.home_team} vs {row.away_team}: skipped ({e})")
            continue
        x = pred["1x2"]
        print(f"{str(row.date.date())}  {row.home_team:>16} vs {row.away_team:<16}  "
              f"H {_fmt_pct(x['home'])}  D {_fmt_pct(x['draw'])}  A {_fmt_pct(x['away'])}")


def cmd_ratings(args) -> None:
    model = _load_model(args.model)
    print(f"Top {args.n} by Elo:")
    for i, (team, r) in enumerate(model.elo.top(args.n), 1):
        print(f"  {i:2d}. {team:<22} {r:7.1f}")


def cmd_injuries(args) -> None:
    from worldcup_model.squad import fetch_injuries
    try:
        inj = fetch_injuries(api_key=args.api_key)
    except RuntimeError as e:
        raise SystemExit(str(e))
    if not inj:
        print("No injuries/suspensions reported.")
        return
    for team in sorted(inj):
        print(f"  {team:<22} {', '.join(inj[team])}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=MODEL_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_match(p):
        p.add_argument("--home", required=True)
        p.add_argument("--away", required=True)
        p.add_argument("--neutral", action="store_true", help="neutral venue (most WC games)")
        p.add_argument("--home-out", default="", help="home injuries/suspensions, comma-separated")
        p.add_argument("--away-out", default="", help="away injuries/suspensions, comma-separated")

    p = sub.add_parser("predict", help="market probabilities for one match")
    add_match(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("value", help="find value bets vs odds")
    add_match(p)
    p.add_argument("--odds", required=True, help="see module docstring for format")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.25, help="fractional Kelly multiplier")
    p.add_argument("--min-edge", type=float, default=0.02)
    p.set_defaults(func=cmd_value)

    p = sub.add_parser("fixtures", help="predict upcoming fixtures from the dataset")
    p.add_argument("--tournament", default="FIFA World Cup")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("ratings", help="show Elo rating table")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=cmd_ratings)

    p = sub.add_parser("injuries", help="current injuries/suspensions (API-Football)")
    p.add_argument("--api-key", default=None, help="or set API_FOOTBALL_KEY")
    p.set_defaults(func=cmd_injuries)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
