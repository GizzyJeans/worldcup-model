"""Simulate the rest of the 2026 World Cup and print each team's odds.

    python simulate.py [--sims 20000] [--group A] [--full]

Uses results already played plus the model to Monte-Carlo the remaining group
games and the knockout rounds (neutral random draw — see tournament.py).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from worldcup_model import data as datamod
from worldcup_model.model import ExpertModel
from worldcup_model.paths import MODEL_PATH
from worldcup_model.tournament import TournamentSimulator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def _pct(x: float) -> str:
    return "  -  " if x < 0.0005 else f"{100 * x:4.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--group", default=None, help="show one group's table (A-L)")
    ap.add_argument("--full", action="store_true", help="show all 48 teams")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    model = ExpertModel.load(args.model)
    md = datamod.load()
    allwc = pd.concat([md.played, md.upcoming])
    wc = allwc[(allwc["tournament"] == "FIFA World Cup")
               & (allwc["date"] >= pd.Timestamp("2026-01-01"))]
    played = wc[wc["home_score"].notna()]
    upcoming = wc[wc["home_score"].isna()]

    sim = TournamentSimulator(model, played, upcoming)
    print(f"Simulating from current state: {len(played)} group games played, "
          f"{len(upcoming)} remaining.  {args.sims:,} simulations.\n")
    res = sim.run(n_sims=args.sims)

    if args.group:
        sub = res[res["group"] == args.group.upper()].sort_values("qualify", ascending=False)
        print(f"Group {args.group.upper()}:")
        print(f"  {'team':<22}{'win grp':>8}{'top2':>8}{'qualify':>9}")
        for r in sub.itertuples(index=False):
            print(f"  {r.team:<22}{_pct(r.win_group):>8}"
                  f"{_pct(r.win_group + r.runner_up):>8}{_pct(r.qualify):>9}")
        return

    show = res if args.full else res.head(24)
    print(f"{'team':<20}{'grp':>4}{'qualify':>8}{'QF':>7}{'SF':>7}{'final':>7}{'CHAMP':>7}")
    print("-" * 60)
    for r in show.itertuples(index=False):
        print(f"{r.team:<20}{r.group:>4}{_pct(r.qualify):>8}{_pct(r.reach_QF):>7}"
              f"{_pct(r.reach_SF):>7}{_pct(r.reach_final):>7}{_pct(r.win_cup):>7}")

    fav = res.iloc[0]
    print(f"\nFavourite: {fav.team} ({100*fav.win_cup:.1f}% to win the cup).  "
          f"Sum of win%: {res['win_cup'].sum():.2f}")


if __name__ == "__main__":
    main()
