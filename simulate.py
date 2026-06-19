"""Simulate the rest of the 2026 World Cup and print each team's odds.

    python simulate.py [--sims 20000] [--group A] [--full]

Uses results already played plus the model to Monte-Carlo the remaining group
games and the knockout rounds (official 2026 bracket — see tournament.py).
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


def _load_injuries(args) -> dict | None:
    """Injuries dict from a JSON file or a live API-Football fetch, or None."""
    if args.injuries_file:
        import json
        with open(args.injuries_file, encoding="utf-8") as f:
            return json.load(f)
    if args.injuries:
        from worldcup_model.squad import fetch_injuries
        try:
            return fetch_injuries()
        except RuntimeError as e:
            raise SystemExit(str(e))
    return None


def _report_injuries(sim, injuries: dict | None) -> None:
    """Show which teams had absences applied (and any teams we couldn't match)."""
    if not injuries:
        return
    if sim.injured_teams:
        print("\nInjuries/suspensions applied (expected-goals impact):")
        for team, (players, own, opp) in sorted(sim.injured_teams.items()):
            shown = ", ".join(players[:4]) + ("..." if len(players) > 4 else "")
            print(f"  {team:<16} {len(players):>2} out  "
                  f"-{own:.2f} own / +{opp:.2f} opp xG   {shown}")
    unmatched = sorted(t for t, p in injuries.items() if p and t not in sim.idx)
    if unmatched:
        print(f"  ({len(unmatched)} non-tournament/unmatched teams ignored: "
              f"{', '.join(unmatched[:6])}{'...' if len(unmatched) > 6 else ''})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--group", default=None, help="show one group's table (A-L)")
    ap.add_argument("--full", action="store_true", help="show all 48 teams")
    ap.add_argument("--injuries", action="store_true",
                    help="auto-fetch injuries/suspensions (needs API_FOOTBALL_KEY)")
    ap.add_argument("--injuries-file", default=None,
                    help="JSON {team: [players out]} to apply (offline alternative)")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    model = ExpertModel.load(args.model)
    md = datamod.load()
    allwc = pd.concat([md.played, md.upcoming])
    wc = allwc[(allwc["tournament"] == "FIFA World Cup")
               & (allwc["date"] >= pd.Timestamp("2026-01-01"))]
    played = wc[wc["home_score"].notna()]
    upcoming = wc[wc["home_score"].isna()]

    injuries = _load_injuries(args)
    sim = TournamentSimulator(model, played, upcoming, injuries=injuries)
    print(f"Simulating from current state: {len(played)} group games played, "
          f"{len(upcoming)} remaining.  {args.sims:,} simulations.")
    _report_injuries(sim, injuries)
    print()
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
