"""Fit the expert model on historical international results and save it.

    python train.py [--since 2014-01-01] [--half-life 540] [--dc-weight 0.7]

The recency window (`--since`) keeps the fit on the modern game; the half-life
controls how fast older matches fade; `--dc-weight` blends Dixon-Coles vs Elo.
"""

from __future__ import annotations

import argparse
import sys

from worldcup_model import data as datamod
from worldcup_model.host import estimate_host_advantage
from worldcup_model.model import ExpertModel
from worldcup_model.paths import MODEL_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2014-01-01", help="earliest match date to fit on")
    ap.add_argument("--half-life", type=float, default=540.0, help="recency half-life in days")
    ap.add_argument("--dc-weight", type=float, default=0.7, help="Dixon-Coles vs Elo blend [0-1]")
    ap.add_argument("--refresh", action="store_true", help="re-download the dataset first")
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    if args.refresh:
        datamod.download(force=True)

    md = datamod.load(since=args.since)
    print(f"Loaded {len(md.played):,} played matches "
          f"({md.played['date'].min().date()} -> {md.played['date'].max().date()}), "
          f"{len(md.teams)} teams, {len(md.upcoming):,} upcoming fixtures.")

    # Host advantage is estimated from ALL World Cup history (recent windows
    # have too few hosts, and Qatar 2022 was an atypically weak one).
    host_adv, se, n = estimate_host_advantage(datamod.load().played)
    print(f"WC host advantage: +{host_adv:.3f} goals supremacy "
          f"(se {se:.3f}, n={n} host games)")

    model = ExpertModel(dc_weight=args.dc_weight, half_life_days=args.half_life,
                        host_adv=host_adv)
    model.fit(md)
    model.save(args.out)
    print(f"Saved model -> {args.out}  (rho={model.dc.rho:+.4f}, "
          f"home_adv={model.dc.home_adv:+.3f}, host_adv=+{model.host_adv:.3f}, "
          f"asof={model.asof.date()})")

    print("\nTop 15 by Elo:")
    for i, (team, r) in enumerate(model.elo.top(15), 1):
        print(f"  {i:2d}. {team:<22} {r:7.1f}")

    print("\nTop 15 by Dixon-Coles strength (attack - defence):")
    tbl = model.dc.strength_table().head(15)
    for i, row in enumerate(tbl.itertuples(index=False), 1):
        print(f"  {i:2d}. {row.team:<22} rating={row.rating:+.3f} "
              f"(att={row.attack:+.3f}, def={row.defence:+.3f})")


if __name__ == "__main__":
    main()
