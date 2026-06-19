"""Download the bookmaker-odds dataset used by roi_backtest.py / market_blend.py.

These files are large (~56 MB) and git-ignored, so fetch them after cloning on
a new machine:  python fetch_odds_data.py

Source: eatpizzanot/soccer-dataset (Pinnacle closing odds + fixtures + teams).
"""

from __future__ import annotations

import os
import urllib.request

from worldcup_model.paths import ODDS_DIR

BASE = "https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/csv"
FILES = ["fixtures.csv", "odds.csv", "teams.csv", "leagues.csv"]


def main() -> None:
    os.makedirs(ODDS_DIR, exist_ok=True)
    for f in FILES:
        dest = os.path.join(ODDS_DIR, f)
        print(f"downloading {f} ...", end=" ", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{f}", dest)
        print(f"{os.path.getsize(dest) / 1e6:.1f} MB")
    print(f"done -> {ODDS_DIR}")


if __name__ == "__main__":
    main()
