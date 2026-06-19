"""Loading and preparing international match-results data.

Source: martj42/international_results (`results.csv`), the canonical open
dataset of men's international football results from 1872 to the present. It
already contains the scheduled fixtures for upcoming tournaments (with empty
scores), which we use as prediction targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)


@dataclass
class MatchData:
    """Played matches (with scores) plus the list of upcoming fixtures."""

    played: pd.DataFrame   # date, home_team, away_team, home_score, away_score, tournament, neutral
    upcoming: pd.DataFrame  # same columns, scores are NaN

    @property
    def teams(self) -> list[str]:
        t = pd.concat([self.played["home_team"], self.played["away_team"]])
        return sorted(t.unique().tolist())


def download(force: bool = False) -> str:
    """Ensure `data/results.csv` exists locally; download it if missing."""
    if force or not os.path.exists(RESULTS_CSV):
        import urllib.request

        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(RESULTS_URL, RESULTS_CSV)
    return RESULTS_CSV


def load(
    path: str | None = None,
    since: str | None = None,
    drop_friendlies: bool = False,
) -> MatchData:
    """Load results, split into played vs. upcoming, optionally trim history.

    Parameters
    ----------
    since: ISO date string; keep only matches on/after it (model recency window).
    drop_friendlies: exclude friendlies from the *played* set used for fitting.
    """
    path = path or download()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns=str.strip)

    # Normalise the neutral flag to a real boolean.
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])

    has_score = df["home_score"].notna() & df["away_score"].notna()
    played = df[has_score].copy()
    upcoming = df[~has_score].copy()

    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    if since is not None:
        played = played[played["date"] >= pd.Timestamp(since)]
    if drop_friendlies:
        played = played[played["tournament"].str.lower() != "friendly"]

    played = played.sort_values("date").reset_index(drop=True)
    upcoming = upcoming.sort_values("date").reset_index(drop=True)
    return MatchData(played=played, upcoming=upcoming)


def time_weights(dates: pd.Series, asof: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Exponential recency weights phi = 0.5 ** (age / half_life).

    Dixon-Coles down-weights older matches so the fit tracks current form.
    `half_life_days` is the age at which a match counts half as much.
    """
    age = (asof - dates).dt.days.to_numpy(dtype=float)
    age = np.clip(age, 0.0, None)
    return np.power(0.5, age / float(half_life_days))
