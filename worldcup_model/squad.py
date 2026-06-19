"""Squad availability: injuries / suspensions and their effect on the model.

Two honest limitations up front:
  1. Our results dataset has no player data, so a player's worth can't be
     learned here — impacts below are tiered, literature-informed *estimates*
     (a talisman is worth ~a third of a goal a game; a squad player ~nothing),
     and they are meant to be edited.
  2. Auto-fetching who is actually out needs a data source. `fetch_injuries`
     uses API-Football (free tier), which is a SEPARATE key from the odds feed:
     `set API_FOOTBALL_KEY=...`. Without it, use the manual interface.

A missing attacker lowers their own team's expected goals; a missing
defender/keeper raises the opponent's. `absence_penalty` returns that pair,
which `ExpertModel.expected_goals(..., home_out=, away_out=)` applies.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# Goal value of an absence, by importance tier.
TIER_IMPACT = {"talisman": 0.32, "star": 0.18, "key": 0.10, "squad": 0.04}
CAP = 0.55  # a team can't lose more than ~half a goal of quality (replacements)

# Notable players -> (team, tier, side). Illustrative and editable; verify the
# current roster before relying on any entry. side: "att" lowers own goals,
# "def" raises opponent goals.
ROSTER: dict[str, tuple[str, str, str]] = {
    "Lionel Messi": ("Argentina", "talisman", "att"),
    "Lautaro Martinez": ("Argentina", "star", "att"),
    "Kylian Mbappe": ("France", "talisman", "att"),
    "Erling Haaland": ("Norway", "talisman", "att"),
    "Harry Kane": ("England", "talisman", "att"),
    "Jude Bellingham": ("England", "star", "att"),
    "Lamine Yamal": ("Spain", "star", "att"),
    "Vinicius Junior": ("Brazil", "star", "att"),
    "Raphinha": ("Brazil", "star", "att"),
    "Jamal Musiala": ("Germany", "star", "att"),
    "Florian Wirtz": ("Germany", "star", "att"),
    "Christian Pulisic": ("United States", "star", "att"),
    "Achraf Hakimi": ("Morocco", "star", "att"),
    "Virgil van Dijk": ("Netherlands", "star", "def"),
    "Bruno Fernandes": ("Portugal", "star", "att"),
}


def player_impact(name: str, tier: str | None = None, side: str = "att",
                  default_tier: str = "key") -> tuple[float, str]:
    """Goal impact + side for a player.

    A name in ROSTER uses its known tier; otherwise we fall back to `tier` (or
    `default_tier` when none is given). Live auto-fetch passes default_tier=
    'squad' so an unrecognised injured player counts as ~nothing while a known
    talisman/star still gets full weight."""
    if name in ROSTER:
        _, t, s = ROSTER[name]
        return TIER_IMPACT[t], s
    return TIER_IMPACT.get(tier or default_tier, TIER_IMPACT["key"]), side


def absence_penalty(absent: list, default_tier: str = "key") -> tuple[float, float]:
    """Net (own-goals-reduction, opponent-goals-increase) for a team's absences.

    `absent` items are either a player name (looked up / defaulted to
    `default_tier`) or a (name, tier, side) tuple for manual control."""
    own = opp = 0.0
    for a in absent:
        if isinstance(a, (tuple, list)):
            name, tier, side = (list(a) + ["key", "att"])[:3]
            imp, s = player_impact(name, tier, side, default_tier)
        else:
            imp, s = player_impact(a, default_tier=default_tier)
        if s == "def":
            opp += imp
        else:
            own += imp
    return min(own, CAP), min(opp, CAP)


# API-Football team names -> our dataset spellings (so fetched injuries match
# the team names used by the model and tournament simulator).
TEAM_ALIASES = {
    "USA": "United States", "Korea Republic": "South Korea",
    "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde", "Curacao": "Curaçao",
}


def normalize_team(name: str) -> str:
    """Map an API-Football team name to the dataset spelling."""
    return TEAM_ALIASES.get(name, name)


# ---- optional auto-fetch (API-Football v3) --------------------------------
def fetch_injuries(league: int = 1, season: int = 2026,
                   api_key: str | None = None) -> dict[str, list[str]]:
    """Current injuries/suspensions per team via API-Football. Needs a free key
    in API_FOOTBALL_KEY. Returns {team_name: [player names out]} with team names
    normalised to the dataset spelling and players de-duplicated."""
    key = api_key or os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("No API_FOOTBALL_KEY set. Get a free key at "
                           "https://www.api-football.com/ for injury auto-fetch.")
    url = f"https://v3.football.api-sports.io/injuries?league={league}&season={season}"
    req = urllib.request.Request(url, headers={"x-apisports-key": key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API-Football HTTP {e.code}: {e.read()[:200]!r}")
    out: dict[str, list[str]] = {}
    for item in data.get("response", []):
        team = item.get("team", {}).get("name")
        player = item.get("player", {}).get("name")
        if team and player:
            out.setdefault(normalize_team(team), [])
            if player not in out[normalize_team(team)]:
                out[normalize_team(team)].append(player)
    return out
