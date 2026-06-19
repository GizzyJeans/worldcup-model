"""Live multi-bookmaker odds via The Odds API (the-odds-api.com).

Free tier: sign up for a key and `set ODDS_API_KEY=...` (or pass --api-key).
Request cost = (#markets) x (#regions); the remaining quota is printed after
each call. We fetch 1X2 (`h2h`) and totals, normalise every bookmaker to a
common shape, and expose two things value betting actually needs: the best
available price per outcome (line shopping) and a de-vigged market consensus.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

API_BASE = "https://api.the-odds-api.com/v4"
SHARP_BOOKS = {"pinnacle", "betfair", "smarkets"}  # used as the "fair" anchor


class OddsAPIError(Exception):
    pass


@dataclass
class Event:
    id: str
    home: str
    away: str
    commence_time: str
    # bookmaker title -> {"1x2": {home,draw,away}, "totals": {line: {over,under}}}
    books: dict[str, dict] = field(default_factory=dict)


def get_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("ODDS_API_KEY")
    if not key:
        raise OddsAPIError(
            "No API key. Get a free one at https://the-odds-api.com/ , then "
            "`set ODDS_API_KEY=yourkey` (Windows) / `export ODDS_API_KEY=...`.")
    return key


def _get(url: str) -> tuple[object, dict]:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
            quota = {h: r.headers.get(h) for h in
                     ("x-requests-remaining", "x-requests-used", "x-requests-last")}
            return data, quota
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        hint = {401: " (bad API key)", 422: " (bad sport/region/market)",
                429: " (quota exhausted)"}.get(e.code, "")
        raise OddsAPIError(f"HTTP {e.code}{hint}: {body}")
    except urllib.error.URLError as e:
        raise OddsAPIError(f"network error: {e.reason}")


def list_soccer_sports(api_key: str | None = None) -> list[dict]:
    """Active soccer competitions and their exact sport keys."""
    key = get_api_key(api_key)
    data, _ = _get(f"{API_BASE}/sports/?apiKey={key}")
    return [s for s in data if "soccer" in s.get("key", "").lower()
            or s.get("group", "").lower() == "soccer"]


def _parse_event(e: dict) -> Event:
    home, away = e["home_team"], e["away_team"]
    books: dict[str, dict] = {}
    for b in e.get("bookmakers", []):
        mk: dict = {}
        for m in b.get("markets", []):
            if m["key"] == "h2h":
                d = {}
                for o in m["outcomes"]:
                    if o["name"] == home:
                        d["home"] = o["price"]
                    elif o["name"] == away:
                        d["away"] = o["price"]
                    elif o["name"].lower() == "draw":
                        d["draw"] = o["price"]
                if len(d) == 3:
                    mk["1x2"] = d
            elif m["key"] == "totals":
                tot: dict = {}
                for o in m["outcomes"]:
                    line = o.get("point")
                    if line is not None:
                        tot.setdefault(float(line), {})[o["name"].lower()] = o["price"]
                if tot:
                    mk["totals"] = tot
        if mk:
            books[b.get("title", b.get("key", "?"))] = mk
    return Event(e["id"], home, away, e.get("commence_time", ""), books)


def fetch_odds(sport: str = "soccer_fifa_world_cup", regions: str = "eu,uk",
               markets: str = "h2h,totals", api_key: str | None = None,
               odds_format: str = "decimal") -> tuple[list[Event], dict]:
    key = get_api_key(api_key)
    url = (f"{API_BASE}/sports/{sport}/odds/?apiKey={key}&regions={regions}"
           f"&markets={markets}&oddsFormat={odds_format}")
    data, quota = _get(url)
    return [_parse_event(e) for e in data], quota


# ---- aggregation helpers --------------------------------------------------
def devig_1x2(odds3: dict[str, float]) -> dict[str, float]:
    raw = {k: 1.0 / v for k, v in odds3.items()}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def consensus_1x2(ev: Event, sharp_only: bool = False) -> dict[str, float] | None:
    """De-vigged average across books (optionally sharp books only)."""
    probs = []
    for title, b in ev.books.items():
        if "1x2" not in b:
            continue
        if sharp_only and not any(s in title.lower() for s in SHARP_BOOKS):
            continue
        probs.append(devig_1x2(b["1x2"]))
    if not probs:
        return None
    return {k: sum(p[k] for p in probs) / len(probs) for k in ("home", "draw", "away")}


def best_1x2(ev: Event) -> dict[str, tuple[float, str]]:
    """Best (highest) decimal price per outcome and which book offers it."""
    out: dict[str, tuple[float, str]] = {}
    for title, b in ev.books.items():
        for sel, price in b.get("1x2", {}).items():
            if sel not in out or price > out[sel][0]:
                out[sel] = (price, title)
    return out
