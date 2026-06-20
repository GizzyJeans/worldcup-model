"""CLV-first bet picking: scan a board for line-shopping value, then measure
closing-line value (CLV) — the metric that actually tracks bet-picking skill.

    python picks.py scan   [--sport soccer_fifa_world_cup | --odds-file board.json]
    python picks.py settle [--sport ...                   | --odds-file close.json]
    python picks.py report [--log picks_log.csv]

scan    flags every outcome whose best available price beats the de-vigged sharp
        consensus by --min-edge, sizes it with fractional Kelly, and APPENDS each
        pick to the log (so its CLV can be measured later). Add --spreads to also
        line-shop the Asian-handicap board (each handicap pick records its line).
settle  re-reads the (now closing) odds, fills each open pick's closing consensus
        and CLV, and marks it settled. Handicap picks settle against the closing
        consensus for their own (side, line); --spreads is auto-enabled if needed.
report  aggregates the log: average CLV, % of picks that beat the close, and the
        staked totals. Positive average CLV = genuine bet-picking skill.

Live odds use The Odds API (set ODDS_API_KEY). For testing / offline use, pass a
JSON board: a list of events, each {"home","away","commence_time"?,"books":
{"<book>": {"1x2": {"home":..,"draw":..,"away":..},
            "spreads": {"<home_line>": {"home":..,"away":..}}}}}. This is the same
shape the live feed normalises to, so both paths run identical code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from worldcup_model.clv import (OUTCOMES, clv, make_ah_picks, make_picks,
                                sharp_consensus, sharp_consensus_ah)
from worldcup_model.odds_feed import Event, fetch_odds
from worldcup_model.paths import ROOT

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

DEFAULT_LOG = os.path.join(ROOT, "picks_log.csv")
LOG_COLS = ["ts", "event", "commence_time", "market", "selection", "line",
            "book", "price", "consensus_prob", "edge", "ev", "stake", "status",
            "close_prob", "clv", "beat_close"]


# ---- odds input (live or offline file) ------------------------------------
def _coerce_line_keys(books: dict) -> dict:
    """JSON object keys are strings; the live feed keys totals/spreads lines by
    float. Coerce offline boards to the same shape so both paths run identically."""
    for b in books.values():
        for mkt in ("totals", "spreads"):
            if mkt in b:
                b[mkt] = {float(k): v for k, v in b[mkt].items()}
    return books


def load_events(args) -> list[Event]:
    if args.odds_file:
        with open(args.odds_file, encoding="utf-8") as f:
            raw = json.load(f)
        evs = [Event(id=e.get("id", f"{e['home']}|{e['away']}"),
                     home=e["home"], away=e["away"],
                     commence_time=e.get("commence_time", ""),
                     books=_coerce_line_keys(e.get("books", {}))) for e in raw]
        print(f"Loaded {len(evs)} events from {args.odds_file}")
        return evs
    markets = "h2h,spreads" if getattr(args, "spreads", False) else "h2h"
    events, quota = fetch_odds(sport=args.sport, regions=args.regions,
                              markets=markets, api_key=args.api_key)
    print(f"Fetched {len(events)} events from The Odds API "
          f"(quota left: {quota.get('x-requests-remaining')})")
    return events


def _read_log(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        # reindex so logs written before a column was added (e.g. "line") load.
        return pd.read_csv(path).reindex(columns=LOG_COLS)
    return pd.DataFrame(columns=LOG_COLS)


def _write_log(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


# ---- commands -------------------------------------------------------------
def cmd_scan(args) -> None:
    events = load_events(args)
    rows = []
    for ev in events:
        for p in make_picks(ev, bankroll=args.bankroll, kelly=args.kelly,
                            min_edge=args.min_edge, max_stake_frac=args.max_stake,
                            top_only=not args.all_outcomes):
            rows.append(p)
        if args.spreads:
            for p in make_ah_picks(ev, bankroll=args.bankroll, kelly=args.kelly,
                                   min_edge=args.min_edge,
                                   max_stake_frac=args.max_stake,
                                   top_only=not args.all_outcomes):
                rows.append(p)
    if not rows:
        print(f"No line-shopping value found (min edge {args.min_edge:.0%}).")
        return

    rows.sort(key=lambda p: p.edge, reverse=True)
    print(f"\n{len(rows)} value pick(s) — best price vs sharp consensus:\n")
    print(f"  {'event':<26}{'mkt':>4}{'sel':>5}{'line':>6}{'price':>7}{'book':>13}"
          f"{'cons':>7}{'edge':>7}{'ev':>7}{'stake':>7}")
    print("  " + "-" * 88)
    for p in rows:
        line = f"{p.line:+g}" if p.market == "ah" else ""
        print(f"  {p.event:<26}{p.market:>4}{p.selection:>5}{line:>6}"
              f"{p.price:>7.2f}{p.book:>13}{p.consensus_prob:>7.1%}"
              f"{p.edge:>+7.1%}{p.ev:>+7.1%}{p.stake:>7.2f}")
    print(f"\n  total staked: {sum(p.stake for p in rows):.2f} "
          f"of {args.bankroll:.0f} bankroll")

    if args.no_log:
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log = _read_log(args.log)
    new = pd.DataFrame([{
        "ts": ts, "event": p.event, "commence_time": p.commence_time,
        "market": p.market, "selection": p.selection,
        "line": (p.line if p.market == "ah" else ""), "book": p.book,
        "price": p.price, "consensus_prob": p.consensus_prob, "edge": p.edge,
        "ev": p.ev, "stake": p.stake, "status": "open", "close_prob": "",
        "clv": "", "beat_close": "",
    } for p in rows])
    _write_log(pd.concat([log, new], ignore_index=True), args.log)
    print(f"  logged {len(new)} pick(s) -> {args.log}")


def cmd_settle(args) -> None:
    log = _read_log(args.log)
    open_mask = log["status"] == "open"
    if not open_mask.any():
        print("No open picks to settle.")
        return
    # Pull the handicap board too if any open pick needs it.
    if (log.loc[open_mask, "market"] == "ah").any():
        args.spreads = True
    events = load_events(args)
    # Closing de-vigged consensus keyed by (event, market, selection, line).
    close: dict[tuple[str, str, str, float | None], float] = {}
    for ev in events:
        name = f"{ev.home} v {ev.away}"
        cons = sharp_consensus(ev)
        if cons is not None:
            for sel in OUTCOMES:
                close[(name, "1x2", sel, None)] = cons[sel]
        ah = sharp_consensus_ah(ev)
        if ah:
            for (side, line), p in ah.items():
                close[(name, "ah", side, line)] = p

    settled = 0
    for i in log.index[open_mask]:
        market = log.at[i, "market"]
        if market == "ah":
            try:
                key = (log.at[i, "event"], "ah", log.at[i, "selection"],
                       float(log.at[i, "line"]))
            except (ValueError, TypeError):
                continue
        else:
            key = (log.at[i, "event"], "1x2", log.at[i, "selection"], None)
        cp = close.get(key)
        if cp is None:
            continue
        v = clv(float(log.at[i, "price"]), cp)
        log.at[i, "close_prob"] = round(cp, 4)
        log.at[i, "clv"] = round(v, 4)
        log.at[i, "beat_close"] = int(v > 0)
        log.at[i, "status"] = "settled"
        settled += 1
    _write_log(log, args.log)
    print(f"Settled {settled} pick(s) against the closing line "
          f"({open_mask.sum() - settled} still open / unmatched).")


def cmd_report(args) -> None:
    log = _read_log(args.log)
    if log.empty:
        print(f"No picks logged yet ({args.log}).")
        return
    n_open = int((log["status"] == "open").sum())
    s = log[log["status"] == "settled"].copy()
    print(f"Picks log: {len(log)} total  |  {len(s)} settled  |  {n_open} open\n")
    if s.empty:
        print("No settled picks yet — run `settle` once the lines close.")
        return
    s["clv"] = pd.to_numeric(s["clv"])
    s["beat_close"] = pd.to_numeric(s["beat_close"])
    s["stake"] = pd.to_numeric(s["stake"])
    avg_clv = s["clv"].mean()
    beat = s["beat_close"].mean()
    print(f"  settled picks      : {len(s)}")
    print(f"  avg CLV            : {avg_clv:+.2%}   (>0 = bet-picking skill)")
    print(f"  beat-the-close rate: {beat:.1%}")
    print(f"  stake-weighted CLV : {(s['clv'] * s['stake']).sum() / s['stake'].sum():+.2%}")
    print(f"  total staked       : {s['stake'].sum():.2f}")
    verdict = ("positive CLV — the picks are beating the close (skill signal)"
               if avg_clv > 0 else
               "non-positive CLV — no demonstrated edge yet (need more picks)")
    print(f"\n  VERDICT: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_odds_args(p):
        p.add_argument("--sport", default="soccer_fifa_world_cup")
        p.add_argument("--regions", default="eu,uk")
        p.add_argument("--api-key", default=None)
        p.add_argument("--odds-file", default=None,
                       help="JSON board for offline use (see module docstring)")
        p.add_argument("--log", default=DEFAULT_LOG)
        p.add_argument("--spreads", action="store_true",
                       help="also line-shop the Asian-handicap board (costs an "
                            "extra market on each live fetch)")

    ps = sub.add_parser("scan", help="find line-shopping value and log picks")
    add_odds_args(ps)
    ps.add_argument("--bankroll", type=float, default=1000.0)
    ps.add_argument("--kelly", type=float, default=0.25)
    ps.add_argument("--min-edge", type=float, default=0.02)
    ps.add_argument("--max-stake", type=float, default=0.05,
                    help="cap on any single stake as a share of bankroll")
    ps.add_argument("--all-outcomes", action="store_true",
                    help="flag every +edge outcome, not just the best per event")
    ps.add_argument("--no-log", action="store_true")
    ps.set_defaults(func=cmd_scan)

    pt = sub.add_parser("settle", help="record closing line + CLV for open picks")
    add_odds_args(pt)
    pt.set_defaults(func=cmd_settle)

    pr = sub.add_parser("report", help="aggregate CLV / skill stats from the log")
    pr.add_argument("--log", default=DEFAULT_LOG)
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
