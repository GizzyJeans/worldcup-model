"""Live odds across bookmakers, with line-shopping and market-anchored value.

    python fetch_odds.py --list-sports                 # find the exact sport key
    python fetch_odds.py --sport soccer_fifa_world_cup  # best prices + consensus
    python fetch_odds.py --sport soccer_fifa_world_cup --value   # + value scan

Needs a free key from https://the-odds-api.com (set ODDS_API_KEY). Without a
key it prints setup instructions and exits. The value scan covers 1X2, the
Asian-handicap board and over/under, and for EVERY market anchors the model to
the sharp consensus (blended at w=BLEND_W) before line-shopping the best price —
so the EV shown is already market-anchored, not a raw model number (the
backtests show the raw model is overconfident; the realistic edge is the soft
price beating the sharp consensus, not the model beating the close).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from worldcup_model import odds_feed as feed
from worldcup_model.host import HOSTS_2026, is_host_game
from worldcup_model.paths import MODEL_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# the-odds-api team names -> model (martj42) names, where they differ.
NAME_FIX = {"USA": "United States", "Korea Republic": "South Korea",
            "IR Iran": "Iran", "Czechia": "Czech Republic"}

MIN_EV = 0.02
MAX_EV = 0.20   # above this on a 40-book market = stale/bad single-book price, not value
KELLY, MAX_STAKE_FRAC = 0.25, 0.05
BLEND_W = 0.9   # market weight when anchoring the model (market_blend.py: w*~0.92)


def _odds_fair(consensus: dict) -> dict:
    return {k: 1.0 / v for k, v in consensus.items()}  # probs -> pseudo decimal


def _is_upcoming(ev) -> bool:
    """True if the match hasn't kicked off — live odds vs a pre-match model is noise."""
    try:
        t = datetime.fromisoformat(ev.commence_time.replace("Z", "+00:00"))
        return t > datetime.now(timezone.utc)
    except (ValueError, AttributeError):
        return True


def cmd_value(events, model, bankroll):
    from worldcup_model.markets import asian_handicap, markets
    from worldcup_model.value import blend_market

    rows = []
    for ev in events:
        # Fair line = SHARP books only (Betfair/Pinnacle/Smarkets); skip matches
        # without one. Comparing soft best-price to a soft-inclusive average is
        # the classic way to manufacture fake longshot "value".
        sharp = feed.consensus_1x2(ev, sharp_only=True)
        best = feed.best_1x2(ev)
        if not sharp or not best:
            continue
        home = NAME_FIX.get(ev.home, ev.home)
        away = NAME_FIX.get(ev.away, ev.away)
        if home not in model.dc.attack or away not in model.dc.attack:
            continue
        host = home in HOSTS_2026
        lam, mu = model.expected_goals(home, away, not host, host)
        rho = model.dc.rho

        def consider(label, price, book, fair_p):
            """Blended fair prob vs a line-shopped price -> logged if it clears EV."""
            if price > 13.0:        # extreme longshots: de-vig noise + loose books
                return
            ev_pct = fair_p * price - 1
            if ev_pct > MAX_EV:     # implausibly large -> one book's bad/stale price
                return
            # Longshots: de-vig is unreliable below ~8%; require a bigger margin.
            thresh = MIN_EV if fair_p >= 0.08 else 0.05
            if ev_pct >= thresh:
                f = min(max(0.0, ev_pct / (price - 1)) * KELLY, MAX_STAKE_FRAC) * bankroll
                rows.append((ev_pct, ev.home, ev.away, label, price, book, fair_p, f))

        # 1X2: model blended to the sharp consensus, then line-shopped.
        mp = markets(lam, mu, rho)["1x2"]
        fair = blend_market(mp, _odds_fair(sharp), w=BLEND_W)
        for sel in ("home", "draw", "away"):
            if sel in best:
                consider(sel.upper(), *best[sel], fair[sel])

        # Asian handicap: blend each two-way (side, line) pair the same way.
        sharp_ah, best_ah = feed.consensus_spreads(ev, sharp_only=True), feed.best_spreads(ev)
        if sharp_ah and best_ah:
            for L in sorted({l for (s, l) in sharp_ah if s == "home"}):
                ph, pa = sharp_ah.get(("home", L)), sharp_ah.get(("away", -L))
                if ph is None or pa is None:
                    continue
                cov = asian_handicap(lam, mu, rho, L)
                fp = blend_market(cov, _odds_fair({"home": ph, "away": pa}), w=BLEND_W)
                for side, line in (("home", L), ("away", -L)):
                    if (side, line) in best_ah:
                        consider(f"AH {side[0].upper()} {line:+g}", *best_ah[(side, line)], fp[side])

        # Totals: blend over/under per line.
        sharp_tot, best_tot = feed.consensus_totals(ev, sharp_only=True), feed.best_totals(ev)
        if sharp_tot and best_tot:
            lines = sorted({l for (l, s) in sharp_tot})
            ou = markets(lam, mu, rho, ou_lines=tuple(lines))["over_under"]
            for L in lines:
                po, pu = sharp_tot.get((L, "over")), sharp_tot.get((L, "under"))
                if po is None or pu is None or str(L) not in ou:
                    continue
                fp = blend_market(ou[str(L)], _odds_fair({"over": po, "under": pu}), w=BLEND_W)
                for side in ("over", "under"):
                    if (L, side) in best_tot:
                        consider(f"{side.upper()} {L:g}", *best_tot[(L, side)], fp[side])

    rows.sort(reverse=True)
    for ev_pct, h, a, label, price, book, fp, f in rows:
        print(f"  {h} v {a}  {label:11} @ {price:.2f} ({book[:12]})  "
              f"fair {fp:.1%}  EV {ev_pct:+.1%}  stake {f:.0f}")
    if not rows:
        print("  No line-shopping value vs the sharp line (as expected most days).")
    else:
        print(f"\n  {len(rows)} prices beat the sharp line (model blended to market "
              f"at w={BLEND_W:g}). Line-shop discrepancies, not model edges — small, "
              "perishable, verify before betting.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", default="soccer_fifa_world_cup")
    ap.add_argument("--regions", default="eu,uk")
    ap.add_argument("--list-sports", action="store_true")
    ap.add_argument("--value", action="store_true", help="scan for line-shopping value")
    ap.add_argument("--include-live", action="store_true",
                    help="also include already-started matches (live odds)")
    ap.add_argument("--bankroll", type=float, default=10000.0)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    try:
        if args.list_sports:
            for s in feed.list_soccer_sports(args.api_key):
                act = "" if s.get("active", True) else "  (inactive)"
                print(f"  {s['key']:<42}{s['title']}{act}")
            return

        mkts = "h2h,spreads,totals" if args.value else "h2h,totals"
        events, quota = feed.fetch_odds(args.sport, regions=args.regions,
                                        markets=mkts, api_key=args.api_key)
        total = len(events)
        if not args.include_live:
            events = [e for e in events if _is_upcoming(e)]
        skipped = total - len(events)
        print(f"{len(events)} upcoming events with odds"
              + (f" ({skipped} live/started skipped)" if skipped else "")
              + f"  |  quota left: {quota.get('x-requests-remaining')}"
              f"  (used {quota.get('x-requests-used')})\n")

        for ev in events:
            cons = feed.consensus_1x2(ev)
            best = feed.best_1x2(ev)
            if not cons:
                continue
            print(f"{ev.home} v {ev.away}   ({len(ev.books)} books)")
            print(f"  consensus:  H {cons['home']:.0%}  D {cons['draw']:.0%}  A {cons['away']:.0%}")
            shop = "  ".join(f"{s.upper()} {best[s][0]:.2f} [{best[s][1][:10]}]"
                             for s in ("home", "draw", "away") if s in best)
            print(f"  best price: {shop}")

        if args.value:
            from worldcup_model.model import ExpertModel
            print("\nValue scan (model anchored to sharp market, line-shopped):")
            cmd_value(events, ExpertModel.load(MODEL_PATH), args.bankroll)
        print(f"\nquota remaining: {quota.get('x-requests-remaining')}")
    except feed.OddsAPIError as e:
        print(f"[odds feed] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
