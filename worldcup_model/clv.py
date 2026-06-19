"""Closing-line-value (CLV) bet picking — the honest, evidence-based core.

A results-only model cannot beat the sharp closing line, so this engine does
not try. It does the one thing the backtests support — LINE SHOPPING — and
scores it with the metric that actually measures bet-picking skill: closing-line
value.

Per event we de-vig a sharp consensus (our best estimate of the true
probabilities), find the best available price per outcome across books, and flag
a pick only when that best price beats the consensus by a margin. Picks are
sized with fractional Kelly on the consensus probability. When the line later
closes, `clv` compares the price we took against the de-vigged closing
consensus: positive CLV means we beat the close, which is the leading indicator
of long-run profit and far less noisy than ROI on a small sample.
"""

from __future__ import annotations

from dataclasses import dataclass

from .odds_feed import Event, best_1x2, consensus_1x2
from .value import kelly_fraction

OUTCOMES = ("home", "draw", "away")


@dataclass
class Pick:
    event: str
    commence_time: str
    market: str
    selection: str
    book: str               # book offering the best (taken) price
    price: float            # best available decimal price (line-shopped)
    consensus_prob: float   # de-vigged sharp consensus at pick time
    edge: float             # consensus_prob - 1/price (probability edge)
    ev: float               # consensus_prob * price - 1 (EV per unit staked)
    stake: float            # fractional-Kelly stake


def sharp_consensus(ev: Event) -> dict[str, float] | None:
    """De-vigged sharp-book consensus, falling back to all books if needed."""
    cons = consensus_1x2(ev, sharp_only=True)
    if cons is None:
        cons = consensus_1x2(ev, sharp_only=False)
    return cons


def make_picks(ev: Event, bankroll: float = 1000.0, kelly: float = 0.25,
               min_edge: float = 0.02, max_stake_frac: float = 0.05,
               top_only: bool = True) -> list[Pick]:
    """Line-shopping picks for one event: best price beats the sharp consensus.

    `min_edge` is the minimum *probability* edge (consensus_prob - 1/price) to
    flag a pick — thresholding on probability rather than EV keeps longshots,
    where consensus is noisiest, from being flagged on tiny mispricings. Staking
    is fractional Kelly on the consensus probability, capped at `max_stake_frac`.
    `top_only` keeps just the strongest pick per event (one bet per match)."""
    cons = sharp_consensus(ev)
    if cons is None:
        return []
    best = best_1x2(ev)
    picks: list[Pick] = []
    for sel in OUTCOMES:
        if sel not in best:
            continue
        price, book = best[sel]
        p = cons[sel]
        edge = p - 1.0 / price
        if edge < min_edge:
            continue
        frac = min(kelly_fraction(p, price) * kelly, max_stake_frac)
        picks.append(Pick(
            event=f"{ev.home} v {ev.away}", commence_time=ev.commence_time,
            market="1x2", selection=sel, book=book, price=round(price, 3),
            consensus_prob=round(p, 4), edge=round(edge, 4),
            ev=round(p * price - 1.0, 4), stake=round(frac * bankroll, 2)))
    if top_only and picks:
        picks = [max(picks, key=lambda q: q.edge)]
    return picks


def clv(price_taken: float, close_consensus_prob: float) -> float:
    """Closing-line value: EV of the taken price against the de-vigged close.

    >0 means we beat the close (the price we took implied a longer shot than the
    market's final fair estimate) — the signature of bet-picking skill."""
    return close_consensus_prob * price_taken - 1.0
