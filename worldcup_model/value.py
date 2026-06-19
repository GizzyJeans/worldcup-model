"""From model probabilities + bookmaker odds to value bets and stakes.

The edge of any bet is the gap between the model's probability and the price
the bookmaker offers. We strip the bookmaker's margin (overround) to recover
their implied "fair" probability for context, compute expected value, and size
stakes with the Kelly criterion (fractional, capped) for bankroll safety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def blend_market(model_probs: dict[str, float], market_odds: dict[str, float],
                 w: float = 0.9) -> dict[str, float]:
    """Market-anchored probabilities: log-pool of model and de-vigged market.

    Backtesting (`market_blend.py`) shows the sharp market carries essentially
    all the model's signal (optimal w ~= 0.9), so when a reliable price exists
    we anchor to it and let the model nudge only slightly. `w`=1 is market-only,
    `w`=0 is model-only."""
    fair = remove_margin(market_odds)
    keys = list(model_probs)
    lp = {k: (1 - w) * math.log(max(model_probs[k], 1e-9))
             + w * math.log(max(fair.get(k, 1e-9), 1e-9)) for k in keys}
    hi = max(lp.values())
    e = {k: math.exp(lp[k] - hi) for k in keys}
    z = sum(e.values())
    return {k: e[k] / z for k in keys}


def remove_margin(decimal_odds: dict[str, float]) -> dict[str, float]:
    """Proportionally de-vig a set of mutually exclusive odds -> fair probs."""
    raw = {k: 1.0 / v for k, v in decimal_odds.items()}
    total = sum(raw.values())  # = 1 + overround
    return {k: p / total for k, p in raw.items()}


def overround(decimal_odds: dict[str, float]) -> float:
    return sum(1.0 / v for v in decimal_odds.values()) - 1.0


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full-Kelly stake fraction; 0 when there's no edge."""
    b = decimal_odds - 1.0
    f = (prob * decimal_odds - 1.0) / b
    return max(0.0, f)


@dataclass
class ValueBet:
    market: str
    selection: str
    odds: float
    model_prob: float
    fair_prob: float        # bookmaker's de-vigged probability
    edge: float             # model_prob - implied_prob(odds)
    ev: float               # expected profit per unit staked
    kelly: float            # full-Kelly fraction
    stake: float            # recommended stake (fractional Kelly x bankroll)


def evaluate(
    model_probs: dict[str, float],
    odds: dict[str, float],
    market: str,
    bankroll: float = 1000.0,
    kelly_fraction_used: float = 0.25,
    max_stake_frac: float = 0.05,
    min_edge: float = 0.02,
    mutually_exclusive: bool = True,
) -> list[ValueBet]:
    """Score every selection in a market; return the +value ones, best first.

    `kelly_fraction_used` shrinks the aggressive full-Kelly stake (0.25 =
    quarter Kelly). `max_stake_frac` caps any single stake as a share of
    bankroll. `min_edge` is the minimum model-vs-price edge to flag a bet.
    `mutually_exclusive` enables overround removal for the "fair" column; set
    it False for overlapping markets like double chance.
    """
    if mutually_exclusive and len(odds) > 1:
        fair = remove_margin(odds)
    else:
        fair = {k: implied_prob(v) for k, v in odds.items()}
    bets: list[ValueBet] = []
    for sel, dec in odds.items():
        p = model_probs.get(sel)
        if p is None:
            continue
        edge = p - implied_prob(dec)
        if edge < min_edge:
            continue
        full_k = kelly_fraction(p, dec)
        stake = min(full_k * kelly_fraction_used, max_stake_frac) * bankroll
        bets.append(
            ValueBet(
                market=market,
                selection=sel,
                odds=dec,
                model_prob=p,
                fair_prob=fair.get(sel, implied_prob(dec)),
                edge=edge,
                ev=p * dec - 1.0,
                kelly=full_k,
                stake=round(stake, 2),
            )
        )
    bets.sort(key=lambda b: b.ev, reverse=True)
    return bets
