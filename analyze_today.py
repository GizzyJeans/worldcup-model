"""Value analysis of today's World Cup 2026 board vs. the expert model.

Board odds are Malay-style; converted to decimal here:
    positive p (0..1) -> 1 + p          e.g.  0.88 -> 1.88
    negative p        -> 1 + 1/abs(p)   e.g. -0.98 -> 2.02

Markets priced: Full-Time 1X2 and Full-Time Over/Under (half and Asian
quarter lines). Asian quarter lines (e.g. 2.75) settle as two half-bets, so
their expected value accounts for the push on the integer half.
"""

from __future__ import annotations

import sys

import numpy as np

from worldcup_model.host import is_host_game
from worldcup_model.markets import score_matrix
from worldcup_model.model import ExpertModel
from worldcup_model.paths import MODEL_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

BANKROLL = 10_000.0
KELLY_FRAC = 0.25          # quarter-Kelly
MAX_STAKE_FRAC = 0.05      # cap any single bet at 5% of bankroll
MIN_EV = 0.03             # require >=3% model expected value to stake
MIN_EDGE = 0.02           # require >=2pt probability edge (filters noise)
MAX_EDGE = 0.12           # edge beyond this = model error/stale line, not value;
                           # the whole game is treated as untrustworthy


def kelly(ev: float, dec: float) -> float:
    """Kelly fraction = EV/(odds-1); exact for binary bets, conservative with
    pushes (quarter lines). Never negative."""
    return max(0.0, ev / (dec - 1.0))


def malay(p: float) -> float:
    """Malay odds -> decimal."""
    return 1.0 + p if p >= 0 else 1.0 + 1.0 / abs(p)


# (home, away, neutral, {market: ...}). Board 1X2 column order is Home/Away/Draw.
GAMES = [
    ("United States", "Australia", False, {
        "1x2": {"home": 1.59, "away": 5.49, "draw": 4.15},
        "ou": [(2.5, malay(0.88), malay(1.00))],           # over, under
    }),
    ("Scotland", "Morocco", True, {
        "1x2": {"home": 5.20, "away": 1.70, "draw": 3.60},
        "ou": [(2.25, malay(-0.99), malay(0.87))],
    }),
    ("Brazil", "Haiti", True, {
        "1x2": {"home": 1.10, "away": 20.99, "draw": 11.00},
        "ou": [(3.5, malay(0.85), malay(-0.99))],
    }),
    ("Turkey", "Paraguay", True, {
        "1x2": {"home": 2.02, "away": 3.79, "draw": 3.40},
        "ou": [(2.25, malay(0.85), malay(-0.97))],
    }),
]


def total_goals_dist(model: ExpertModel, home, away, neutral, host=False) -> np.ndarray:
    """P(total goals = k) from the blended scoreline matrix."""
    lam, mu = model.expected_goals(home, away, neutral, host)
    mat = score_matrix(lam, mu, model.dc.rho, max_goals=12)
    n = mat.shape[0]
    dist = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            dist[i + j] += mat[i, j]
    return dist


def _is_half(line: float) -> bool:
    return (line * 2) % 2 == 1  # x.5 line -> binary, no push


def _half_over(dist: np.ndarray, h: float, b: float) -> tuple[float, float]:
    """(win prob incl. half-push, EV) for an over bet on a single line `h`."""
    if _is_half(h):
        thr = int(np.ceil(h))                    # win if goals >= thr
        pw = dist[thr:].sum()
        return pw, pw * b - (1 - pw)
    k = int(round(h))                            # integer line: push on exactly k
    push, pw, pl = dist[k], dist[k + 1:].sum(), dist[:k].sum()
    return pw + 0.5 * push, pw * b - pl


def _half_under(dist: np.ndarray, h: float, b: float) -> tuple[float, float]:
    if _is_half(h):
        thr = int(np.floor(h))                   # win if goals <= thr
        pw = dist[: thr + 1].sum()
        return pw, pw * b - (1 - pw)
    k = int(round(h))
    push, pw, pl = dist[k], dist[:k].sum(), dist[k + 1:].sum()
    return pw + 0.5 * push, pw * b - pl


def _line_ev(half_fn, dist, line, dec):
    """Average two half-bets for quarter lines; single bet for half lines."""
    b = dec - 1.0
    if _is_half(line):
        return half_fn(dist, line, b)
    p1, ev1 = half_fn(dist, line - 0.25, b)      # lower half
    p2, ev2 = half_fn(dist, line + 0.25, b)      # upper half
    return 0.5 * (p1 + p2), 0.5 * (ev1 + ev2)


def over_ev(dist, line, dec):
    return _line_ev(_half_over, dist, line, dec)


def under_ev(dist, line, dec):
    return _line_ev(_half_under, dist, line, dec)


def main() -> None:
    model = ExpertModel.load(MODEL_PATH)
    candidates = []

    for home, away, neutral, mk in GAMES:
        host = is_host_game(home, neutral)
        pred = model.predict(home, away, neutral, host=host)
        x = pred["1x2"]
        venue = f"{home} HOST" if host else ("neutral" if neutral else f"{home} home")
        label_game = f"{home[:3].upper()} v {away[:3].upper()}"

        # Gather every selection's (label, odds, model prob, edge, ev).
        rows = []
        for sel, dec in mk["1x2"].items():
            p = x[sel]
            rows.append((f"1X2 {sel}", dec, p, p - 1 / dec, p * dec - 1))
        dist = total_goals_dist(model, home, away, neutral, host)
        for line, o_dec, u_dec in mk["ou"]:
            po, evo = over_ev(dist, line, o_dec)
            pu, evu = under_ev(dist, line, u_dec)
            rows.append((f"over {line}", o_dec, po, po - 1 / o_dec, evo))
            rows.append((f"under {line}", u_dec, pu, pu - 1 / u_dec, evu))

        # If the model grossly disagrees with the price on *any* selection,
        # distrust the whole game rather than bet the "huge edge".
        game_max_edge = max(abs(r[3]) for r in rows)
        flagged = game_max_edge > MAX_EDGE

        print(f"\n{home} vs {away}  ({venue})   Elo {pred['elo']['home']} vs {pred['elo']['away']}")
        print(f"  model 1X2: H {x['home']:.1%}  D {x['draw']:.1%}  A {x['away']:.1%}"
              f"   xG {pred['expected_goals']['home']}-{pred['expected_goals']['away']}"
              + ("   [FLAGGED: model vs market gap too large -> skip game]" if flagged else ""))
        for label, dec, p, edge, ev in rows:
            tag = ""
            if not flagged and ev >= MIN_EV and edge >= MIN_EDGE:
                tag = "  <= value"
                candidates.append([label_game, label, dec, p, edge, ev])
            print(f"    {label:<11} @{dec:>5.2f}  model {p:5.1%}  imp {1/dec:5.1%}"
                  f"  edge {edge:+6.1%}  EV {ev:+7.1%}{tag}")

    # Keep only the single best-EV 1X2 selection per game (mutually exclusive).
    best_1x2: dict[str, list] = {}
    deduped = []
    for c in candidates:
        if c[1].startswith("1X2"):
            cur = best_1x2.get(c[0])
            if cur is None or c[5] > cur[5]:
                best_1x2[c[0]] = c
        else:
            deduped.append(c)
    deduped += list(best_1x2.values())
    candidates = deduped

    # --- staking ---
    print("\n" + "=" * 70)
    print(f"RECOMMENDED SLIP  (bankroll {BANKROLL:,.0f}, {KELLY_FRAC:g}x Kelly, "
          f"cap {MAX_STAKE_FRAC:.0%}, min EV {MIN_EV:.0%})")
    print("=" * 70)
    if not candidates:
        print("No +value bets clear the threshold. Best to pass.")
        return
    candidates.sort(key=lambda c: c[5], reverse=True)
    total = exp_profit = 0.0
    print(f"  {'game':<12}{'bet':<11}{'odds':>6}{'model':>7}{'edge':>7}{'EV':>7}{'stake':>8}")
    for game, bet, dec, p, edge, ev in candidates:
        stake = min(kelly(ev, dec) * KELLY_FRAC, MAX_STAKE_FRAC) * BANKROLL
        total += stake
        exp_profit += stake * ev
        print(f"  {game:<12}{bet:<11}{dec:>6.2f}{p:>7.1%}{edge:>+7.1%}{ev:>+7.1%}{stake:>8.0f}")
    print("-" * 70)
    print(f"  {'TOTAL STAKED':<43}{total:>8.0f}  ({total/BANKROLL:.1%} of bankroll)")
    print(f"  {'EXPECTED PROFIT (if model is right)':<43}{exp_profit:>+8.0f}")


if __name__ == "__main__":
    main()
