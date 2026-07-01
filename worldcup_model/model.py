"""ExpertModel: blends Dixon-Coles and Elo into market probs and value bets."""

from __future__ import annotations

import json

import pandas as pd

from . import data as datamod
from .dixon_coles import DixonColes
from .elo import EloModel
from .markets import markets
from .value import ValueBet, evaluate


def _ou_line(market: str) -> float:
    """Over/under line from a market key: 'over_under' -> 2.5, '...@3.5' -> 3.5."""
    return float(market.split("@", 1)[1]) if "@" in market else 2.5


class ExpertModel:
    def __init__(self, dc_weight: float = 0.8, half_life_days: float = 1460.0,
                 host_adv: float = 0.39, goal_cal: float = 1.0):
        # dc_weight blends Dixon-Coles vs Elo at the expected-goals level.
        # host_adv: extra goal-supremacy for a World Cup host (see host.py).
        # goal_cal: multiplicative calibration on expected goals. The base fit
        #   (trained on all internationals, incl. cagey qualifiers/friendlies)
        #   under-predicts goals at a high-scoring World Cup -- a walk-forward
        #   check on WC 2026 showed predicted 2.53 vs actual 2.94 total goals
        #   (over-2.5 predicted 42% vs actual 52%; favourite handicap covers
        #   under-predicted at -1/-1.5/-2). Scaling lam,mu up corrects the
        #   totals AND the favourite-margin bias in one parameter. 1.0 = off.
        self.dc_weight = dc_weight
        self.half_life_days = half_life_days
        self.host_adv = host_adv
        self.goal_cal = goal_cal
        self.dc = DixonColes()
        self.elo = EloModel()
        self.asof: pd.Timestamp | None = None

    # ---- training ------------------------------------------------------
    def fit(self, md: datamod.MatchData, asof: pd.Timestamp | None = None) -> "ExpertModel":
        self.asof = asof or md.played["date"].max()
        self.elo.fit(md.played)
        w = datamod.time_weights(md.played["date"], self.asof, self.half_life_days)
        self.dc.fit(md.played, w)
        return self

    # ---- prediction ----------------------------------------------------
    def expected_goals(self, home: str, away: str, neutral: bool = False,
                       host: bool = False, home_out: list | None = None,
                       away_out: list | None = None) -> tuple[float, float]:
        ld, md_ = self.dc.expected_goals(home, away, neutral)
        le, me = self.elo.expected_goals(home, away, neutral)
        w = self.dc_weight
        lam = w * ld + (1 - w) * le
        mu = w * md_ + (1 - w) * me
        lam *= self.goal_cal   # tournament goal calibration (see __init__)
        mu *= self.goal_cal
        if host:  # shift supremacy toward the host, keeping total goals ~fixed
            s = self.host_adv / 2.0
            lam, mu = lam + s, mu - s
        if home_out or away_out:  # injuries / suspensions
            from .squad import absence_penalty
            h_own, h_opp = absence_penalty(home_out or [])
            a_own, a_opp = absence_penalty(away_out or [])
            lam += a_opp - h_own   # home scores: less if home attackers out, more if away defenders out
            mu += h_opp - a_own
        return max(0.05, lam), max(0.05, mu)

    def predict(self, home: str, away: str, neutral: bool = False, ou_lines=(2.5,),
                host: bool = False, home_out: list | None = None,
                away_out: list | None = None) -> dict:
        lam, mu = self.expected_goals(home, away, neutral, host, home_out, away_out)
        out = markets(lam, mu, self.dc.rho, ou_lines=ou_lines)
        out["fixture"] = {"home": home, "away": away, "neutral": neutral}
        out["elo"] = {
            "home": round(self.elo.rating(home), 1),
            "away": round(self.elo.rating(away), 1),
        }
        return out

    def find_value(
        self,
        home: str,
        away: str,
        odds: dict[str, dict[str, float]],
        neutral: bool = False,
        bankroll: float = 1000.0,
        kelly: float = 0.25,
        min_edge: float = 0.02,
        host: bool = False,
        home_out: list | None = None,
        away_out: list | None = None,
    ) -> list[ValueBet]:
        """`odds` is {market: {selection: decimal_odds}}; returns value bets.

        Over/under markets are keyed "over_under" (line 2.5) or "over_under@3.5".
        """
        ou_lines = sorted({_ou_line(m) for m in odds if m.startswith("over_under")})
        pred = self.predict(home, away, neutral, ou_lines=tuple(ou_lines) or (2.5,),
                            host=host, home_out=home_out, away_out=away_out)
        bets: list[ValueBet] = []
        for market, sels in odds.items():
            model_probs = self._model_probs_for(pred, market)
            bets += evaluate(
                model_probs, sels, market, bankroll=bankroll,
                kelly_fraction_used=kelly, min_edge=min_edge,
                mutually_exclusive=(market != "double_chance"),
            )
        bets.sort(key=lambda b: b.ev, reverse=True)
        return bets

    @staticmethod
    def _model_probs_for(pred: dict, market: str) -> dict[str, float]:
        """Model probabilities for `market`, keyed to match the odds parser
        (selections are lower-cased; double-chance is 1x/12/x2)."""
        if market == "1x2":
            return pred["1x2"]
        if market == "double_chance":
            return {k.lower(): v for k, v in pred["double_chance"].items()}
        if market == "btts":
            return pred["btts"]
        if market.startswith("over_under"):
            d = pred["over_under"].get(str(_ou_line(market)))
            return {"over": d["over"], "under": d["under"]} if d else {}
        return {}

    # ---- persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        blob = {
            "dc_weight": self.dc_weight,
            "half_life_days": self.half_life_days,
            "host_adv": self.host_adv,
            "goal_cal": self.goal_cal,
            "asof": None if self.asof is None else self.asof.isoformat(),
            "dc": {
                "home_adv": self.dc.home_adv,
                "rho": self.dc.rho,
                "teams": self.dc.teams,
                "attack": self.dc.attack,
                "defence": self.dc.defence,
            },
            "elo": {
                "base": self.elo.base,
                "home_adv": self.elo.home_adv,
                "ratings": self.elo.ratings,
                "supremacy_slope": self.elo.supremacy_slope,
                "avg_total_goals": self.elo.avg_total_goals,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "ExpertModel":
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        m = cls(dc_weight=blob["dc_weight"], half_life_days=blob["half_life_days"],
                host_adv=blob.get("host_adv", 0.39),
                goal_cal=blob.get("goal_cal", 1.0))
        m.asof = None if blob["asof"] is None else pd.Timestamp(blob["asof"])
        d = blob["dc"]
        m.dc.home_adv, m.dc.rho, m.dc.teams = d["home_adv"], d["rho"], d["teams"]
        m.dc.attack, m.dc.defence = d["attack"], d["defence"]
        e = blob["elo"]
        m.elo.base, m.elo.home_adv = e["base"], e["home_adv"]
        m.elo.ratings = e["ratings"]
        m.elo.supremacy_slope, m.elo.avg_total_goals = e["supremacy_slope"], e["avg_total_goals"]
        return m
