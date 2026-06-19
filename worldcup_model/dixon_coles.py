"""Dixon-Coles time-weighted Poisson goals model.

Each team gets an attack and a defence rating. Expected goals are

    log(lambda_home) = attack_home + defence_away + home_adv      (home side)
    log(mu_away)     = attack_away + defence_home                 (away side)

with `home_adv` dropped at neutral venues. Goals are Poisson around those
rates, plus the Dixon-Coles low-score dependence correction (rho) that fixes
the well-known under-prediction of 0-0/1-1 draws. The log-likelihood is
weighted by match recency (see `data.time_weights`).

Fitting is two-stage: attack/defence/home_adv by L-BFGS-B with an analytic
gradient and a small ridge penalty (which also removes the attack/defence
additive degeneracy), then rho by a 1-D search holding the rates fixed. The
rho correction is small, so this is both fast and faithful to the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize


def _tau(h: np.ndarray, a: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles dependence correction on the four low-score cells."""
    out = np.ones_like(lam)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class DixonColes:
    ridge: float = 1e-3
    home_adv: float = 0.25
    rho: float = 0.0
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    teams: list[str] = field(default_factory=list)

    # ---- fitting -------------------------------------------------------
    def fit(self, played: pd.DataFrame, weights: np.ndarray) -> "DixonColes":
        teams = sorted(set(played["home_team"]) | set(played["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = played["home_team"].map(idx).to_numpy()
        ai = played["away_team"].map(idx).to_numpy()
        hs = played["home_score"].to_numpy(dtype=float)
        as_ = played["away_score"].to_numpy(dtype=float)
        home_flag = (~played["neutral"].to_numpy()).astype(float)
        w = np.asarray(weights, dtype=float)

        # Parameter vector: [attack(n), defence(n), home_adv].
        def unpack(theta):
            return theta[:n], theta[n : 2 * n], theta[2 * n]

        def rates(att, dfc, g):
            lam = np.exp(att[hi] + dfc[ai] + g * home_flag)
            mu = np.exp(att[ai] + dfc[hi])
            return lam, mu

        def negll_and_grad(theta):
            att, dfc, g = unpack(theta)
            lam, mu = rates(att, dfc, g)
            # Poisson log-likelihood (rho handled separately in stage 2).
            ll = w * ((hs * np.log(lam) - lam) + (as_ * np.log(mu) - mu))
            nll = -ll.sum() + self.ridge * (att @ att + dfc @ dfc)

            rh = w * (hs - lam)  # home-goal residual
            ra = w * (as_ - mu)  # away-goal residual
            g_att = np.zeros(n)
            g_dfc = np.zeros(n)
            # attack appears as home attack (rh) and as away attack (ra)
            np.add.at(g_att, hi, rh)
            np.add.at(g_att, ai, ra)
            # defence appears as away defence (rh) and as home defence (ra)
            np.add.at(g_dfc, ai, rh)
            np.add.at(g_dfc, hi, ra)
            g_home = float((home_flag * rh).sum())

            grad = np.concatenate([g_att, g_dfc, [g_home]])
            grad = -grad
            grad[:n] += 2 * self.ridge * att
            grad[n : 2 * n] += 2 * self.ridge * dfc
            return nll, grad

        theta0 = np.concatenate([np.zeros(2 * n), [self.home_adv]])
        res = optimize.minimize(
            negll_and_grad, theta0, jac=True, method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        att, dfc, g = unpack(res.x)
        # Centre attack to mean zero for interpretability (rates unchanged
        # because the ridge already fixed the gauge; this is cosmetic).
        shift = att.mean()
        att = att - shift
        dfc = dfc + shift

        self.teams = teams
        self.attack = dict(zip(teams, att))
        self.defence = dict(zip(teams, dfc))
        self.home_adv = float(g)

        self._fit_rho(hi, ai, hs, as_, home_flag, w, att, dfc, g)
        return self

    def _fit_rho(self, hi, ai, hs, as_, home_flag, w, att, dfc, g):
        lam = np.exp(att[hi] + dfc[ai] + g * home_flag)
        mu = np.exp(att[ai] + dfc[hi])
        hs_i = hs.astype(int)
        as_i = as_.astype(int)

        def neg_rho_ll(rho):
            tau = _tau(hs_i, as_i, lam, mu, rho)
            if np.any(tau <= 0):
                return 1e12
            return -float((w * np.log(tau)).sum())

        res = optimize.minimize_scalar(
            neg_rho_ll, bounds=(-0.2, 0.2), method="bounded"
        )
        self.rho = float(res.x)

    # ---- prediction ----------------------------------------------------
    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        if home not in self.attack or away not in self.attack:
            missing = [t for t in (home, away) if t not in self.attack]
            raise KeyError(f"unknown team(s): {missing}")
        g = 0.0 if neutral else self.home_adv
        lam = float(np.exp(self.attack[home] + self.defence[away] + g))
        mu = float(np.exp(self.attack[away] + self.defence[home]))
        return lam, mu

    def strength_table(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "team": self.teams,
                    "attack": [self.attack[t] for t in self.teams],
                    "defence": [self.defence[t] for t in self.teams],
                }
            )
            .assign(rating=lambda d: d["attack"] - d["defence"])
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )
