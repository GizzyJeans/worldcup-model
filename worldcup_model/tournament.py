"""Monte-Carlo simulation of the 2026 World Cup from the current state.

Format: 48 teams in 12 groups of 4. Top 2 of each group (24) plus the 8 best
third-placed teams advance to a 32-team knockout (R32 -> R16 -> QF -> SF ->
Final). We reconstruct the groups from the fixture list, seed each group's
table with the results already played, then simulate the remaining group games
(model Poisson scorelines, FIFA tiebreakers: points, goal difference, goals
for) and the knockout rounds.

Knockout draw: the official bracket assigns slots by group position via a
fixed (and fiddly) table; to stay honest we instead use a NEUTRAL RANDOM DRAW
of the 32 qualifiers each simulation. That averages out bracket-draw luck and
gives strength-based "reach round X / win cup" odds — it slightly under-rewards
the seeding protection real group winners get. Host nations keep their host
advantage throughout (they play in their own country).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .host import HOSTS_2026, is_host_game
from .markets import score_matrix


def reconstruct_groups(matches: pd.DataFrame) -> list[list[str]]:
    """Recover the 12 groups as connected components (cliques of 4)."""
    adj: dict[str, set[str]] = {}
    for h, a in matches[["home_team", "away_team"]].itertuples(index=False):
        adj.setdefault(h, set()).add(a)
        adj.setdefault(a, set()).add(h)
    seen, groups = set(), []
    for t in adj:
        if t in seen:
            continue
        comp = {t} | adj[t]            # a team + its 3 opponents = its group
        seen |= comp
        groups.append(sorted(comp))
    return groups


def _knockout_advance_prob(model, ti: str, tj: str) -> float:
    """P(ti beats tj) at a neutral knockout (ET/pens weighted by strength)."""
    lam, mu = model.expected_goals(ti, tj, neutral=True, host=False)
    bonus = (int(ti in HOSTS_2026) - int(tj in HOSTS_2026)) * model.host_adv
    lam, mu = max(0.05, lam + bonus / 2), max(0.05, mu - bonus / 2)
    mat = score_matrix(lam, mu, model.dc.rho)
    n = mat.shape[0]
    iu = np.triu_indices(n, 1)
    p_i = float(mat[np.tril_indices(n, -1)].sum())   # home (ti) more goals
    p_j = float(mat[iu].sum())
    p_d = float(np.trace(mat))
    shoot = p_i / (p_i + p_j) if (p_i + p_j) > 0 else 0.5
    return p_i + p_d * shoot


class TournamentSimulator:
    ROUNDS = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win_cup"]

    def __init__(self, model, wc_played: pd.DataFrame, wc_upcoming: pd.DataFrame):
        self.model = model
        groups = reconstruct_groups(pd.concat([wc_played, wc_upcoming]))
        self.teams = sorted(t for g in groups for t in g)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        # Order groups A.. by their strongest team (Elo), for readable output.
        groups.sort(key=lambda g: -max(model.elo.rating(t) for t in g))
        self.group_of = {t: chr(65 + gi) for gi, g in enumerate(groups) for t in g}
        self.groups_idx = [np.array([self.idx[t] for t in g]) for g in groups]

        # Base table from games already played.
        self.base_pts = np.zeros(self.n)
        self.base_gd = np.zeros(self.n)
        self.base_gf = np.zeros(self.n)
        for r in wc_played.itertuples(index=False):
            self._apply(self.base_pts, self.base_gd, self.base_gf,
                        self.idx[r.home_team], self.idx[r.away_team],
                        int(r.home_score), int(r.away_score))

        # Remaining group fixtures with model expected goals.
        h, a, lam, mu, grp = [], [], [], [], []
        team_to_group = {t: gi for gi, g in enumerate(groups) for t in g}
        for r in wc_upcoming.itertuples(index=False):
            host = is_host_game(r.home_team, bool(r.neutral))
            lg, mg = model.expected_goals(r.home_team, r.away_team, bool(r.neutral), host)
            h.append(self.idx[r.home_team]); a.append(self.idx[r.away_team])
            lam.append(lg); mu.append(mg); grp.append(team_to_group[r.home_team])
        self.rh, self.ra = np.array(h), np.array(a)
        self.rlam, self.rmu = np.array(lam), np.array(mu)

        # Pairwise knockout advance-probability matrix.
        self.adv = np.full((self.n, self.n), 0.5)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.adv[i, j] = _knockout_advance_prob(
                        model, self.teams[i], self.teams[j])

    @staticmethod
    def _apply(pts, gd, gf, hi, ai, hs, as_):
        gd[hi] += hs - as_; gd[ai] += as_ - hs
        gf[hi] += hs; gf[ai] += as_
        if hs > as_:
            pts[hi] += 3
        elif hs < as_:
            pts[ai] += 3
        else:
            pts[hi] += 1; pts[ai] += 1

    def run(self, n_sims: int = 20000, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        C = {k: np.zeros(self.n) for k in
             ["win_group", "runner_up", "qualify", *self.ROUNDS]}

        for _ in range(n_sims):
            pts = self.base_pts.copy(); gd = self.base_gd.copy(); gf = self.base_gf.copy()
            gh = rng.poisson(self.rlam); ga = rng.poisson(self.rmu)
            # Apply simulated remaining group results.
            res = np.sign(gh - ga)
            np.add.at(pts, self.rh, np.where(res > 0, 3, np.where(res == 0, 1, 0)))
            np.add.at(pts, self.ra, np.where(res < 0, 3, np.where(res == 0, 1, 0)))
            np.add.at(gd, self.rh, gh - ga); np.add.at(gd, self.ra, ga - gh)
            np.add.at(gf, self.rh, gh); np.add.at(gf, self.ra, ga)

            # Rank within each group (pts, gd, gf, random tiebreak).
            key = pts * 1e6 + (gd + 200) * 1e3 + gf + rng.random(self.n)
            thirds = []
            qualifiers = []
            for g in self.groups_idx:
                order = g[np.argsort(-key[g])]
                C["win_group"][order[0]] += 1
                C["runner_up"][order[1]] += 1
                qualifiers.extend([order[0], order[1]])
                thirds.append(order[2])
            # 8 best third-placed teams.
            thirds = np.array(thirds)
            best = thirds[np.argsort(-key[thirds])[:8]]
            qualifiers = np.array(qualifiers + list(best))
            C["qualify"][qualifiers] += 1

            # Knockout: random neutral draw, 5 rounds.
            bracket = rng.permutation(qualifiers)
            for stage in self.ROUNDS:
                w = []
                for k in range(0, len(bracket), 2):
                    x, y = bracket[k], bracket[k + 1]
                    x_wins = rng.random() < self.adv[x, y]
                    w.append(x if x_wins else y)
                bracket = np.array(w)
                C[stage][bracket] += 1

        out = pd.DataFrame({"team": self.teams,
                            "group": [self.group_of[t] for t in self.teams]})
        for k, v in C.items():
            out[k] = v / n_sims
        return out.sort_values("win_cup", ascending=False).reset_index(drop=True)
