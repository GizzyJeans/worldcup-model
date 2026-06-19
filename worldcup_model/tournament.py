"""Monte-Carlo simulation of the 2026 World Cup from the current state.

Format: 48 teams in 12 groups of 4. Top 2 of each group (24) plus the 8 best
third-placed teams advance to a 32-team knockout (R32 -> R16 -> QF -> SF ->
Final). We reconstruct the groups from the fixture list, label them with their
real FIFA letters (A-L) by matching team-sets to the official draw, seed each
group's table with the results already played, then simulate the remaining
group games (model Poisson scorelines, FIFA tiebreakers: points, goal
difference, goals for) and the knockout rounds.

Knockout draw: we follow the OFFICIAL 2026 bracket. Group winners and runners-up
are seeded into fixed Round-of-32 slots, and the 8 best third-placed teams are
allocated to their designated winner-slots via FIFA's eligibility table (each
third-slot may only receive a third from a fixed set of groups). The R16/QF/SF
tree is the published fixed bracket, so a team's path -- and which opponents it
can meet, and when -- matches the real tournament (group winners get the
seeding protection of facing a third-placed team in the R32). Where FIFA's
table permits more than one eligibility-respecting allocation of thirds we pick
one deterministically; that only reshuffles which winner faces which third
within the allowed set, with negligible effect on reach-round odds. Host nations
keep their host advantage throughout (they play in their own country).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .host import HOSTS_2026, is_host_game
from .markets import score_matrix

# Official 2026 group draw (dataset spellings: Czech Republic, Turkey, Ivory
# Coast, Cape Verde, Iran). Used to label reconstructed groups with their real
# FIFA letter, which the official bracket is keyed to.
OFFICIAL_GROUPS_2026 = {
    "A": {"Mexico", "South Africa", "South Korea", "Czech Republic"},
    "B": {"Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"},
    "C": {"Brazil", "Morocco", "Haiti", "Scotland"},
    "D": {"United States", "Paraguay", "Australia", "Turkey"},
    "E": {"Germany", "Curaçao", "Ivory Coast", "Ecuador"},
    "F": {"Netherlands", "Japan", "Sweden", "Tunisia"},
    "G": {"Belgium", "Egypt", "Iran", "New Zealand"},
    "H": {"Spain", "Cape Verde", "Saudi Arabia", "Uruguay"},
    "I": {"France", "Senegal", "Iraq", "Norway"},
    "J": {"Argentina", "Algeria", "Austria", "Jordan"},
    "K": {"Portugal", "DR Congo", "Uzbekistan", "Colombia"},
    "L": {"England", "Croatia", "Ghana", "Panama"},
}

# Official Round-of-32 bracket (matches 73-88, in order). Each slot is
# ("W", group) winner, ("R", group) runner-up, or ("T", match_no) a
# third-placed team allocated to that slot's match by THIRD_SLOTS below.
R32_BRACKET = [
    (("R", "A"), ("R", "B")),    # 73
    (("W", "E"), ("T", 74)),     # 74
    (("W", "F"), ("R", "C")),    # 75
    (("W", "C"), ("R", "F")),    # 76
    (("W", "I"), ("T", 77)),     # 77
    (("R", "E"), ("R", "I")),    # 78
    (("W", "A"), ("T", 79)),     # 79
    (("W", "L"), ("T", 80)),     # 80
    (("W", "D"), ("T", 81)),     # 81
    (("W", "G"), ("T", 82)),     # 82
    (("R", "K"), ("R", "L")),    # 83
    (("W", "H"), ("R", "J")),    # 84
    (("W", "B"), ("T", 85)),     # 85
    (("W", "J"), ("R", "H")),    # 86
    (("W", "K"), ("T", 87)),     # 87
    (("R", "D"), ("R", "G")),    # 88
]

# Third-placed allocation: each third-slot (keyed by its R32 match number) may
# only receive a third-placed team from one of these groups (FIFA's official
# table). Verified to admit a valid assignment for all 495 ways 8 of the 12
# groups can supply the qualifying thirds.
THIRD_SLOTS = {
    74: set("ABCDF"), 77: set("CDFGH"), 79: set("CEFHI"), 80: set("EHIJK"),
    81: set("BEFIJ"), 82: set("AEHIJ"), 85: set("EFGIJ"), 87: set("DEIJL"),
}
_THIRD_MATCHES = list(THIRD_SLOTS)              # fixed slot order
_THIRD_ELIG = [THIRD_SLOTS[m] for m in _THIRD_MATCHES]

# Fixed knockout tree as index gathers into the previous round's winners
# (winners are ordered by match number). R32 winners 0..15 = matches 73..88.
R16_LEFT = np.array([1, 0, 3, 6, 10, 8, 13, 12])   # 89:74-77 90:73-75 ...
R16_RIGHT = np.array([4, 2, 5, 7, 11, 9, 15, 14])
QF_LEFT = np.array([0, 4, 2, 6])                   # 97:89-90 98:93-94 ...
QF_RIGHT = np.array([1, 5, 3, 7])
SF_LEFT = np.array([0, 2])                         # 101:97-98 102:99-100
SF_RIGHT = np.array([1, 3])


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


def label_groups(groups: list[list[str]]) -> dict[str, str]:
    """Map each reconstructed group to its official FIFA letter by team-set.

    Returns {team: letter}. Raises if a reconstructed group does not exactly
    match an official 2026 group (so a data change fails loudly rather than
    silently mislabelling the bracket)."""
    official = {frozenset(v): k for k, v in OFFICIAL_GROUPS_2026.items()}
    out: dict[str, str] = {}
    for g in groups:
        letter = official.get(frozenset(g))
        if letter is None:
            raise ValueError(
                f"group {sorted(g)} does not match any official 2026 group; "
                "update OFFICIAL_GROUPS_2026 (team spellings must match the data)"
            )
        for t in g:
            out[t] = letter
    return out


def _knockout_advance_prob(model, ti: str, tj: str,
                           inj_i: tuple[float, float] = (0.0, 0.0),
                           inj_j: tuple[float, float] = (0.0, 0.0)) -> float:
    """P(ti beats tj) at a neutral knockout (ET/pens weighted by strength).

    inj_* are each team's (own-goal reduction, opponent-goal increase) from
    absences: ti scores less if its attackers are out and more if tj's
    defenders are out."""
    own_i, opp_i = inj_i
    own_j, opp_j = inj_j
    lam, mu = model.expected_goals(ti, tj, neutral=True, host=False)
    bonus = (int(ti in HOSTS_2026) - int(tj in HOSTS_2026)) * model.host_adv
    lam = max(0.05, lam + bonus / 2 + opp_j - own_i)
    mu = max(0.05, mu - bonus / 2 + opp_i - own_j)
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

    def __init__(self, model, wc_played: pd.DataFrame, wc_upcoming: pd.DataFrame,
                 injuries: dict[str, list] | None = None):
        self.model = model
        groups = reconstruct_groups(pd.concat([wc_played, wc_upcoming]))
        self.group_of = label_groups(groups)        # team -> real FIFA letter
        # Order groups A..L by their official letter for stable indexing.
        groups.sort(key=lambda g: self.group_of[g[0]])
        self.teams = sorted(t for g in groups for t in g)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        self.group_letters = [self.group_of[g[0]] for g in groups]
        self.groups_idx = [np.array([self.idx[t] for t in g]) for g in groups]

        # Per-team absence penalties (own-goal reduction, opponent-goal rise),
        # applied to every future game that team plays. Unrecognised injured
        # players default to ~negligible "squad" impact; known stars keep theirs.
        self.inj_own = np.zeros(self.n)
        self.inj_opp = np.zeros(self.n)
        self.injured_teams: dict[str, tuple[list, float, float]] = {}
        if injuries:
            from .squad import absence_penalty
            for team, players in injuries.items():
                i = self.idx.get(team)
                if i is None or not players:
                    continue
                own, opp = absence_penalty(list(players), default_tier="squad")
                self.inj_own[i], self.inj_opp[i] = own, opp
                if own or opp:
                    self.injured_teams[team] = (list(players), own, opp)

        # Base table from games already played.
        self.base_pts = np.zeros(self.n)
        self.base_gd = np.zeros(self.n)
        self.base_gf = np.zeros(self.n)
        for r in wc_played.itertuples(index=False):
            self._apply(self.base_pts, self.base_gd, self.base_gf,
                        self.idx[r.home_team], self.idx[r.away_team],
                        int(r.home_score), int(r.away_score))

        # Remaining group fixtures with model expected goals (+ injury shift).
        h, a, lam, mu = [], [], [], []
        for r in wc_upcoming.itertuples(index=False):
            host = is_host_game(r.home_team, bool(r.neutral))
            lg, mg = model.expected_goals(r.home_team, r.away_team, bool(r.neutral), host)
            hi, ai = self.idx[r.home_team], self.idx[r.away_team]
            lg = max(0.05, lg + self.inj_opp[ai] - self.inj_own[hi])
            mg = max(0.05, mg + self.inj_opp[hi] - self.inj_own[ai])
            h.append(hi); a.append(ai)
            lam.append(lg); mu.append(mg)
        self.rh, self.ra = np.array(h, dtype=int), np.array(a, dtype=int)
        self.rlam, self.rmu = np.array(lam), np.array(mu)

        # Pairwise knockout advance-probability matrix (with injury shifts).
        self.adv = np.full((self.n, self.n), 0.5)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.adv[i, j] = _knockout_advance_prob(
                        model, self.teams[i], self.teams[j],
                        (self.inj_own[i], self.inj_opp[i]),
                        (self.inj_own[j], self.inj_opp[j]))

        self._alloc_cache: dict[frozenset, dict[int, str]] = {}

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

    def _allocate_thirds(self, qualifying: frozenset[str]) -> dict[int, str]:
        """Assign the 8 qualifying third-place groups to their R32 slots.

        Returns {match_no: group_letter}. Caches per set of qualifying groups
        (at most 495 distinct), so this is a dict lookup after warm-up."""
        cached = self._alloc_cache.get(qualifying)
        if cached is not None:
            return cached
        groups = sorted(qualifying)
        cost = np.array([[0 if g in elig else 1000 for g in groups]
                         for elig in _THIRD_ELIG])
        rows, cols = linear_sum_assignment(cost)
        if cost[rows, cols].sum() != 0:           # guaranteed not to happen
            raise RuntimeError(f"no valid third-place allocation for {groups}")
        alloc = {_THIRD_MATCHES[i]: groups[j] for i, j in zip(rows, cols)}
        self._alloc_cache[qualifying] = alloc
        return alloc

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
            win: dict[str, int] = {}
            runner: dict[str, int] = {}
            third: dict[str, int] = {}
            third_letters: list[str] = []
            third_keys: list[float] = []
            for letter, g in zip(self.group_letters, self.groups_idx):
                order = g[np.argsort(-key[g])]
                win[letter] = order[0]; runner[letter] = order[1]; third[letter] = order[2]
                C["win_group"][order[0]] += 1
                C["runner_up"][order[1]] += 1
                third_letters.append(letter); third_keys.append(key[order[2]])

            # 8 best third-placed teams (FIFA ranking = pts, gd, gf via `key`).
            best = np.argsort(-np.array(third_keys))[:8]
            alloc = self._allocate_thirds(frozenset(third_letters[b] for b in best))

            # Fill the 16 R32 matches from the official slot definitions.
            left = np.empty(16, dtype=int); right = np.empty(16, dtype=int)
            for mi, (sa, sb) in enumerate(R32_BRACKET):
                left[mi] = self._slot_team(sa, win, runner, third, alloc)
                right[mi] = self._slot_team(sb, win, runner, third, alloc)
            C["qualify"][np.concatenate([left, right])] += 1

            # Knockout: official fixed bracket, 5 rounds.
            w = np.where(rng.random(16) < self.adv[left, right], left, right)
            C["reach_R16"][w] += 1
            l, r = w[R16_LEFT], w[R16_RIGHT]
            w = np.where(rng.random(8) < self.adv[l, r], l, r)
            C["reach_QF"][w] += 1
            l, r = w[QF_LEFT], w[QF_RIGHT]
            w = np.where(rng.random(4) < self.adv[l, r], l, r)
            C["reach_SF"][w] += 1
            l, r = w[SF_LEFT], w[SF_RIGHT]
            w = np.where(rng.random(2) < self.adv[l, r], l, r)
            C["reach_final"][w] += 1
            champ = w[0] if rng.random() < self.adv[w[0], w[1]] else w[1]
            C["win_cup"][champ] += 1

        out = pd.DataFrame({"team": self.teams,
                            "group": [self.group_of[t] for t in self.teams]})
        for k, v in C.items():
            out[k] = v / n_sims
        return out.sort_values("win_cup", ascending=False).reset_index(drop=True)

    @staticmethod
    def _slot_team(slot, win, runner, third, alloc) -> int:
        kind, key = slot
        if kind == "W":
            return win[key]
        if kind == "R":
            return runner[key]
        return third[alloc[key]]          # ("T", match_no) -> allocated group
