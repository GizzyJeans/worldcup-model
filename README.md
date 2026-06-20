# World Cup Betting — Expert Model

A statistical "expert" for international football betting. It fits a
**time-weighted Dixon-Coles Poisson** goals model and a **World Football Elo**
rating system on the full open history of men's internationals, blends them
into calibrated market probabilities, and compares those against bookmaker
odds to surface **positive expected-value bets** sized with the **Kelly
criterion**.

> Built for analysis and education. Betting carries risk; a model edge is a
> long-run statistical expectation, not a guaranteed outcome. Bet responsibly.

## Quick start on a new machine

Prerequisites: Python 3.12+, git (and `gh` to clone this private repo).

```bash
gh repo clone GizzyJeans/worldcup-model      # or: git clone <https-url>
cd worldcup-model
pip install -r requirements.txt
python install_skill.py                       # registers the /worldcup-betting skill
python predict.py predict --home Spain --away England --neutral   # verify
```

`model.json` is committed so prediction works immediately; `data/` auto-downloads.
Optional: `python fetch_odds_data.py` (validation scripts), and set `ODDS_API_KEY`
for live odds. No-install option: open in **GitHub Codespaces** (auto-configured).

**Daily sync:** `git pull` before working; `git add -A && git commit -m "..." && git push`
after. If you retrain (`python train.py`), push the new `model.json`.

## How it works

**Data** — `data/results.csv` from the open
[martj42/international_results](https://github.com/martj42/international_results)
dataset (every men's international from 1872 to today). It also contains the
*scheduled* fixtures for upcoming tournaments, which double as prediction
targets. Downloaded automatically on first run.

**Elo (`worldcup_model/elo.py`)** — each team carries a strength rating updated
chronologically after every match. The update scales with the result, the goal
margin, and fixture importance (a World Cup game moves ratings 3× a friendly).
In the same pass the model learns how an Elo gap maps to expected goal
*supremacy*, so Elo can emit its own scoreline expectation.

**Dixon-Coles (`worldcup_model/dixon_coles.py`)** — every team gets an *attack*
and *defence* parameter. Expected goals are

```
log λ_home = attack_home + defence_away + home_adv   (home_adv dropped if neutral)
log μ_away  = attack_away + defence_home
```

Goals are Poisson around `λ`/`μ`, plus the Dixon-Coles low-score dependence
correction (`ρ`) that fixes the classic under-prediction of 0-0/1-1 draws. The
log-likelihood is **recency-weighted** (an exponential half-life) so the fit
tracks current form. Fitting is two-stage: attack/defence/home-advantage via
L-BFGS-B with an analytic gradient and a ridge penalty, then `ρ` by a 1-D
search — fast (~4s on 12k matches) and faithful to the original method.

**Blend + markets (`markets.py`, `model.py`)** — Dixon-Coles and Elo expected
goals are blended (`--dc-weight`, default 0.8) into a single `λ`/`μ`, which
generates a full scoreline matrix. From that one matrix we derive **1X2**,
**double chance**, **over/under** (any line), **BTTS**, and **correct score**.

**Value + staking (`value.py`)** — for each selection: strip the bookmaker
margin (overround) for a "fair" reference, compute the model-vs-price **edge**
and **expected value**, and size the stake with **fractional Kelly** (default
quarter-Kelly), capped as a share of bankroll.

## Setup

```bash
pip install -r requirements.txt
python train.py            # downloads data, fits, writes model.json
```

Useful flags: `--since 2014-01-01` (history window), `--half-life 1460` (recency
in days), `--dc-weight 0.8` (Dixon-Coles vs Elo), `--refresh` (re-download).
The half-life and blend defaults were tuned on a walk-forward backtest with a
held-out validation slice (`python tune.py`).

## Usage

```bash
# Market probabilities for one match (most WC games are --neutral)
python predict.py predict --home Spain --away England --neutral

# Find value vs decimal odds, with bankroll + staking
python predict.py value --home Brazil --away Croatia --neutral \
    --odds "1x2:home=1.95,draw=3.5,away=4.2;over_under:over=2.0,under=1.85;btts:yes=1.8,no=2.0" \
    --bankroll 1000 --kelly 0.25

# Predict every upcoming World Cup fixture in the dataset
python predict.py fixtures --tournament "FIFA World Cup" --limit 40

# Rating table
python predict.py ratings --n 25
```

### Odds format

`market:selection=decimal_odds,...` blocks separated by `;`:

| Market          | Key                     | Selections          |
| --------------- | ----------------------- | ------------------- |
| Match result    | `1x2`                   | `home` `draw` `away`|
| Double chance   | `double_chance`         | `1x` `12` `x2`      |
| Over/Under 2.5  | `over_under`            | `over` `under`      |
| Over/Under N    | `over_under@3.5`        | `over` `under`      |
| Both teams score| `btts`                  | `yes` `no`          |

Example: `--odds "1x2:home=1.95,draw=3.5,away=4.2;over_under@3.5:over=2.6,under=1.5"`

## Layout

```
worldcup_model/
  data.py         load/clean results, recency weights
  elo.py          World Football Elo + Elo→goal-supremacy calibration
  dixon_coles.py  time-weighted Poisson fit (analytic gradient + ρ)
  host.py         World Cup host-nation advantage (data-estimated)
  markets.py      scoreline matrix → market probabilities
  value.py        de-vig, edge, EV, Kelly staking, market-anchored blend
  model.py        ExpertModel: blend, predict, find_value, save/load
  tournament.py   Monte-Carlo of the 2026 World Cup from the current state
  odds_feed.py    live multi-book odds (The Odds API): 1X2 + handicap consensus, best price
  squad.py        injury/suspension impact on expected goals (+ API-Football fetch)
  clv.py          line-shopping pick selection + closing-line-value scoring
train.py          fit and save model.json
predict.py        CLI: predict / value / fixtures / ratings
simulate.py       CLI: tournament odds (qualify / reach round / win cup)
fetch_odds.py     CLI: live odds, line-shopping, market-anchored value scan
picks.py          CLI: scan line-shopping value, settle, report CLV (skill metric)
analyze_today.py  value scan of a day's board (1X2 + Asian O/U)
backtest.py       walk-forward skill/calibration (log-loss, Brier, RPS)
wc_backtest.py    walk-forward 1X2 / O/U / handicap hit-rate on played fixtures
tune.py           grid-search half-life / blend weight (tune vs held-out split)
roi_backtest.py   ROI vs real bookmaker odds (--book pinnacle-close|...)
market_blend.py   forecast combination: model vs market signal
```

## Live odds & line shopping

Get a free key at [the-odds-api.com](https://the-odds-api.com) (500 calls/month),
then `set ODDS_API_KEY=yourkey`:

```bash
python fetch_odds.py --list-sports                          # find the sport key
python fetch_odds.py --sport soccer_fifa_world_cup          # best price + consensus
python fetch_odds.py --sport soccer_fifa_world_cup --value  # line-shopping value
```

For every match it shows the de-vigged **sharp consensus** and the **best
available price** per outcome across books. `--value` anchors the model to the
sharp market and flags any book whose price beats that consensus — the only
edge path the backtests support (line-shopping soft prices, not beating the
close). Discrepancies are small and short-lived; treat stakes as small.

## Bet picking & closing-line value (CLV)

`picks.py` is the bet-picking engine, built around the one honest truth the
backtests leave standing: a results model **cannot beat the sharp close**
(`roi_backtest.py`: ~−8% yield into Pinnacle, no edge), so it doesn't try. It
**line-shops** — bets a soft book only when its best price beats the de-vigged
sharp consensus — and scores every pick with **closing-line value**, the metric
that actually measures bet-picking skill (and is far less noisy than ROI).

```bash
python picks.py scan   --sport soccer_fifa_world_cup   # log today's value picks
python picks.py settle --sport soccer_fifa_world_cup   # record the close + CLV
python picks.py report                                 # avg CLV, beat-close rate
```

`scan` flags the outcome whose best price clears the consensus by `--min-edge`
(probability edge, so longshots aren't flagged on noise), sizes it with
fractional Kelly, and appends it to `picks_log.csv`. After the lines close,
`settle` fills each pick's closing consensus and CLV; `report` aggregates them —
**positive average CLV is the skill signal.** For offline use / testing, pass
`--odds-file board.json` (a list of events in the live feed's shape) instead of
`--sport`.

> Why forward-only: line-shopping needs several books per match, but the
> historical odds dataset has just **one book per international** — so the edge
> can't be backtested here, only measured going forward via the CLV log. And it
> remains line-shopping (exploiting soft-book price dispersion), **not** the
> model beating the market. Bet small; this is analysis, not a guaranteed edge.

## Injuries & suspensions

Adjust a prediction for who's missing — a talisman out is worth ~⅓ of a goal,
a squad player ~nothing (`worldcup_model/squad.py`, tiered estimates, editable):

```bash
python predict.py predict --home "United States" --away Australia --neutral \
    --home-out "Christian Pulisic"            # USA xG 1.00 -> 0.82, win 26% -> 21%
python predict.py injuries                     # auto-fetch (needs API_FOOTBALL_KEY)
```

A missing attacker lowers their team's expected goals; a missing defender/keeper
raises the opponent's. Impacts are literature-informed estimates, not learned
from data (our dataset has no player info) — verify rosters and tune before use.

## Tournament simulation

```bash
python simulate.py --sims 20000          # each team's qualify/QF/SF/final/cup odds
python simulate.py --group A             # one group's table
python simulate.py --injuries            # apply live injuries (needs API_FOOTBALL_KEY)
python simulate.py --injuries-file out.json   # apply injuries from a JSON file
```

Reconstructs the 12 groups from the fixtures (labelled with their real FIFA
letters A–L from the official draw), seeds tables with results already played,
then Monte-Carlos the remaining group games (model Poisson, FIFA tiebreakers)
and the knockout through the **official 2026 bracket**: group winners and
runners-up take their fixed Round-of-32 slots, the 8 best third-placed teams are
allocated to winners via FIFA's eligibility table, and the R16→QF→SF→final tree
is the published one — so group winners get real seeding protection and a team's
path matches the actual tournament.

With `--injuries` (or `--injuries-file '{"Team": ["Player", ...]}'`) each team's
absences shift its expected goals for *every* remaining game — group and
knockout — so a talisman injury ripples through the whole simulation. Known
stars use their `squad.py` tier; unrecognised auto-fetched players default to
~negligible impact (we can't gauge their importance). A summary lists the
applied teams, their goal impact, and any names that couldn't be matched.

## Validation summary

The model is **well-calibrated** (walk-forward: predicted ≈ observed, ~60%
accuracy) but has **no demonstrated betting edge**: it does not beat Pinnacle
closing lines, loses to soft books, and forecast combination puts ~92% weight
on the market (it adds essentially no independent signal). Treat it as a
**forecasting/analysis tool**, not a money-maker. See `worldcup-model-findings`.

## Continue on another device

This repo is the portable unit. On a new machine with Python + git:

```bash
git clone <your-repo-url> worldcup-model && cd worldcup-model
pip install -r requirements.txt
python install_skill.py        # registers the /worldcup-betting skill here
python fetch_odds_data.py       # only needed for roi_backtest / market_blend
python predict.py predict --home Spain --away England --neutral   # works immediately
```

`model.json` is committed, so prediction/simulation work right after clone;
`data/` (match results) auto-downloads on first use. For live odds set
`ODDS_API_KEY`.

From a **phone or any browser**: open the repo in **GitHub Codespaces** (Code ▸
Codespaces ▸ Create). The included `.devcontainer/` auto-installs dependencies,
so once it boots you can run e.g. `python simulate.py` straight from the browser
terminal. Set `ODDS_API_KEY` / `API_FOOTBALL_KEY` as Codespaces secrets to use
the live-odds and injury features there.

## Notes & extensions

- Team names match the dataset exactly (e.g. `South Korea`, `United States`).
- Possible extensions: bivariate-Poisson goal correlation and squad/lineup-
  based ratings. (The official knockout bracket is now implemented.)
