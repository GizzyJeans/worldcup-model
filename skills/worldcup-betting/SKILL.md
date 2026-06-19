---
name: worldcup-betting
description: Predict 2026 World Cup matches, simulate the tournament, fetch live multi-book odds, scan for betting value, and adjust for injuries — using the trained Dixon-Coles + Elo model in this repo. Use whenever the user asks about World Cup / international football match probabilities, odds, value bets, who will win the cup, group/knockout chances, or a betting board screenshot.
---

# World Cup betting model

A trained Dixon-Coles + Elo model lives in `{{ROOT}}`. The scripts resolve their
own data/model paths, so run them from any directory with absolute paths:
`python {{ROOT}}/<script>.py ...` (needs `pip install -r {{ROOT}}/requirements.txt`).
All commands print to stdout.

## Honest framing (state this, don't oversell)
Validation showed the model is **well-calibrated (~60% accuracy)** but has **no
proven edge against sharp closing lines** — it adds ~nothing beyond the market
(forecast combination puts ~92% weight on the price). Use it as a **forecasting
/ analysis tool**. The only edge path the backtests support is **line-shopping**
a soft price that beats the sharp consensus, and those are small and perishable.
Always size stakes small and caveat accordingly. Full write-up: `{{ROOT}}/README.md`.

## Commands

- Predict one match (markets, xG, 1X2; `--neutral` for most WC games):
  `python {{ROOT}}/predict.py predict --home Spain --away England --neutral`
- Adjust for injuries/suspensions (talisman ≈ −0.32 goals, star −0.18):
  `python {{ROOT}}/predict.py predict --home "United States" --away Australia --neutral --home-out "Christian Pulisic"`
- Find value vs given odds (decimal; supports 1x2 / over_under / btts / double_chance):
  `python {{ROOT}}/predict.py value --home Brazil --away Croatia --neutral --odds "1x2:home=1.95,draw=3.5,away=4.2" --kelly 0.25`
- Tournament odds (qualify / reach round / win cup) from the live state:
  `python {{ROOT}}/simulate.py --sims 20000`  (or `--group A`)
- Live multi-book odds + line-shopping + market-anchored value scan
  (needs a free key: `set ODDS_API_KEY=...` from the-odds-api.com):
  `python {{ROOT}}/fetch_odds.py --sport soccer_fifa_world_cup --value`
- Elo ratings: `python {{ROOT}}/predict.py ratings --n 25`
- Current injuries (needs `API_FOOTBALL_KEY` from api-football.com):
  `python {{ROOT}}/predict.py injuries`
- Validation: `backtest.py`, `roi_backtest.py` (`--book pinnacle-close|william-hill|best-soft`),
  `market_blend.py`. These need the odds dataset: `python {{ROOT}}/fetch_odds_data.py` first.
- Retrain (refresh data + refit, writes `model.json`): `python {{ROOT}}/train.py`

## Reading a betting-board screenshot
The 1X2 column is ordered **Home / Away / Draw** (not Home/Draw/Away). Odds are
often Malay: positive p → decimal `1+p`; negative p → `1 + 1/|p|`. "2-2.5" = the
Asian quarter line 2.25; "2.5-3" = 2.75. Hosts (USA/Canada/Mexico) play at home
(non-neutral, +host advantage); other 2026 matchups are neutral.

## Team names
Use the dataset's names exactly: `United States` (not USA), `South Korea`,
`DR Congo`, `Cape Verde`. Check with `predict.py ratings` if unsure.
