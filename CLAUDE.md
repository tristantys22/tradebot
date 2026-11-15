# tradebot — handoff notes for AI assistants

Read this before changing anything. Keep it up to date: add a line to the **Changelog** and fix any
section your change makes wrong.

## What this is
A Telegram bot that sends a daily SPY signal to subscribers and paper-trades two strategies on
**Alpaca paper** (never live money). Owner: Telegram admin chat `224111652` (`tg.ADMIN_CHAT_ID`).
Owner's goal: **risk-adjusted** performance (Sharpe, max drawdown) vs buy-and-hold. Allowed: other
tickers, intraday, shorting. Not allowed: leverage/margin. Ask the owner before changing what the
live strategies trade.

## Runtime
- **Fly.io** app `tradebot-tristan`, region `sin`, **exactly one** machine, `shared-cpu-1x` 512 MB.
  Two machines = two Telegram pollers (HTTP 409) and double trades. `fly scale count 1`.
- `Dockerfile` → `python bot_worker.py`. `.dockerignore` ships only root `*.py` + `requirements.txt`
  (so `research/` is not in the image). Image ~144 MB; idle RAM ~100 MB, peak ~135 MB.
- Deploy = `fly deploy` from the repo root. **Pushing to GitHub does not deploy** (no CI).
- Secrets (`fly secrets list`): `TELEGRAM_BOT_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`, `ALPACA_API_KEY`,
  `ALPACA_SECRET`, `ALPACA_PAPER_ENABLED=true`. Optional `TELEGRAM_CHAT_ID` (broadcast fallback).
- An unused 1 GB volume `tradebot_data` is mounted at `/data` (`fly.toml`); nothing writes to it.

## Code map
| File | Role |
|---|---|
| `bot_worker.py` | Entry point. Main thread: Telegram long-poll + command table (`PUBLIC`, `ADMIN`; add a command = add a dict entry). Background thread: `schedule`, one `daily_job` at **13:00 UTC (21:00 SGT, before the US open)** on weekdays: SPY, then NUAI. Registers the `/` menu and bot description on start. |
| `spy_strategy.py` | SPY strategy, Telegram signal text, `/signal` `/history` `/accuracy` `/paper` text, `rebalance()`, `--backtest`, `--force`. |
| `nuai_swing.py` | NUAI swing model, `rebalance()`, `/nuai` text, `backtest_report()` (`python nuai_swing.py`). |
| `broker.py` | Alpaca REST (`api()` returns `None` on 404). `free_cash()` = cash minus open *notional* buy orders. |
| `tg.py` | `send`, `send_admin`, `broadcast`, `call` (any Bot API method), subscribers, `poll_forever`. |
| `store.py` | Supabase table `bot_state(key, value text)`; values are JSON-encoded. `get` raises on HTTP errors so callers never overwrite data after a failed read. |
| `research/` | Offline research scripts (not deployed). See **Research**. |

Supabase keys: `subscribers`, `update_offset`, `last_signal_state`, `prediction_history` (≤1000; old
entries have `prob` from the retired ML model, new ones `weight`), `nuai_trades` (≤500),
`nuai_last_check`. `nuai_last_summary` is a stale key from the retired intraday strategy.

## Live strategies
**SPY (`spy_strategy.py`, 80% of equity).** Target weight = 0.5 × trend + 0.5 × dip.
- trend = 1 if close > SMA200, scaled by min(1, 15% / 20-day realised vol).
- dip = enter when RSI(2) < 10 and close > SMA200; exit when close > SMA5 (state recomputed from history).
- Rebalance to weight × 80% of equity, market orders at the open, cash only, skip if gap < 5% of the
  allocation, cancel stale SPY orders first. Drops today's still-forming bar.
- Backtest (decide at close, fill next open, 2 bp/side): Sharpe **1.00**, max DD −13%, ~8.7%/yr
  (2012-01→2024-11); holdout 2024-11→2026-11 Sharpe **0.98**, max DD −8%. SPY: 0.95/−32%, 0.91/−20%.

**NUAI (`nuai_swing.py`, 20% of equity, long only).** XGBoost (depth 2, 150 rounds, regularised),
features = NUAI returns/RSI/SMA gaps/vol/volume/gap/range + SPY & IWM 5-day returns; label = close
higher in 3 trading days. Retrained every run, averaged over 5 seeds; long if P(up) > 0.5, whole
shares, buys only with `broker.free_cash()`.
- **NUAI was a ~$10 SPAC until Nov 2024** — only data from 2024-12-01 is used (~490 days). Never
  backtest NUAI on earlier data.
- Walk-forward (10 bp/side): Sharpe 0.74 (2025-01→2025-11), **1.24** holdout 2025-12→2026-11
  (+93%, max DD −56% on the sleeve) vs buy-and-hold 1.05 / 0.32 (−40%). 16/18 seed/horizon variants
  positive in both periods. Only ~28 trades: treat live paper results as the real test.

## Research done (don't repeat without a new idea)
All with next-open fills, costs, walk-forward retraining, holdout checked once.
- **SPY XGBoost (retired 2026-11-15):** 5-day direction model. Walk-forward Sharpe ≤ 0.66 vs B&H
  0.95; adding macro features (TLT, HYG, GLD, UUP, QQQ, IWM, VIX term, yield curve) didn't help.
  Old deployed version trained once on the first 70% of data and never learned from the last 3 years.
- **je-suis-tm/quant-trading rules** (MACD, RSI 30/70, Bollinger, Parabolic SAR, Heikin-Ashi,
  Awesome Oscillator, Dual Thrust): none beat B&H on SPY; every short version lost money.
- **Pairs trading** (KO/PEP, XOM/CVX, V/MA, HD/LOW, GLD/GDX, JPM/BAC; rolling hedge, z=2): basket ≈ 0.
- **Foundation models, zero-shot, SPY 5-day:** Chronos-Bolt small/base Sharpe 0.66–0.78 (mostly
  predicts "up"), Kronos-small 49.9% hit rate, holdout Sharpe 0.34. Matches arXiv 2511.18578.
- **NUAI trend/breakout rules** looked great in 2025 (one +1700% run) and lost 46–87% in the holdout.
- **NUAI intraday z-score mean reversion** (retired) was never backtestable (60 days of 5-min data).

Scripts in `research/` are snapshots: `spy_strategies.py` imports the old `spy_model` module (now
`spy_strategy`; adapt before re-running). `foundation_models/` has its own Dockerfile (torch, ~1.5 GB).

## How to test (no local Python on the owner's PC)
Run code in a **throwaway Fly machine** built from the repo, never on the live machine (512 MB; a
second Python process there got the bot OOM-killed once):
```
fly deploy --build-only --push --image-label mytest
fly console -a tradebot-tristan --image registry.fly.io/tradebot-tristan:mytest \
  --vm-cpu-kind shared --vm-cpus 1 --vm-memory 1024 \
  --file-local /tmp/t.py=<local path> -C "sh -c 'cd /app; PYTHONPATH=/app python /tmp/t.py'"
```
- The machine gets the app's real secrets. In tests, monkeypatch `tg.send`, `store.set`,
  `broker.market_order`, `broker.close_position`, `broker.cancel_open_orders` before calling
  anything that writes or trades.
- A freshly pushed tag can 404 for ~1 min ("manifest unknown"); wait and retry.
- `fly console` sessions that time out can leave the machine running — `fly machine list` and
  `fly machine destroy <id> --force` anything that isn't the bot (`5683e766c10218`).
- For long jobs use `fly machine run <image> --rm --restart no ...` and read `fly logs`.
- After deploying: `fly logs --no-tail` should show `Bot worker started`. A `Traceback` a few
  seconds earlier is the old process being stopped, not an error.

## Gotchas
- Owner's shell is Windows PowerShell 5.1: quoting args to `flyctl ssh -C` breaks; base64-encode the
  script and run `sh -c 'echo <b64> | base64 -d | python'`, or use `fly console --file-local`.
  flyctl: `%USERPROFILE%\.fly\bin\flyctl.exe`; git: `C:\Program Files\Git\cmd\git.exe`.
- FRED (fred.stlouisfed.org) times out from Fly `sin`; use Yahoo series (`^TNX`, `^IRX`) instead.
- `xgboost.XGBClassifier` needs scikit-learn; this repo uses `xgb.train` (+ `QuantileDMatrix` when
  you need to match XGBClassifier exactly). Package is `xgboost-cpu` (no 300 MB NVIDIA lib).
- yfinance returns MultiIndex columns; flatten with `get_level_values(0)`.
- Alpaca keeps one position per symbol — two strategies on the same ticker will fight.
- Telegram messages use legacy `Markdown`: unbalanced `*`, `_` or backticks make sends fail.

## Open follow-ups
- 2026-11-16: confirm the SPY de-leverage sell filled (old bug had SPY at ~3.8× equity) and the
  first daily SPY + NUAI runs traded and messaged correctly (`/paper`, `/nuai`, `fly logs`).
- Review NUAI swing after a few weeks of live paper trades before trusting it.
- Possible: lower memory to 256 MB (peak ~135 MB) or delete the unused volume to save cost.
- `--backtest` in old commits overstated returns (compounded overlapping 5-day returns); current
  backtests use daily open-to-open returns.

## Changelog
- 2026-06-12 → 2026-11-14: original single-file XGBoost SPY bot; DB "not saving" was OOM kills at 256 MB.
- 2026-11-14: memory → 512 MB; refactor into modules; image 603 → 144 MB; Telegram long polling;
  SPY sizing fixed to cash (was margin buying power); prediction history kept to 1000.
- 2026-11-14: Telegram UX: `/help`, `/signal`, `/` menu, bot description, replies to unknown DMs.
- 2026-11-15: SPY XGBoost → trend + vol target / RSI(2) dip strategy; queued de-leverage sell.
- 2026-11-15: NUAI intraday mean reversion → NUAI daily XGBoost swing model; SPY 90% → 80%.
- 2026-11-15: added this file and `research/`.
