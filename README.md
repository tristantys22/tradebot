# tradebot

Telegram trading bot on Fly.io (`tradebot-tristan`) with Alpaca paper trading and Supabase state.

| File | What it does |
|---|---|
| `bot_worker.py` | Entry point: Telegram commands + schedule (SPY daily 13:00 UTC, NUAI every minute) |
| `spy_strategy.py` | SPY trend + vol target / RSI(2) dip strategy: signal, broadcast, rebalance |
| `nuai_mean_revert.py` | NUAI intraday mean reversion on 5-min bars |
| `broker.py` | Alpaca REST client |
| `tg.py` | Telegram send / broadcast / subscribers / polling |
| `store.py` | Supabase `bot_state` key/value store |

Commands: `/start` `/stop` `/history` `/accuracy`; admin: `/force` `/paper` `/nuai` `/broadcast <msg>`.

Secrets (`fly secrets set`): `TELEGRAM_BOT_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`,
`ALPACA_API_KEY`, `ALPACA_SECRET`, `ALPACA_PAPER_ENABLED=true`.

Deploy: `fly deploy` · Logs: `fly logs` · Backtest: `fly ssh console -C "python spy_strategy.py --backtest"`

Run exactly one machine (`fly scale count 1`): two pollers make Telegram return 409 and double-trade.
