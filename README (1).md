# SPY ML Signal Bot — Setup Guide

Sends a Telegram message every weekday evening with tomorrow's SPY signal.
You then execute the trade manually on any platform (Syfe Trade, IBKR, Tiger, etc.)

---

## Files

| File | Purpose |
|---|---|
| `ml_trading_bot.py` | Core script — data, model, signal, Telegram |
| `scheduler.py` | Keeps the bot running daily |
| `last_signal_state.json` | Auto-created — tracks last signal to avoid spam |
| `requirements.txt` | Python dependencies |

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Create your Telegram bot (5 minutes)

1. Open Telegram, search for **@BotFather**
2. Send `/newbot` and follow prompts → you'll get a **Bot Token** like:
   `7123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
3. Start a chat with your new bot (search its name, hit Start)
4. Get your **Chat ID** by visiting in your browser:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   Look for `"chat":{"id": 123456789}` — that number is your Chat ID

---

## Step 3 — Set your credentials

**Option A — Environment variables (recommended):**
```bash
export TELEGRAM_BOT_TOKEN="7123456789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TELEGRAM_CHAT_ID="123456789"
```

Add these to your `~/.zshrc` or `~/.bashrc` so they persist.

**Option B — Edit the script directly:**
Open `ml_trading_bot.py` and replace:
```python
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID_HERE"
```

---

## Step 4 — Test it

```bash
# Run once, force a Telegram message regardless of signal state
python ml_trading_bot.py --force

# Run full backtest + signal (takes ~30s)
python ml_trading_bot.py --backtest --force
```

---

## Step 5 — Schedule it

### Option A: Keep a terminal open (simple)
```bash
python scheduler.py                  # runs at 9:30 PM SGT by default
python scheduler.py --time 21:30     # custom time
```

### Option B: cron job (reliable, runs even if terminal is closed)
```bash
crontab -e
```
Add this line (runs at 9:30 PM SGT every weekday):
```
30 21 * * 1-5 cd /path/to/your/folder && python ml_trading_bot.py >> bot.log 2>&1
```

### Option C: Free cloud VM (runs 24/7 without your laptop)
- **Google Cloud** e2-micro (free tier) or **Oracle Cloud** free VM
- SSH in, clone your files, set up cron as above
- Never need to leave your laptop on

---

## What you'll receive on Telegram

```
📊 SPY Daily Signal — 2026-04-21

Signal:  BUY 📈
Confidence:  61.0%

📌 Key Indicators
Close Price:     $524.30
RSI (14):        48.3
BB Z-Score:      0.12
vs SMA-20:       +0.84%

Model threshold: 55% | Act at next day's open
```

Signal only fires when it **changes** (BUY → OUT or OUT → BUY), so you won't get spammed.
You can override this with `--force` to get a daily reminder regardless.

---

## How to act on signals

| Signal | Action |
|---|---|
| BUY 📈 | Buy SPY at next day's open (or equivalent ETF on your platform) |
| STAY OUT 🔴 | Exit position / stay in cash |

**Platforms that work well for manual execution:**
- **IBKR** (best for SPY directly, available in SG)
- **Tiger Brokers / Moomoo** (US stocks, SG-friendly)
- **Syfe Trade** (if they carry SPY or similar US ETFs)

---

## Caveats

- This is a research tool, not financial advice
- Past backtest performance does not guarantee future results
- Always use position sizing appropriate to your risk tolerance
- SPY signals apply to the US market; check trading hours
