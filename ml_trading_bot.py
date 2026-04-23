import warnings
warnings.filterwarnings("ignore")
ADMIN_CHAT_ID = "224111652"  # your personal Telegram chat ID

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import schedule
from datetime import datetime, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path

# =========================
# CONFIG
# =========================
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None
TRAIN_SIZE = 0.7
INITIAL_CAPITAL = 10_000
PROB_THRESHOLD = 0.55
TRANSACTION_COST = 0.0005

# Telegram config — fill these in after creating your bot (see README)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

def sb_get(key: str):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/bot_state?key=eq.{key}&select=value",
        headers=sb_headers(), timeout=10
    )
    data = r.json()
    return data[0]["value"] if data else None

def sb_set(key: str, value: str):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/bot_state",
        headers=sb_headers(),
        json={"key": key, "value": value},
        timeout=10
    )


# =========================
# TELEGRAM
# =========================
def send_telegram(message: str, chat_id: str = None) -> bool:
    """Send a Telegram message. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram] ⚠️  Bot token not configured — skipping send.")
        print(f"[Telegram] Message would have been:\n{message}")
        return False

    target_chat_id = str(chat_id) if chat_id is not None else TELEGRAM_CHAT_ID
    if not target_chat_id:
        print("[Telegram] ⚠️  No chat_id available — skipping send.")
        print(f"[Telegram] Message would have been:\n{message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print(f"[Telegram] ✅ Message sent successfully to {target_chat_id}.")
        return True
    except Exception as e:
        print(f"[Telegram] ❌ Failed to send message to {target_chat_id}: {e}")
        return False

def broadcast_telegram(message: str) -> bool:
    """
    Send a message to all subscribers.
    Falls back to TELEGRAM_CHAT_ID if no subscribers exist.
    """
    chat_ids = load_subscribers()
    print(f"[Debug] broadcast_telegram subscribers={chat_ids}")

    if not chat_ids:
        print("[Telegram] No subscribers found. Falling back to TELEGRAM_CHAT_ID.")
        return send_telegram(message)

    success = False
    for chat_id in chat_ids:
        sent = send_telegram(message, chat_id)
        success = success or sent
        time.sleep(0.2)

    return success


def load_subscribers() -> list[str]:
    val = sb_get("subscribers")
    return json.loads(val) if val else []


def save_subscribers(chat_ids: list[str]):
    unique = sorted(set(str(c) for c in chat_ids))
    sb_set("subscribers", json.dumps(unique))


def add_subscriber(chat_id: str):
    chat_ids = load_subscribers()
    chat_id = str(chat_id)
    if chat_id not in chat_ids:
        chat_ids.append(chat_id)
        save_subscribers(chat_ids)
        print(f"[Telegram] Added subscriber: {chat_id}")


def remove_subscriber(chat_id: str):
    chat_id = str(chat_id)
    chat_ids = load_subscribers()
    updated = [cid for cid in chat_ids if cid != chat_id]
    save_subscribers(updated)
    print(f"[Telegram] Removed subscriber: {chat_id}")


def load_update_offset():
    val = sb_get("update_offset")
    return int(val) if val else None


def save_update_offset(offset: int):
    sb_set("update_offset", str(offset))


def get_updates(offset=None) -> list[dict]:
    if not TELEGRAM_BOT_TOKEN:
        return []

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": 10}
    if offset is not None:
        params["offset"] = offset

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("result", [])
    except Exception as e:
        print(f"[Telegram] ❌ Failed to fetch updates: {e}")
        return []

def save_prediction(sig: dict):
    """Append a prediction to the history log in Supabase."""
    history = load_prediction_history()
    history.append({
        "date": sig["date"],
        "signal": sig["signal_label"],
        "prob": sig["prob"],
        "close": sig["close"],
    })
    history = history[-30:]  # keep last 30
    sb_set("prediction_history", json.dumps(history))


def load_prediction_history() -> list:
    val = sb_get("prediction_history")
    return json.loads(val) if val else []


def calculate_accuracy() -> str:
    history = load_prediction_history()
    if len(history) < 2:
        return "Not enough data yet — need at least 2 predictions."

    correct = 0
    total = 0
    for i in range(len(history) - 1):
        current = history[i]
        next_entry = history[i + 1]
        predicted_up = current["signal"] == "BUY 📈"
        actually_up = next_entry["close"] > current["close"]
        if predicted_up == actually_up:
            correct += 1
        total += 1

    pct = (correct / total) * 100
    return f"🎯 *Bot Accuracy*\n\nCorrect: {correct}/{total} predictions\nAccuracy: `{pct:.1f}%`"

def sync_subscribers():
    offset = load_update_offset()
    updates = get_updates(offset=offset)

    print(f"[Debug] offset={offset} updates_found={len(updates)}")

    if not updates:
        return

    next_offset = offset or 0

    for upd in updates:
        next_offset = max(next_offset, upd["update_id"] + 1)
        msg = upd.get("message", {})
        text = (msg.get("text") or "").strip().lower()
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        print(f"[Debug] raw_text={msg.get('text')} chat_id={chat_id}")

        # Auto-subscribe when bot is added to a group
        my_chat_member = upd.get("my_chat_member", {})
        if my_chat_member:
            new_status = my_chat_member.get("new_chat_member", {}).get("status")
            group_chat = my_chat_member.get("chat", {})
            group_id = str(group_chat.get("id", ""))
            group_name = group_chat.get("title", "this group")

            if new_status == "member" and group_id:
                print(f"[Debug] Bot added to group {group_name} ({group_id})")
                add_subscriber(group_id)
                send_telegram(
                    f"👋 Thanks for adding me to *{group_name}*!\n\n"
                    "I'll send daily SPY signals here automatically.\n\n"
                    "📋 *Commands:*\n"
                    "/start — Subscribe\n"
                    "/stop — Unsubscribe\n"
                    "/history — Last 5 predictions\n"
                    "/accuracy — Bot accuracy",
                    group_id,
                )
            elif new_status in ("kicked", "left") and group_id:
                print(f"[Debug] Bot removed from group {group_name} ({group_id})")
                remove_subscriber(group_id)

        if not chat_id:
            continue

        chat_id = str(chat_id)

        if text == "/start":
            print(f"[Debug] /start from {chat_id}")
            add_subscriber(chat_id)
            print(f"[Debug] subscribers_now={load_subscribers()}")
            send_telegram(
                "✅ *Subscribed to Tristan's SPY ML Bot!*\n\n"
                "📋 *Available Commands:*\n"
                "/start — Subscribe to daily alerts\n"
                "/stop — Unsubscribe from alerts\n"
                "/history — View the last 5 predictions\n"
                "/accuracy — See the bot's prediction accuracy\n",
                chat_id,
            )

        elif text == "/stop":
            print(f"[Debug] /stop from {chat_id}")
            remove_subscriber(chat_id)
            print(f"[Debug] subscribers_now={load_subscribers()}")
            send_telegram("You have been unsubscribed from me. I hate you.", chat_id)


        elif text == "/history":
            history = load_prediction_history()
            if not history:
                send_telegram("No predictions recorded yet.", chat_id)
            else:
                lines = ["📅 *Last Predictions:*\n"]
                for p in history[-5:][::-1]:
                    lines.append(f"`{p['date']}` — {p['signal']} (`{p['prob']:.1%}`  ${p['close']})")
                send_telegram("\n".join(lines), chat_id)

        elif text == "/accuracy":
            msg = calculate_accuracy()
            send_telegram(msg, chat_id)

        elif text == "/force":
            print(f"[Debug] /force from {chat_id}, admin={ADMIN_CHAT_ID}")

            if chat_id != ADMIN_CHAT_ID:
                send_telegram(f"❌ You are not authorized. Your chat_id is {chat_id}", chat_id)
                continue

            send_telegram("⚡ Forcing prediction...", chat_id)
            # Re-run the signal generation and broadcast
            from sklearn.ensemble import RandomForestClassifier
            df = download_data(TICKER, START_DATE, END_DATE)
            df = add_features(df)
            model_df = df.dropna(subset=FEATURE_COLS + ["target", "future_ret_1d"]).copy()
            train_df, _ = time_split(model_df, TRAIN_SIZE)
            model = train_model(train_df)
            sig = generate_signal(model, df)
            prev_state = load_last_state()
            msg = build_telegram_message(sig, prev_state)
            broadcast_telegram(msg)
            save_state(sig["signal"], sig["prob"])

        elif text.startswith("/broadcast"):
            print(f"[Debug] /broadcast from {chat_id}, admin={ADMIN_CHAT_ID}")

            if chat_id != ADMIN_CHAT_ID:
                send_telegram(f"❌ You are not authorized. Your chat_id is {chat_id}", chat_id)
                continue

            parts = msg.get("text", "").split(" ", 1)
            if len(parts) < 2 or not parts[1].strip():
                send_telegram("Usage: /broadcast your message here", chat_id)
                continue

            broadcast_msg = parts[1].strip()
            print(f"[Debug] broadcasting='{broadcast_msg}' subscribers={load_subscribers()}")

            send_telegram("📢 Broadcasting message...", chat_id)
            broadcast_telegram(f"📢 *Broadcast*\n\n{broadcast_msg}")

    print(f"[Debug] saving next_offset={next_offset}")
    save_update_offset(next_offset)




def load_last_state() -> dict:
    val = sb_get("last_signal_state")
    return json.loads(val) if val else {"signal": None, "date": None, "prob": None}


def save_state(signal: int, prob: float, sig: dict = None):
    sb_set("last_signal_state", json.dumps({
        "signal": signal,
        "date": str(date.today()),
        "prob": round(prob, 4),
    }))
    if sig:
        save_prediction(sig)






# =========================
# DATA
# =========================
def download_data(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[["open", "high", "low", "close", "volume"]].copy()


# =========================
# FEATURE ENGINEERING
# =========================
def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"]  = out["close"].pct_change()
    out["ret_5d"]  = out["close"].pct_change(5)
    out["ret_10d"] = out["close"].pct_change(10)
    out["ret_20d"] = out["close"].pct_change(20)

    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["price_vs_sma20"] = out["close"] / out["sma_20"] - 1
    out["sma10_vs_sma50"] = out["sma_10"] / out["sma_50"] - 1

    bb_mid   = out["close"].rolling(20).mean()
    bb_std   = out["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_width"]       = (bb_upper - bb_lower) / bb_mid
    out["bb_zscore"]      = (out["close"] - bb_mid) / bb_std
    out["bb_pos"]         = (out["close"] - bb_lower) / (bb_upper - bb_lower)
    out["bb_break_upper"] = (out["close"] > bb_upper).astype(int)
    out["bb_break_lower"] = (out["close"] < bb_lower).astype(int)

    out["rsi_14"]  = compute_rsi(out["close"], 14)
    out["atr_14"]  = compute_atr(out, 14)
    out["atr_pct"] = out["atr_14"] / out["close"]
    out["vol_20d"] = out["ret_1d"].rolling(20).std()

    out["vol_chg_5d"]  = out["volume"].pct_change(5)
    out["vol_ratio_20"] = out["volume"] / out["volume"].rolling(20).mean()

    out["future_ret_1d"] = out["close"].pct_change().shift(-1)
    out["target"] = (out["future_ret_1d"] > 0).astype(int)
    return out


FEATURE_COLS = [
    "ret_5d", "ret_10d", "ret_20d",
    "price_vs_sma20", "sma10_vs_sma50",
    "bb_width", "bb_zscore", "bb_pos", "bb_break_upper", "bb_break_lower",
    "rsi_14", "atr_pct", "vol_20d", "vol_chg_5d", "vol_ratio_20",
]


# =========================
# TRAIN / TEST SPLIT
# =========================
def time_split(df: pd.DataFrame, train_size: float = 0.7):
    idx = int(len(df) * train_size)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


# =========================
# MODEL
# =========================
def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["target"])
    return model


# =========================
# BACKTEST
# =========================
def backtest(test_df: pd.DataFrame) -> pd.DataFrame:
    bt = test_df.copy()
    bt["position"]     = (bt["pred_prob"] >= PROB_THRESHOLD).astype(int)
    bt["position_lag"] = bt["position"].shift(1).fillna(0)
    bt["turnover"]     = bt["position_lag"].diff().abs().fillna(bt["position_lag"].abs())
    bt["cost"]         = bt["turnover"] * TRANSACTION_COST
    bt["strategy_ret"] = bt["position_lag"] * bt["future_ret_1d"] - bt["cost"]
    bt["buy_hold_ret"] = bt["future_ret_1d"]
    bt["equity_curve"]   = (1 + bt["strategy_ret"]).cumprod() * INITIAL_CAPITAL
    bt["buy_hold_curve"] = (1 + bt["buy_hold_ret"]).cumprod() * INITIAL_CAPITAL
    return bt


def annualized_return(returns, periods=252):
    r = returns.dropna()
    if len(r) == 0: return np.nan
    years = len(r) / periods
    return (1 + r).prod() ** (1 / years) - 1 if years > 0 else np.nan

def sharpe_ratio(returns, periods=252):
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0: return np.nan
    return np.sqrt(periods) * r.mean() / r.std()

def max_drawdown(equity):
    return (equity / equity.cummax() - 1).min()


def print_summary(bt):
    strat, bh = bt["strategy_ret"], bt["buy_hold_ret"]
    print("\n=== Backtest Summary ===")
    print(f"Strategy annualized return: {annualized_return(strat):.2%}")
    print(f"Strategy Sharpe ratio:      {sharpe_ratio(strat):.2f}")
    print(f"Strategy max drawdown:      {max_drawdown(bt['equity_curve']):.2%}")
    print(f"Final equity:               ${bt['equity_curve'].iloc[-1]:,.2f}")
    print("\n=== Buy & Hold Benchmark ===")
    print(f"Buy&Hold annualized return: {annualized_return(bh):.2%}")
    print(f"Buy&Hold Sharpe ratio:      {sharpe_ratio(bh):.2f}")
    print(f"Buy&Hold max drawdown:      {max_drawdown(bt['buy_hold_curve']):.2%}")
    print(f"Final equity:               ${bt['buy_hold_curve'].iloc[-1]:,.2f}")


# =========================
# SIGNAL GENERATION
# =========================
def generate_signal(model: RandomForestClassifier, df: pd.DataFrame) -> dict:
    """
    Use the latest available row to generate tomorrow's signal.
    Returns a dict with signal, probability, and key feature values.
    """
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    features = latest[FEATURE_COLS].values.reshape(1, -1)
    prob = model.predict_proba(features)[0, 1]
    signal = 1 if prob >= PROB_THRESHOLD else 0

    return {
        "date": str(date.today()),
        "signal": signal,
        "signal_label": "BUY 📈" if signal == 1 else "STAY OUT 🔴",
        "prob": round(prob, 4),
        "rsi": round(float(latest["rsi_14"]), 1),
        "bb_zscore": round(float(latest["bb_zscore"]), 2),
        "price_vs_sma20": round(float(latest["price_vs_sma20"]) * 100, 2),
        "close": round(float(df["close"].iloc[-1]), 2),
    }


def build_telegram_message(sig: dict, prev_state: dict) -> str:
    """Build a nicely formatted Telegram message."""
    signal_changed = prev_state["signal"] != sig["signal"]
    change_tag = "🔄 *SIGNAL CHANGED*\n" if signal_changed else ""

    lines = [
        f"{change_tag}",
        f"📊 *SPY Daily Signal — {sig['date']}*",
        f"",
        f"Signal:  *{sig['signal_label']}*",
        f"Confidence:  `{sig['prob']:.1%}`",
        f"",
        f"📌 *Key Indicators*",
        f"Close Price:     `${sig['close']}`",
        f"RSI (14):        `{sig['rsi']}`",
        f"BB Z-Score:      `{sig['bb_zscore']}`",
        f"vs SMA-20:       `{sig['price_vs_sma20']:+.2f}%`",
        f"",
        f"_Model threshold: {PROB_THRESHOLD:.0%} | Act at next day's open_",
    ]
    return "\n".join(lines)


# =========================
# MAIN — FULL PIPELINE
# =========================
def run_pipeline(backtest_mode: bool = False, force_notify: bool = False):
    print(f"\n{'='*50}")
    print(f"  SPY ML Signal Bot  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    sync_subscribers()

    print("\n[1/4] Downloading data...")
    df = download_data(TICKER, START_DATE, END_DATE)

    print("[2/4] Engineering features...")
    df = add_features(df)
    model_df = df.dropna(subset=FEATURE_COLS + ["target", "future_ret_1d"]).copy()

    print("[3/4] Training model...")
    train_df, test_df = time_split(model_df, TRAIN_SIZE)
    print(f"      Train: {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df)} rows)")
    print(f"      Test:  {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df)} rows)")

    model = train_model(train_df)

    test_df = test_df.copy()
    test_df["pred_prob"] = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    test_df["pred"]      = (test_df["pred_prob"] >= 0.5).astype(int)

    if backtest_mode:
        print("\n=== Classification Report ===")
        print(classification_report(test_df["target"], test_df["pred"], digits=4))
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print("\n=== Feature Importances ===")
        print(importances)
        bt = backtest(test_df)
        print_summary(bt)
        bt[["close","future_ret_1d","target","pred_prob","pred",
            "position","position_lag","strategy_ret","buy_hold_ret",
            "equity_curve","buy_hold_curve"]].to_csv("ml_trading_backtest_results.csv")
        print("\nSaved: ml_trading_backtest_results.csv")

    print("\n[4/4] Generating today's signal...")
    sig = generate_signal(model, df)
    prev_state = load_last_state()

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  Signal:  {sig['signal_label']:<20}│")
    print(f"  │  Prob:    {sig['prob']:<20.1%} │")
    print(f"  │  RSI:     {sig['rsi']:<20} │")
    print(f"  │  Close:   ${sig['close']:<19} │")
    print(f"  └─────────────────────────────┘")

    signal_changed = prev_state["signal"] != sig["signal"]
    already_alerted_today = prev_state["date"] == str(date.today())

    if signal_changed or force_notify or not already_alerted_today:
        msg = build_telegram_message(sig, prev_state)
        broadcast_telegram(msg)
        save_state(sig["signal"], sig["prob"],sig)
    else:
        print(f"\n[Telegram] Signal unchanged ({sig['signal_label']}) — no alert sent.")

    return sig


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPY ML Signal Bot")
    parser.add_argument("--backtest", action="store_true", help="Run full backtest report")
    parser.add_argument("--force",    action="store_true", help="Force Telegram alert even if signal unchanged")
    args = parser.parse_args()

    if args.backtest:
        run_pipeline(backtest_mode=True)
    elif args.force:
        run_pipeline(force_notify=True)
    else:
        print("Bot started. Running continuous loop with time check...")
    
        def get_last_run_date():
            return sb_get("last_run_date")
    
        def set_last_run_date(d):
            sb_set("last_run_date", d)
    
        while True:
            try:
                now = datetime.utcnow()
                today = str(now.date())
    
                # 13:00 UTC = 21:00 SGT
                if now.hour == 13 and 30 <= now.minute <= 45:
                    if get_last_run_date() != today:
                        print("[Scheduler] Running forced daily pipeline...")
                        run_pipeline(force_notify=True)
                        set_last_run_date(today)
                    else:
                        print(f"[Scheduler] Already ran today ({today})")
    
                # Always process Telegram commands
                sync_subscribers()
    
                time.sleep(10)
    
            except Exception as e:
                print(f"[Main Loop Error] {e}")
                time.sleep(5)
