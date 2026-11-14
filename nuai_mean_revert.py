"""
NUAI intraday mean reversion (long-only) on 5-minute bars.
Called every minute by bot_worker.py. Alpaca position is the source of truth;
trades and status are logged to Supabase bot_state, alerts go to the admin chat.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import alpaca_trader
from ml_trading_bot import sb_get, sb_set, send_telegram, ADMIN_CHAT_ID

SYMBOL = "NUAI"
LOOKBACK = 20                  # 5-min bars in the rolling mean (~100 min)
ENTRY_Z = -2.0                 # buy when price is this many std devs below the mean
EXIT_Z = 0.0                   # sell once price is back to the mean
STOP_LOSS_PCT = 0.05           # hard stop: -5% from entry
POSITION_PCT = 0.10            # fraction of account equity per entry
FLATTEN_MINUTES_BEFORE_CLOSE = 10


def _alpaca(method: str, path: str, **kwargs):
    r = requests.request(
        method, f"{alpaca_trader.ALPACA_BASE_URL}{path}",
        headers=alpaca_trader._headers(), timeout=15, **kwargs,
    )
    r.raise_for_status()
    return r.json()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return str(datetime.now(timezone.utc).date())


# =========================
# DATA / SIGNAL
# =========================
def get_bars() -> pd.DataFrame:
    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower).dropna(subset=["close"])


def compute_z(df: pd.DataFrame) -> tuple[float, float, float]:
    close = df["close"]
    mean = close.rolling(LOOKBACK).mean()
    std = close.rolling(LOOKBACK).std()
    z = (close - mean) / std
    return float(close.iloc[-1]), float(mean.iloc[-1]), float(z.iloc[-1])


# =========================
# TRADE LOG (Supabase)
# =========================
def load_trades() -> list:
    val = sb_get("nuai_trades")
    return json.loads(val) if val else []


def log_trade(entry: dict):
    trades = load_trades()
    trades.append(entry)
    sb_set("nuai_trades", json.dumps(trades[-200:]))


# =========================
# EXECUTION
# =========================
def _has_open_order() -> bool:
    return len(_alpaca("GET", "/v2/orders", params={"status": "open", "symbols": SYMBOL})) > 0


def _enter(price: float, z: float):
    account = alpaca_trader.get_account()
    if not account:
        return
    qty = int(float(account["equity"]) * POSITION_PCT // price)
    if qty < 1:
        print(f"[NUAI] ⚠️ Position size rounds to 0 shares at ${price:.2f}")
        return
    try:
        order = _alpaca("POST", "/v2/orders", json={
            "symbol": SYMBOL, "qty": str(qty), "side": "buy",
            "type": "market", "time_in_force": "day",
        })
    except Exception as e:
        print(f"[NUAI] ❌ Buy failed: {e}")
        send_telegram(f"❌ *NUAI buy failed*\n`{e}`", ADMIN_CHAT_ID)
        return
    log_trade({"time": _now_iso(), "side": "BUY", "price": round(price, 4), "qty": qty,
               "z": round(z, 2), "order_id": order.get("id")})
    send_telegram(
        f"🟢 *NUAI BUY*\n\nQty: `{qty}`\nPrice: `${price:.2f}`\nZ-Score: `{z:.2f}`",
        ADMIN_CHAT_ID,
    )


def _exit(position: dict, reason: str, z: float = None):
    if not alpaca_trader.submit_sell_all(SYMBOL):
        send_telegram(f"❌ *NUAI sell failed* ({reason})", ADMIN_CHAT_ID)
        return
    price = float(position["current_price"])
    pnl = float(position["unrealized_pl"])
    pnl_pct = float(position["unrealized_plpc"]) * 100
    log_trade({"time": _now_iso(), "side": "SELL", "price": round(price, 4),
               "qty": float(position["qty"]), "pnl": round(pnl, 2),
               "pnl_pct": round(pnl_pct, 2), "reason": reason,
               "z": round(z, 2) if z is not None else None})
    emoji = "✅" if pnl >= 0 else "❌"
    send_telegram(
        f"{emoji} *NUAI SELL* — {reason}\n\n"
        f"Entry: `${float(position['avg_entry_price']):.2f}`\nExit: `${price:.2f}`\n"
        f"P&L: `${pnl:+,.2f}` (`{pnl_pct:+.2f}%`)",
        ADMIN_CHAT_ID,
    )


# =========================
# REPORTING
# =========================
def _stats(sells: list) -> str:
    if not sells:
        return "No closed trades."
    wins = sum(1 for t in sells if t["pnl"] > 0)
    total = sum(t["pnl"] for t in sells)
    return (f"Trades: `{len(sells)}` | Wins: `{wins}` | Win rate: `{wins / len(sells):.0%}`\n"
            f"Realized P&L: `${total:+,.2f}`")


def _send_daily_summary_once():
    today = _today()
    if sb_get("nuai_last_summary") == today:
        return
    sells = [t for t in load_trades() if t["side"] == "SELL" and t["time"].startswith(today)]
    send_telegram(f"📊 *NUAI Daily Summary — {today}*\n\n{_stats(sells)}", ADMIN_CHAT_ID)
    sb_set("nuai_last_summary", today)


def get_status_message() -> str:
    last = sb_get("nuai_last_check")
    last = json.loads(last) if last else {}
    position = alpaca_trader.get_position(SYMBOL)
    sells = [t for t in load_trades() if t["side"] == "SELL"]
    lines = [f"📈 *NUAI Mean Reversion*", ""]
    if last:
        lines += [f"Price: `${last['price']:.2f}` | Mean: `${last['mean']:.2f}`",
                  f"Z-Score: `{last['z']:.2f}` (buy ≤ `{ENTRY_Z}`, exit ≥ `{EXIT_Z}`)",
                  f"Checked: `{last['time']}`", ""]
    if position:
        lines.append(f"Position: `{float(position['qty']):g}` sh @ `${float(position['avg_entry_price']):.2f}` "
                     f"(`{float(position['unrealized_plpc']) * 100:+.2f}%`)")
    else:
        lines.append("Position: *FLAT*")
    lines += ["", "*All-time*", _stats(sells)]
    return "\n".join(lines)


# =========================
# MAIN CYCLE
# =========================
def run_cycle():
    if not alpaca_trader.is_enabled():
        return
    clock = _alpaca("GET", "/v2/clock")
    if not clock["is_open"]:
        return
    minutes_to_close = (pd.Timestamp(clock["next_close"]) - pd.Timestamp(clock["timestamp"])).total_seconds() / 60
    position = alpaca_trader.get_position(SYMBOL)

    if minutes_to_close <= FLATTEN_MINUTES_BEFORE_CLOSE:
        if position:
            _exit(position, "Flatten before close")
        _send_daily_summary_once()
        return

    df = get_bars()
    if len(df) < LOOKBACK:
        return
    price, mean, z = compute_z(df)
    if not np.isfinite(z):
        return
    print(f"[NUAI] price={price:.2f} mean={mean:.2f} z={z:.2f} position={'LONG' if position else 'FLAT'}")
    sb_set("nuai_last_check", json.dumps(
        {"time": _now_iso(), "price": round(price, 4), "mean": round(mean, 4), "z": round(z, 2)}))

    if position:
        if float(position["unrealized_plpc"]) <= -STOP_LOSS_PCT:
            _exit(position, "Stop loss", z)
        elif z >= EXIT_Z:
            _exit(position, "Reverted to mean", z)
    elif z <= ENTRY_Z and not _has_open_order():
        _enter(price, z)
