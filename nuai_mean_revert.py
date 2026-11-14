"""NUAI intraday mean reversion (long-only) on 5-minute bars, run every minute.

Buys when price is ENTRY_Z standard deviations below its rolling mean, sells on reversion,
on a stop loss, or before the close. The Alpaca position is the source of truth.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import broker
import store
import tg

SYMBOL = "NUAI"
LOOKBACK = 20                  # 5-min bars in the rolling mean (~100 min)
ENTRY_Z = -2.0
EXIT_Z = 0.0
STOP_LOSS_PCT = 0.05
POSITION_PCT = 0.10            # share of account equity per entry
FLATTEN_MINUTES_BEFORE_CLOSE = 10
TRADE_LOG_LIMIT = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def zscore() -> tuple[float, float, float]:
    close = yf.download(SYMBOL, period="5d", interval="5m", progress=False, auto_adjust=False)["Close"]
    close = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()
    mean, std = close.rolling(LOOKBACK).mean().iloc[-1], close.rolling(LOOKBACK).std().iloc[-1]
    price = float(close.iloc[-1])
    return price, float(mean), float((price - mean) / std)


def trades() -> list:
    return store.get("nuai_trades", [])


def _log(trade: dict):
    store.set("nuai_trades", (trades() + [trade | {"time": _now()}])[-TRADE_LOG_LIMIT:])


def _enter(price: float, z: float):
    qty = int(float(broker.account()["equity"]) * POSITION_PCT // price)
    if qty < 1:
        print(f"[NUAI] ⚠️ Position size rounds to 0 shares at ${price:.2f}")
        return
    order = broker.market_order(SYMBOL, "buy", qty=qty)
    _log({"side": "BUY", "price": round(price, 4), "qty": qty, "z": round(z, 2), "order_id": order["id"]})
    tg.send_admin(f"🟢 *NUAI BUY*\n\nQty: `{qty}`\nPrice: `${price:.2f}`\nZ-Score: `{z:.2f}`")


def _exit(pos: dict, reason: str, z: float = None):
    if broker.close_position(SYMBOL) is None:
        return
    pnl, pnl_pct = float(pos["unrealized_pl"]), float(pos["unrealized_plpc"]) * 100
    price = float(pos["current_price"])
    _log({"side": "SELL", "price": round(price, 4), "qty": float(pos["qty"]), "pnl": round(pnl, 2),
          "pnl_pct": round(pnl_pct, 2), "reason": reason, "z": None if z is None else round(z, 2)})
    tg.send_admin(f"{'✅' if pnl >= 0 else '❌'} *NUAI SELL* — {reason}\n\n"
                  f"Entry: `${float(pos['avg_entry_price']):.2f}`\nExit: `${price:.2f}`\n"
                  f"P&L: `${pnl:+,.2f}` (`{pnl_pct:+.2f}%`)")


def _stats(sells: list) -> str:
    if not sells:
        return "No closed trades."
    wins = sum(t["pnl"] > 0 for t in sells)
    return (f"Trades: `{len(sells)}` | Wins: `{wins}` | Win rate: `{wins / len(sells):.0%}`\n"
            f"Realized P&L: `${sum(t['pnl'] for t in sells):+,.2f}`")


def _daily_summary_once():
    today = _now()[:10]
    if store.get("nuai_last_summary") == today:
        return
    sells = [t for t in trades() if t["side"] == "SELL" and t["time"].startswith(today)]
    tg.send_admin(f"📊 *NUAI Daily Summary — {today}*\n\n{_stats(sells)}")
    store.set("nuai_last_summary", today)


def status_message() -> str:
    last, pos = store.get("nuai_last_check", {}), broker.position(SYMBOL)
    lines = ["📈 *NUAI Mean Reversion*", ""]
    if last:
        lines += [f"Price: `${last['price']:.2f}` | Mean: `${last['mean']:.2f}`",
                  f"Z-Score: `{last['z']:.2f}` (buy ≤ `{ENTRY_Z}`, exit ≥ `{EXIT_Z}`)",
                  f"Checked: `{last['time']}`", ""]
    lines.append(f"Position: `{float(pos['qty']):g}` sh @ `${float(pos['avg_entry_price']):.2f}` "
                 f"(`{float(pos['unrealized_plpc']) * 100:+.2f}%`)" if pos else "Position: *FLAT*")
    lines += ["", "*All-time*", _stats([t for t in trades() if t["side"] == "SELL"])]
    return "\n".join(lines)


def run_cycle():
    if not broker.ENABLED:
        return
    clock = broker.clock()
    if not clock["is_open"]:
        return
    pos = broker.position(SYMBOL)
    minutes_left = (pd.Timestamp(clock["next_close"]) - pd.Timestamp(clock["timestamp"])).total_seconds() / 60
    if minutes_left <= FLATTEN_MINUTES_BEFORE_CLOSE:
        if pos:
            _exit(pos, "Flatten before close")
        _daily_summary_once()
        return

    price, mean, z = zscore()
    if not np.isfinite(z):
        return
    print(f"[NUAI] price={price:.2f} mean={mean:.2f} z={z:.2f} {'LONG' if pos else 'FLAT'}")
    store.set("nuai_last_check", {"time": _now(), "price": round(price, 4), "mean": round(mean, 4), "z": round(z, 2)})

    if pos:
        if float(pos["unrealized_plpc"]) <= -STOP_LOSS_PCT:
            _exit(pos, "Stop loss", z)
        elif z >= EXIT_Z:
            _exit(pos, "Reverted to mean", z)
    elif z <= ENTRY_Z and not broker.has_open_order(SYMBOL):
        _enter(price, z)
