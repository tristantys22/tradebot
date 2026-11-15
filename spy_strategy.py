"""SPY daily strategy: 50% SMA200 trend with a volatility target + 50% RSI(2) dip-buying in uptrends.

Runs before the US open: decides from the last completed daily bar, trades at the open.
Backtest (open-to-open fills, 2 bp/side), 2012-2024: Sharpe 1.00, max drawdown -13% (SPY: 0.95, -32%).

Manual run:  python spy_strategy.py --force      (run now and broadcast)
             python spy_strategy.py --backtest   (print backtest, no side effects)
"""
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

import broker
import store
import tg

TICKER = "SPY"
START_DATE = "2003-01-01"
TREND_SMA = 200
TARGET_VOL = 0.15            # annualised; the trend half scales down when SPY is more volatile
VOL_WINDOW = 20
DIP_ENTRY_RSI = 10           # RSI(2) below this, in an uptrend, opens the dip half
DIP_EXIT_SMA = 5             # dip half closes once price is back above this average
TREND_WEIGHT = 0.5           # the dip half gets the rest
SPY_ALLOCATION = 0.80        # share of account equity SPY may use (NUAI swing uses up to 20%)
REBALANCE_BAND = 0.05        # skip trades smaller than this share of the SPY allocation
HISTORY_LIMIT = 1000


# ---------- signal ----------
def load_prices(start=START_DATE) -> pd.DataFrame:
    df = yf.download(TICKER, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {TICKER}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    now_ny = pd.Timestamp.now(tz="America/New_York")
    if df.index[-1].date() == now_ny.date() and now_ny.hour < 16:
        df = df.iloc[:-1]  # today's bar is still forming
    return df


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _hold(enter: pd.Series, leave: pd.Series) -> pd.Series:
    """1 from the first `enter` day until a `leave` day, else 0."""
    out, held = np.zeros(len(enter)), 0
    for i, (e, x) in enumerate(zip(enter.to_numpy(), leave.to_numpy())):
        if held and x:
            held = 0
        if not held and e:
            held = 1
        out[i] = held
    return pd.Series(out, enter.index)


def target_weights(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    uptrend = c > c.rolling(TREND_SMA).mean()
    vol = c.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252)
    rsi2 = _rsi(c, 2)
    trend = uptrend * (TARGET_VOL / vol).clip(upper=1.0).fillna(0)
    dip = _hold(uptrend & (rsi2 < DIP_ENTRY_RSI), c > c.rolling(DIP_EXIT_SMA).mean())
    return pd.DataFrame({"weight": TREND_WEIGHT * trend + (1 - TREND_WEIGHT) * dip, "trend": trend,
                         "dip": dip, "uptrend": uptrend, "vol": vol, "rsi2": rsi2,
                         "sma_gap": c / c.rolling(TREND_SMA).mean() - 1})


def _label(weight: float) -> str:
    return "STAY OUT 🔴" if weight < 0.01 else ("FULLY IN 📈" if weight >= 0.95 else "PARTLY IN 🟡")


def compute() -> dict:
    df = load_prices()
    last = target_weights(df).iloc[-1]
    weight = round(float(last["weight"]), 4)
    return {"date": str(date.today()), "data_date": str(df.index[-1].date()), "weight": weight,
            "signal_label": _label(weight), "close": round(float(df["close"].iloc[-1]), 2),
            "trend": float(last["trend"]), "dip": int(last["dip"]), "uptrend": bool(last["uptrend"]),
            "vol": float(last["vol"]), "rsi2": float(last["rsi2"]), "sma_gap": float(last["sma_gap"])}


def format_message(s: dict, changed: bool) -> str:
    return "\n".join([
        "🔄 *POSITION CHANGE*\n" if changed else "",
        f"📊 *SPY Daily Signal — {s['date']}*",
        "",
        f"Signal:  *{s['signal_label']}*",
        f"Target exposure:  `{s['weight']:.0%}` of the portfolio",
        "",
        "📌 *Why*",
        f"Trend: SPY `{s['sma_gap']:+.1%}` vs its 200-day average → trend half `{s['trend']:.0%}` invested",
        f"Volatility: `{s['vol']:.1%}` a year (target {TARGET_VOL:.0%}; higher = smaller position)",
        f"RSI(2): `{s['rsi2']:.0f}` → dip half *{'IN' if s['dip'] else 'OUT'}*",
        f"Close ({s['data_date']}):  `${s['close']}`",
        "",
        "_50% trend + volatility target, 50% RSI(2) dip-buying. Updated weekdays 9pm SGT._",
    ])


# ---------- state ----------
def history() -> list:
    return store.get("prediction_history", [])


def _invested(p: dict) -> bool:
    return p["weight"] > 0 if "weight" in p else p["signal"] == "BUY 📈"  # older entries came from the ML model


def _save(s: dict):
    store.set("last_signal_state", {"signal": s["signal_label"], "date": s["date"], "weight": s["weight"]})
    hist = [p for p in history() if p["date"] != s["date"]]  # a re-run replaces today's entry
    hist.append({"date": s["date"], "signal": s["signal_label"], "weight": s["weight"], "close": s["close"]})
    store.set("prediction_history", hist[-HISTORY_LIMIT:])


def _describe(p: dict) -> str:
    detail = f"{p['weight']:.0%} invested" if "weight" in p else f"{p['prob']:.1%} ML"
    return f"`{p['date']}` — {p['signal']} (`{detail}`  ${p['close']})"


def latest_message() -> str:
    hist = history()
    if not hist:
        return "No signal yet — the first one arrives at 9pm SGT on the next weekday."
    return f"📊 *Latest SPY Signal*\n\n{_describe(hist[-1])}\n\n_Next update: weekdays 9pm SGT_"


def history_message() -> str:
    hist = history()
    if not hist:
        return "No signals recorded yet."
    return "📅 *Last Signals:*\n\n" + "\n".join(_describe(p) for p in hist[-5:][::-1])


def accuracy_message() -> str:
    hist = history()
    if len(hist) < 2:
        return "Not enough data yet — need at least 2 signals."
    pairs = list(zip(hist, hist[1:]))
    right = sum(_invested(cur) == (nxt["close"] > cur["close"]) for cur, nxt in pairs)
    return (f"🎯 *Bot Accuracy*\n\nRight {right}/{len(pairs)} times (invested before a rise, or out before a fall)\n"
            f"Accuracy: `{right / len(pairs) * 100:.1f}%`")


# ---------- trading ----------
def rebalance(weight: float, last_close: float) -> str:
    """Moves the SPY position to `weight` of the SPY allocation, cash only (no margin)."""
    broker.cancel_open_orders(TICKER)  # replace any order still waiting for the open
    acct, pos = broker.account(), broker.position(TICKER)
    base = float(acct["equity"]) * SPY_ALLOCATION
    qty = float(pos["qty"]) if pos else 0.0
    price = float(pos["current_price"]) if pos else last_close
    current, target = qty * price, weight * base
    if abs(target - current) < REBALANCE_BAND * base:
        return f"✅ No trade: holding {current / base:.0%}, target {weight:.0%}."
    if target < 1:
        broker.close_position(TICKER)
        return f"📉 SELL all {TICKER} ({qty:g} sh) → target 0%."
    if target < current:
        sell = min(qty, round((current - target) / price, 4))
        broker.market_order(TICKER, "sell", qty=sell)
        return f"📉 SELL {sell:g} {TICKER} → target {weight:.0%} (${target:,.0f})."
    buy = round(min(target - current, broker.free_cash() * 0.99), 2)
    if buy < 1:
        return "⚠️ Wanted to buy but there is no free cash."
    broker.market_order(TICKER, "buy", notional=buy)
    return f"📈 BUY ${buy:,.2f} of {TICKER} → target {weight:.0%} (${target:,.0f})."


def paper_status_message() -> str:
    if not broker.ENABLED:
        return "Paper trading is disabled."
    acct, pos = broker.account(), broker.position(TICKER)
    equity = float(acct["equity"])
    lines = ["📊 *Alpaca Paper Account*", "",
             f"Equity: `${equity:,.2f}`", f"Cash: `${float(acct['cash']):,.2f}`"]
    if pos:
        value = float(pos["market_value"])
        lines += ["", f"📌 *Position: {TICKER}*", f"Qty: `{float(pos['qty']):.4f}`",
                  f"Value: `${value:,.2f}` (`{value / equity:.0%}` of equity)",
                  f"P&L: `${float(pos['unrealized_pl']):+,.2f}` (`{float(pos['unrealized_plpc']) * 100:+.2f}%`)"]
    else:
        lines.append("\nPosition: *FLAT*")
    return "\n".join(lines)


# ---------- daily run ----------
def run_daily(force_notify: bool = False) -> dict:
    s = compute()
    print(f"[SPY] {s['signal_label']} weight={s['weight']:.0%} trend={s['trend']:.0%} dip={s['dip']} "
          f"vol={s['vol']:.1%} rsi2={s['rsi2']:.0f} close={s['close']} ({s['data_date']})")
    prev = store.get("last_signal_state", {"signal": None, "date": None})
    changed = prev["signal"] != s["signal_label"]
    _save(s)
    if changed or force_notify or prev["date"] != s["date"]:
        tg.broadcast(format_message(s, changed))
    if broker.ENABLED:
        try:
            status = rebalance(s["weight"], s["close"])
        except Exception as e:
            status = f"❌ Rebalance failed: {e}"
        print(f"[SPY] {status}")
        if not status.startswith("✅"):
            tg.send_admin(f"🤖 *Paper Trading Update*\n{status}")
    return s


def backtest_report(dev=("2012-01-01", "2024-11-13"), hold=("2024-11-14", None)):
    df = load_prices()
    w = target_weights(df)["weight"]
    fill = df["open"].shift(-2) / df["open"].shift(-1) - 1  # decided at close t, held from open t+1
    for name, pos in (("Strategy", w), ("Buy & hold", pd.Series(1.0, w.index))):
        r = pos * fill - 0.0002 * pos.diff().abs().fillna(pos)
        for label, (a, b) in (("2012-2024", dev), ("holdout", hold)):
            x = r.loc[a:b].dropna()
            eq = (1 + x).cumprod()
            print(f"{name:10s} {label:9s} Sharpe {x.mean() / x.std() * np.sqrt(252):.3f}  "
                  f"CAGR {eq.iloc[-1] ** (252 / len(x)) - 1:6.1%}  max DD {(eq / eq.cummax() - 1).min():6.1%}  "
                  f"avg exposure {pos.loc[a:b].mean():.0%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPY trend + dip strategy")
    parser.add_argument("--backtest", action="store_true", help="print backtest only")
    parser.add_argument("--force", action="store_true", help="run now and broadcast")
    args = parser.parse_args()
    backtest_report() if args.backtest else run_daily(force_notify=args.force)
