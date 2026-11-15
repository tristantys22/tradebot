"""NUAI daily swing model (long only): XGBoost predicts whether NUAI is higher in 3 trading days.

Runs before the US open with SPY. If the model's probability is above 50%, hold ALLOCATION of equity
in NUAI, otherwise stay in cash. Retrains every run on post-SPAC data only (NUAI was a ~$10 SPAC
until Nov 2024). Walk-forward backtest (next-open fills, 10 bp/side): see backtest_report().
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import broker
import store
import tg

SYMBOL = "NUAI"
DATA_START = "2024-12-01"
HORIZON = 3                  # trading days ahead the model predicts
THRESHOLD = 0.50
SEEDS = range(5)             # averaged to damp seed-to-seed noise
ALLOCATION = 0.20            # share of account equity
REBALANCE_BAND = 0.05        # of the allocation
TRADE_LOG_LIMIT = 500
PARAMS = dict(objective="binary:logistic", eta=0.05, max_depth=2, min_child_weight=10, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=5, verbosity=0)
ROUNDS = 150


# ---------- data / features ----------
def _download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2024-06-01", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    now_ny = pd.Timestamp.now(tz="America/New_York")
    if df.index[-1].date() == now_ny.date() and now_ny.hour < 16:
        df = df.iloc[:-1]  # today's bar is still forming
    return df


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def load() -> pd.DataFrame:
    """Features plus `close`/`open` and the 3-day-ahead label, post-SPAC rows only."""
    n = _download(SYMBOL).loc[DATA_START:]
    o, h, l, c, v = (n[k] for k in ("open", "high", "low", "close", "volume"))
    ret = c.pct_change()
    sma = lambda k: c.rolling(k).mean()
    spy, iwm = (_download(t)["close"].reindex(n.index).ffill() for t in ("SPY", "IWM"))
    X = pd.DataFrame({
        **{f"r{k}": c.pct_change(k) for k in (1, 3, 5, 10, 20)},
        "rsi2": _rsi(c, 2), "rsi14": _rsi(c, 14),
        **{f"gap_sma{k}": c / sma(k) - 1 for k in (5, 10, 20, 50)},
        "vol5": ret.rolling(5).std(), "vol20": ret.rolling(20).std(),
        "vol_ratio": ret.rolling(5).std() / ret.rolling(20).std(),
        "volu_ratio": v / v.rolling(20).mean(), "range": (h - l) / c, "gap": o / c.shift(1) - 1,
        "close_loc": (c - l) / (h - l).replace(0, np.nan),
        "spy_r5": spy.pct_change(5), "iwm_r5": iwm.pct_change(5),
    }, index=n.index).astype(float)
    fwd = c.shift(-HORIZON) / c - 1
    X["label"] = (fwd > 0).astype(float).where(fwd.notna())
    X["close"], X["open"] = c, o
    return X



def predict_last(X: pd.DataFrame, upto: int) -> float:
    """Train on rows whose label was known by row `upto`, return P(up) for row `upto`."""
    import xgboost as xgb
    feats = [k for k in X.columns if k not in ("label", "close", "open")]
    train = X.iloc[:upto - HORIZON + 1]
    train = train[train["label"].notna() & train["gap_sma50"].notna()]
    if len(train) < 40:
        return float("nan")
    data = xgb.DMatrix(train[feats], label=train["label"])
    row = xgb.DMatrix(X[feats].iloc[[upto]])
    return float(np.mean([xgb.train(PARAMS | {"seed": s}, data, ROUNDS).predict(row)[0] for s in SEEDS]))


# ---------- trade log ----------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def trades() -> list:
    return store.get("nuai_trades", [])


def _log(trade: dict):
    store.set("nuai_trades", (trades() + [trade | {"time": _now()}])[-TRADE_LOG_LIMIT:])


def _stats(sells: list) -> str:
    if not sells:
        return "No closed trades."
    wins = sum(t["pnl"] > 0 for t in sells)
    return (f"Trades: `{len(sells)}` | Wins: `{wins}` | Win rate: `{wins / len(sells):.0%}`\n"
            f"Realized P&L (est.): `${sum(t['pnl'] for t in sells):+,.2f}`")


# ---------- trading ----------
def rebalance(prob: float, last_close: float) -> str:
    broker.cancel_open_orders(SYMBOL)
    acct, pos = broker.account(), broker.position(SYMBOL)
    held = int(float(pos["qty"])) if pos else 0
    target = int(float(acct["equity"]) * ALLOCATION // last_close) if prob > THRESHOLD else 0
    if target == 0 and held:
        broker.close_position(SYMBOL)
        pnl = float(pos["unrealized_pl"])
        _log({"side": "SELL", "price": float(pos["current_price"]), "qty": held, "pnl": round(pnl, 2),
              "pnl_pct": round(float(pos["unrealized_plpc"]) * 100, 2), "prob": round(prob, 3)})
        return (f"{'✅' if pnl >= 0 else '❌'} *NUAI SELL* all {held} sh (P(up) `{prob:.0%}`)\n"
                f"Entry `${float(pos['avg_entry_price']):.2f}` → last `${float(pos['current_price']):.2f}`, "
                f"P&L ≈ `${pnl:+,.2f}` (`{float(pos['unrealized_plpc']) * 100:+.1f}%`)")
    if abs(target - held) * last_close < REBALANCE_BAND * ALLOCATION * float(acct["equity"]):
        return ""
    if target > held:
        qty = min(target - held, int(broker.free_cash() * 0.99 // last_close))
        if qty < 1:
            return "⚠️ NUAI wanted to buy but there is no free cash."
        broker.market_order(SYMBOL, "buy", qty=qty)
        _log({"side": "BUY", "price": last_close, "qty": qty, "prob": round(prob, 3)})
        return f"🟢 *NUAI BUY* {qty} sh at the open (last `${last_close:.2f}`, P(up) `{prob:.0%}`)"
    broker.market_order(SYMBOL, "sell", qty=held - target)
    return f"🟠 *NUAI TRIM* {held - target} sh to stay near {ALLOCATION:.0%} of equity"


def run_daily():
    X = load()
    prob, close = predict_last(X, len(X) - 1), float(X["close"].iloc[-1])
    store.set("nuai_last_check", {"time": _now(), "date": str(X.index[-1].date()), "prob": round(prob, 4),
                                  "close": round(close, 4)})
    print(f"[NUAI] P(up {HORIZON}d)={prob:.1%} close={close:.2f} ({X.index[-1].date()})")
    if broker.ENABLED and np.isfinite(prob):
        msg = rebalance(prob, close)
        if msg:
            tg.send_admin(msg)


def status_message() -> str:
    last, pos = store.get("nuai_last_check", {}), broker.position(SYMBOL)
    lines = ["📈 *NUAI Swing Model* (daily, long only)", ""]
    if "prob" in last:
        lines += [f"P(higher in {HORIZON} days): `{last['prob']:.0%}` → {'*IN*' if last['prob'] > THRESHOLD else '*OUT*'}",
                  f"Close ({last['date']}): `${last['close']:.2f}`", ""]
    lines.append(f"Position: `{float(pos['qty']):g}` sh @ `${float(pos['avg_entry_price']):.2f}` "
                 f"(`{float(pos['unrealized_plpc']) * 100:+.1f}%`)" if pos else "Position: *FLAT*")
    lines += ["", "*All-time*", _stats([t for t in trades() if t["side"] == "SELL" and "prob" in t])]
    return "\n".join(lines)


def backtest_report(dev=("2025-01-01", "2025-11-30"), hold=("2025-12-01", None), cost=0.0010):
    X = load()
    prob = pd.Series(np.nan, X.index)
    for i in range(50, len(X)):
        prob.iloc[i] = predict_last(X, i)
    pos = (prob > THRESHOLD).astype(float)
    fill = X["open"].shift(-2) / X["open"].shift(-1) - 1
    for name, p in (("Model", pos), ("Buy & hold", pd.Series(1.0, X.index))):
        r = p * fill - cost * p.diff().abs().fillna(p)
        for label, (a, b) in (("dev", dev), ("holdout", hold)):
            x, pp = r.loc[a:b].dropna(), p.loc[a:b]
            eq = (1 + x).cumprod()
            print(f"{name:10s} {label:8s} Sharpe {x.mean() / x.std() * np.sqrt(252):5.2f}  total {eq.iloc[-1] - 1:+7.0%}  "
                  f"max DD {(eq / eq.cummax() - 1).min():6.0%}  exposure {pp.mean():4.0%}  entries {int((pp.diff() > 0).sum())}")


if __name__ == "__main__":
    backtest_report()
