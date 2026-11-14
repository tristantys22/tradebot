"""SPY 5-day direction model (XGBoost): daily signal, Telegram alert, paper trade.

Manual run:  python spy_model.py --force      (broadcast now)
             python spy_model.py --backtest   (print backtest report, no side effects)
"""
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

import broker
import store
import tg

TICKER = "SPY"
START_DATE = "2015-01-01"
TRAIN_SIZE = 0.7
PROB_THRESHOLD = 0.50
POSITION_PCT = 0.95          # share of cash used on a BUY
HISTORY_LIMIT = 1000         # predictions kept in Supabase
INITIAL_CAPITAL = 10_000
TRANSACTION_COST = 0.0005

FEATURE_COLS = [
    "ret_5d", "ret_10d", "ret_20d", "ret_1d_lag1", "ret_1d_lag2", "ret_1d_lag3",
    "price_vs_sma20", "sma10_vs_sma50", "above_sma200",
    "bb_width", "bb_zscore", "bb_pos", "bb_break_upper", "bb_break_lower",
    "rsi_14", "rsi_5",
    "atr_pct", "vol_20d", "vol_5d",
    "vol_chg_5d", "vol_ratio_20", "vol_spike",
    "vix", "vix_chg_5d", "vix_vs_ma", "vix_high", "vix_extreme",
]


# ---------- data ----------
def _download(ticker: str, end=None) -> pd.DataFrame:
    df = yf.download(ticker, start=START_DATE, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower)


def load_data(end=None) -> pd.DataFrame:
    df = _download(TICKER, end)[["open", "high", "low", "close", "volume"]]
    try:
        vix = _download("^VIX", end)[["close"]].rename(columns={"close": "vix"})
    except Exception as e:
        print(f"[SPY] ⚠️ VIX download failed: {e}")
        vix = None
    return add_features(df, vix)


# ---------- features ----------
def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def add_features(df: pd.DataFrame, vix: pd.DataFrame = None) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["ret_1d"] = close.pct_change()
    for n in (5, 10, 20):
        out[f"ret_{n}d"] = close.pct_change(n)
    for lag in (1, 2, 3):
        out[f"ret_1d_lag{lag}"] = out["ret_1d"].shift(lag)

    sma = {n: close.rolling(n).mean() for n in (10, 20, 50, 200)}
    out["price_vs_sma20"] = close / sma[20] - 1
    out["sma10_vs_sma50"] = sma[10] / sma[50] - 1
    out["above_sma200"] = (close > sma[200]).astype(int)

    bb_std = close.rolling(20).std()
    bb_upper, bb_lower = sma[20] + 2 * bb_std, sma[20] - 2 * bb_std
    out["bb_width"] = (bb_upper - bb_lower) / sma[20]
    out["bb_zscore"] = (close - sma[20]) / bb_std
    out["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower)
    out["bb_break_upper"] = (close > bb_upper).astype(int)
    out["bb_break_lower"] = (close < bb_lower).astype(int)

    out["rsi_14"] = _rsi(close, 14)
    out["rsi_5"] = _rsi(close, 5)

    prev_close = close.shift(1)
    true_range = pd.concat([out["high"] - out["low"], (out["high"] - prev_close).abs(),
                            (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr_pct"] = true_range.rolling(14).mean() / close
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["vol_5d"] = out["ret_1d"].rolling(5).std()

    out["vol_chg_5d"] = out["volume"].pct_change(5)
    out["vol_ratio_20"] = out["volume"] / out["volume"].rolling(20).mean()
    out["vol_spike"] = (out["vol_ratio_20"] > 2.0).astype(int)

    if vix is not None:
        out = out.join(vix, how="left")
        out["vix"] = out["vix"].ffill()
        out["vix_chg_5d"] = out["vix"].pct_change(5)
        out["vix_vs_ma"] = out["vix"] / out["vix"].rolling(10).mean() - 1
        out["vix_high"] = (out["vix"] > 25).astype(int)
        out["vix_extreme"] = (out["vix"] > 35).astype(int)
    else:
        for col in ("vix", "vix_chg_5d", "vix_vs_ma", "vix_high", "vix_extreme"):
            out[col] = np.nan

    out["future_ret_5d"] = close.pct_change(5).shift(-5)
    out["target"] = (out["future_ret_5d"] > 0).astype(int)
    return out


# ---------- model ----------
def train(df: pd.DataFrame):
    """Returns (model, feature_cols, test_df). Trains on the first TRAIN_SIZE of labelled rows."""
    import xgboost as xgb  # lazy: keeps ~100 MB out of memory between daily runs

    labelled = df.dropna(subset=["target", "future_ret_5d"])
    split = int(len(labelled) * TRAIN_SIZE)
    train_df, test_df = labelled.iloc[:split], labelled.iloc[split:]
    cols = [c for c in FEATURE_COLS if train_df[c].notna().any()]
    n_pos = (train_df["target"] == 1).sum()
    params = {
        "objective": "binary:logistic", "eta": 0.03, "max_depth": 5, "seed": 42, "verbosity": 0,
        "scale_pos_weight": (train_df["target"] == 0).sum() / n_pos if n_pos else 1.0,
    }
    data = xgb.QuantileDMatrix(train_df[cols], label=train_df["target"])  # same binning as XGBClassifier
    model = xgb.train(params, data, num_boost_round=500)
    return model, cols, test_df


def predict(model, X: pd.DataFrame):
    """Probability of a positive 5-day return for each row."""
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X.astype(float)))


def make_signal(model, cols: list, df: pd.DataFrame) -> dict:
    latest = df.dropna(subset=cols).iloc[-1]
    prob = float(predict(model, latest[cols].to_frame().T)[0])
    vix = latest.get("vix", np.nan)
    blocked = vix > 35  # panic regime: override the model and stay out
    if blocked:
        signal, label, prob = 0, "STAY OUT (Regime Filter) 🚨", min(prob, 0.45)
    else:
        signal = int(prob >= PROB_THRESHOLD)
        label = "BUY 📈" if signal else "STAY OUT 🔴"
    return {
        "date": str(date.today()),
        "signal": signal,
        "signal_label": label,
        "prob": round(prob, 4),
        "rsi": round(float(latest["rsi_14"]), 1),
        "bb_zscore": round(float(latest["bb_zscore"]), 2),
        "price_vs_sma20": round(float(latest["price_vs_sma20"]) * 100, 2),
        "close": round(float(df["close"].iloc[-1]), 2),
        "vix": "N/A" if np.isnan(vix) else round(float(vix), 1),
        "regime_blocked": bool(blocked),
    }


def format_message(sig: dict, changed: bool) -> str:
    lines = [
        "🔄 *SIGNAL CHANGED*\n" if changed else "",
        f"📊 *SPY 5-Day Signal — {sig['date']}*",
        "",
        f"Signal:      *{sig['signal_label']}*",
        f"Confidence:  `{sig['prob']:.1%}`",
        "",
        "📌 *Key Indicators*",
        f"Close Price:     `${sig['close']}`",
        f"RSI (14):        `{sig['rsi']}`",
        f"BB Z-Score:      `{sig['bb_zscore']}`",
        f"vs SMA-20:       `{sig['price_vs_sma20']:+.2f}%`",
        f"VIX:             `{sig['vix']}`",
        "\n⚠️ _Regime filter active — model overridden_" if sig["regime_blocked"] else "",
        "",
        f"_Model: XGBoost | Target: 5-day return | Threshold: {PROB_THRESHOLD:.0%}_",
    ]
    return "\n".join(lines)


# ---------- state ----------
def history() -> list:
    return store.get("prediction_history", [])


def _save(sig: dict):
    store.set("last_signal_state", {"signal": sig["signal"], "date": sig["date"], "prob": sig["prob"]})
    hist = [p for p in history() if p["date"] != sig["date"]]  # a re-run replaces today's entry
    hist.append({k: sig[k] for k in ("date", "prob", "close")} | {"signal": sig["signal_label"]})
    store.set("prediction_history", hist[-HISTORY_LIMIT:])


def accuracy_message() -> str:
    hist = history()
    if len(hist) < 2:
        return "Not enough data yet — need at least 2 predictions."
    pairs = list(zip(hist, hist[1:]))
    correct = sum((cur["signal"] == "BUY 📈") == (nxt["close"] > cur["close"]) for cur, nxt in pairs)
    return (f"🎯 *Bot Accuracy*\n\nCorrect: {correct}/{len(pairs)} predictions\n"
            f"Accuracy: `{correct / len(pairs) * 100:.1f}%`")


def latest_message() -> str:
    hist = history()
    if not hist:
        return "No signal yet — the first one arrives at 9pm SGT on the next weekday."
    p = hist[-1]
    return (f"📊 *Latest SPY Signal — {p['date']}*\n\nSignal: *{p['signal']}*\nConfidence: `{p['prob']:.1%}`\n"
            f"SPY close: `${p['close']}`\n\n_Next update: weekdays 9pm SGT_")


def history_message() -> str:
    hist = history()
    if not hist:
        return "No predictions recorded yet."
    rows = [f"`{p['date']}` — {p['signal']} (`{p['prob']:.1%}`  ${p['close']})" for p in hist[-5:][::-1]]
    return "📅 *Last Predictions:*\n\n" + "\n".join(rows)


# ---------- trading ----------
def paper_trade(signal: int) -> str:
    pos = broker.position(TICKER)
    if signal and not pos:
        acct = broker.account()
        # cash, not buying_power: buying_power includes margin and would lever the account up ~4x
        notional = round(min(float(acct["cash"]), float(acct["buying_power"])) * POSITION_PCT, 2)
        if notional < 1:
            return "⚠️ Not enough cash."
        broker.market_order(TICKER, "buy", notional=notional)
        return f"📈 Paper BUY {TICKER} submitted (${notional:,.2f})."
    if not signal and pos:
        broker.close_position(TICKER)
        return f"📉 Paper SELL {TICKER} submitted."
    return f"✅ Already {'long' if pos else 'flat'} {TICKER}, no action."


def paper_status_message() -> str:
    if not broker.ENABLED:
        return "Paper trading is disabled."
    acct, pos = broker.account(), broker.position(TICKER)
    lines = ["📊 *Alpaca Paper Account*", "",
             f"Equity: `${float(acct['equity']):,.2f}`", f"Cash: `${float(acct['cash']):,.2f}`"]
    if pos:
        lines += ["", f"📌 *Position: {TICKER}*", f"Qty: `{float(pos['qty']):.4f}`",
                  f"Value: `${float(pos['market_value']):,.2f}`",
                  f"P&L: `${float(pos['unrealized_pl']):+,.2f}` (`{float(pos['unrealized_plpc']) * 100:+.2f}%`)"]
    else:
        lines.append("\nPosition: *FLAT*")
    return "\n".join(lines)


# ---------- daily run ----------
def run_daily(force_notify: bool = False) -> dict:
    print(f"[SPY] Run started {datetime.now():%Y-%m-%d %H:%M:%S}")
    df = load_data()
    model, cols, _ = train(df)
    sig = make_signal(model, cols, df)
    print(f"[SPY] {sig['signal_label']} prob={sig['prob']:.1%} close={sig['close']} vix={sig['vix']}")

    prev = store.get("last_signal_state", {"signal": None, "date": None})
    changed = prev["signal"] != sig["signal"]
    _save(sig)

    if changed or force_notify or prev["date"] != sig["date"]:
        tg.broadcast(format_message(sig, changed))
    if changed and broker.ENABLED:
        try:
            status = paper_trade(sig["signal"])
        except Exception as e:
            status = f"❌ Paper trade failed: {e}"
        print(f"[SPY] {status}")
        tg.send_admin(f"🤖 *Paper Trading Update*\n{status}")
    return sig


def backtest_report():
    df = load_data()
    model, cols, test = train(df)
    test = test.copy()
    test["position"] = (predict(model, test[cols]) >= PROB_THRESHOLD).astype(int)
    print(pd.crosstab(test["target"], test["position"], rownames=["actual"], colnames=["predicted"]))
    print(pd.Series(model.get_score(importance_type="gain")).sort_values(ascending=False).head(15))

    held = test["position"].shift(1).fillna(0)
    strat = held * test["future_ret_5d"] - held.diff().abs().fillna(held) * TRANSACTION_COST
    for name, rets in (("Strategy", strat), ("Buy & Hold", test["future_ret_5d"])):
        equity = (1 + rets).cumprod() * INITIAL_CAPITAL
        years = len(rets) / 252
        print(f"\n=== {name} ===\n"
              f"Annualized return: {(1 + rets).prod() ** (1 / years) - 1:.2%}\n"
              f"Sharpe ratio:      {np.sqrt(252) * rets.mean() / rets.std():.2f}\n"
              f"Max drawdown:      {(equity / equity.cummax() - 1).min():.2%}\n"
              f"Final equity:      ${equity.iloc[-1]:,.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPY ML signal bot")
    parser.add_argument("--backtest", action="store_true", help="print backtest report only")
    parser.add_argument("--force", action="store_true", help="run now and broadcast")
    args = parser.parse_args()
    backtest_report() if args.backtest else run_daily(force_notify=args.force)
