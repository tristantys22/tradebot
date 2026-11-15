"""NUAI daily swing research, long only, post-SPAC data only (Dec 2024+).
Decide at close t, fill at open t+1, 10 bp per side. Rules fixed up front; choose on dev, report holdout."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

COST = 0.0010
DATA_START, DEV = "2024-12-01", ("2025-01-01", "2025-11-30")
HOLD = ("2025-12-01", None)


def dl(t):
    d = yf.download(t, start="2024-06-01", auto_adjust=True, progress=False)
    d.columns = d.columns.get_level_values(0)
    return d.rename(columns=str.lower)


raw = dl("NUAI")
df = raw.loc[DATA_START:].copy()
idx = df.index
o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
R = o.shift(-2) / o.shift(-1) - 1
ret = c.pct_change()
spy, iwm = (dl(t)["close"].reindex(idx).ffill() for t in ("SPY", "IWM"))
print(f"post-SPAC data {idx[0].date()}..{idx[-1].date()} ({len(idx)} days)")


def rsi(s, n):
    d = s.diff()
    g, lo = d.clip(lower=0).rolling(n).mean(), (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / lo.replace(0, np.nan))


def hold(enter, leave):
    out, held = np.zeros(len(enter)), 0
    for i, (e, x) in enumerate(zip(enter.fillna(False).to_numpy(), leave.fillna(False).to_numpy())):
        if held and x:
            held = 0
        if not held and e:
            held = 1
        out[i] = held
    return pd.Series(out, idx)


ROWS = []


def stats(r, pos, a, b):
    r, p = r.loc[a:b].dropna(), pos.loc[a:b]
    eq = (1 + r).cumprod()
    entries = int((p.diff() > 0).sum())
    # per-trade returns
    trades, cur = [], None
    for day, (pp, rr) in enumerate(zip(p.to_numpy(), r.reindex(p.index).fillna(0).to_numpy())):
        if pp and cur is None:
            cur = 1.0
        if cur is not None:
            cur *= 1 + rr
            if not pp:
                trades.append(cur - 1); cur = None
    if cur is not None:
        trades.append(cur - 1)
    return dict(sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, total=eq.iloc[-1] - 1,
                mdd=(eq / eq.cummax() - 1).min(), expo=p.mean(), trades=entries,
                win=np.mean([t > 0 for t in trades]) if trades else np.nan)


def record(name, pos, group):
    pos = pos.reindex(idx).fillna(0).astype(float)
    r = pos * R - COST * pos.diff().abs().fillna(pos)
    d, hd = stats(r, pos, *DEV), stats(r, pos, *HOLD)
    ROWS.append(dict(group=group, name=name, **{f"dev_{k}": x for k, x in d.items()}, **{f"hold_{k}": x for k, x in hd.items()}))


sma = lambda n: c.rolling(n).mean()
ema = lambda n: c.ewm(span=n, adjust=False).mean()
r2 = rsi(c, 2)
one = pd.Series(1.0, idx)

record("Buy & hold NUAI", one, "baseline")
record("Close > SMA20", (c > sma(20)).astype(float), "trend")
record("Close > SMA50", (c > sma(50)).astype(float), "trend")
record("EMA10 > EMA30", (ema(10) > ema(30)).astype(float), "trend")
record("Donchian 20-day high breakout, exit 10-day low", hold(c > h.rolling(20).max().shift(1), c < l.rolling(10).min().shift(1)), "trend")
record("Donchian 20/10 + only if > SMA50", hold((c > h.rolling(20).max().shift(1)) & (c > sma(50)), c < l.rolling(10).min().shift(1)), "trend")
record("RSI(2)<10, exit > SMA5", hold(r2 < 10, c > sma(5)), "dip")
record("RSI(2)<10 & > SMA50, exit > SMA5", hold((r2 < 10) & (c > sma(50)), c > sma(5)), "dip")
mid, sd = sma(20), c.rolling(20).std()
record("Bollinger 20/2 lower band, exit at mid", hold(c < mid - 2 * sd, c >= mid), "dip")
down3 = (ret < 0) & (ret.shift(1) < 0) & (ret.shift(2) < 0)
record("3 down days, exit first up close", hold(down3, ret > 0), "dip")

# ---------- ML: walk-forward XGBoost, monthly retrain, 3-day target ----------
import xgboost as xgb
X = pd.DataFrame({
    **{f"r{k}": c.pct_change(k) for k in (1, 3, 5, 10, 20)},
    "rsi2": r2, "rsi14": rsi(c, 14),
    **{f"gap_sma{k}": c / sma(k) - 1 for k in (5, 10, 20, 50)},
    "vol5": ret.rolling(5).std(), "vol20": ret.rolling(20).std(),
    "vol_ratio": ret.rolling(5).std() / ret.rolling(20).std(),
    "volu_ratio": v / v.rolling(20).mean(), "range": (h - l) / c, "gap": o / c.shift(1) - 1,
    "close_loc": (c - l) / (h - l).replace(0, np.nan),
    "spy_r5": spy.pct_change(5), "iwm_r5": iwm.pct_change(5),
}, index=idx).astype(float)
HZ = 3
fwd = c.shift(-HZ) / c - 1
y = (fwd > 0).astype(float).where(fwd.notna())
P = dict(objective="binary:logistic", eta=0.05, max_depth=2, min_child_weight=10, subsample=0.8,
         colsample_bytree=0.8, reg_lambda=5, seed=42, verbosity=0)
prob = pd.Series(np.nan, idx)
first = idx.searchsorted(pd.Timestamp(DEV[0]))
first = max(first, 60)
n = np.arange(len(idx))
for start in range(first, len(idx), 21):
    tr = (n <= start - HZ) & y.notna().to_numpy() & X["gap_sma50"].notna().to_numpy()
    if tr.sum() < 40:
        continue
    m = xgb.train(P, xgb.DMatrix(X[tr], label=y[tr]), 150)
    prob.iloc[start:start + 21] = m.predict(xgb.DMatrix(X.iloc[start:start + 21]))
known = prob.notna() & fwd.notna()
print(f"ML: {int(known.sum())} predictions, 3-day hit rate {((prob > .5) == (fwd > 0))[known].mean():.1%}, "
      f"base rate up {(fwd > 0)[known].mean():.1%}, mean training rows at end {int(tr.sum())}")
record("ML XGBoost p>0.55 (3-day target)", (prob > 0.55).astype(float), "ml")
record("ML XGBoost p>0.50 (3-day target)", (prob > 0.50).astype(float), "ml")

out = pd.DataFrame(ROWS)
pd.set_option("display.width", 250)
cols = ["group", "name", "dev_sharpe", "dev_total", "dev_mdd", "dev_expo", "dev_trades", "dev_win",
        "hold_sharpe", "hold_total", "hold_mdd", "hold_expo", "hold_trades", "hold_win"]
print(out.sort_values("dev_sharpe", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
