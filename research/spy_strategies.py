"""Honest SPY strategy comparison. Decision at close t, filled at open t+1, 2 bp per side.
Dev window 2012-01-01..2024-11-13 (used to choose); holdout 2024-11-14..today (reported only)."""
import io, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests, yfinance as yf, xgboost as xgb
sys.path.insert(0, "/app")
import spy_model as live

WF_START, DEV_END, HOLD_START = "2012-01-01", "2024-11-13", "2024-11-14"
COST = 0.0002


def dl(t, start="2003-01-01"):
    for _ in range(3):
        try:
            df = yf.download(t, start=start, auto_adjust=True, progress=False)
            if len(df):
                break
        except Exception as e:
            print("download retry", t, e)
        time.sleep(3)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower)


spy = dl("SPY")[["open", "high", "low", "close", "volume"]]
idx = spy.index
o, h, l, c = spy["open"], spy["high"], spy["low"], spy["close"]
R = o.shift(-2) / o.shift(-1) - 1  # earned by a position decided at close t
ret1 = c.pct_change()


def align(s):
    s = s[~s.index.duplicated()]
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


ext = {t: align(dl(t)["close"]) for t in ["^VIX", "^VIX3M", "TLT", "HYG", "GLD", "UUP", "QQQ", "IWM", "^TNX", "^IRX"]}
t10y2y = ext["^TNX"] - ext["^IRX"]  # 10y minus 3m Treasury yield
print("data", idx[0].date(), idx[-1].date(), len(idx), "t10y2y non-null", int(t10y2y.notna().sum()))

RESULTS = []


def stats(r, pos, start, end):
    r = r.loc[start:end].dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    p = pos.loc[start:end]
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * np.sqrt(252),
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                mdd=(eq / eq.cummax() - 1).min(), expo=p.abs().mean(), turns=p.diff().abs().sum() / yrs)


def record(name, pos=None, returns=None, group=""):
    if returns is None:
        pos = pos.reindex(idx).astype(float).fillna(0).clip(-1, 1)
        returns = pos * R - COST * pos.diff().abs().fillna(pos.abs())
    else:
        pos = pos.reindex(idx).astype(float).fillna(0) if pos is not None else (returns != 0).astype(float)
    d, hd = stats(returns, pos, WF_START, DEV_END), stats(returns, pos, HOLD_START, idx[-1])
    RESULTS.append(dict(group=group, name=name, **{f"dev_{k}": v for k, v in d.items()},
                        **{f"hold_{k}": v for k, v in hd.items()}))
    print(f"  {name:38s} dev sharpe {d['sharpe']:5.2f} cagr {d['cagr']:6.1%} mdd {d['mdd']:6.1%} | "
          f"hold sharpe {hd['sharpe']:5.2f} cagr {hd['cagr']:6.1%} mdd {hd['mdd']:6.1%}", flush=True)


sma = lambda s, n: s.rolling(n).mean()
vt = (0.15 / (ret1.rolling(20).std() * np.sqrt(252))).clip(upper=1.0)


def hold_state(el, xl, es=None, xs=None):
    el, xl = el.fillna(False).values, xl.fillna(False).values
    es = es.fillna(False).values if es is not None else None
    xs = xs.fillna(False).values if xs is not None else None
    pos, out = 0, np.zeros(len(el))
    for i in range(len(el)):
        if pos == 1 and xl[i]:
            pos = 0
        elif pos == -1 and xs[i]:
            pos = 0
        if pos == 0:
            if el[i]:
                pos = 1
            elif es is not None and es[i]:
                pos = -1
        out[i] = pos
    return pd.Series(out, idx)


# ---------------- 1. baselines + classic rules ----------------
print("\n# baselines and classic rules")
one = pd.Series(1.0, idx)
record("Buy & hold SPY", one, group="baseline")
record("Vol target 15% (long only)", vt, group="baseline")
record("SMA200 trend", (c > sma(c, 200)).astype(float), group="baseline")
record("SMA200 trend + vol target", (c > sma(c, 200)) * vt, group="baseline")

ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
macd_up = ema(c, 12) > ema(c, 26)
record("MACD 12/26 (long)", macd_up.astype(float), group="repo")
record("MACD 12/26 (long/short)", np.where(macd_up, 1.0, -1.0) * one, group="repo")

rsi2, rsi14 = live._rsi(c, 2), live._rsi(c, 14)
record("RSI(2) Connors (long, >SMA200)", hold_state((rsi2 < 10) & (c > sma(c, 200)), c > sma(c, 5)), group="mean-rev")
record("RSI(2) Connors long + short", hold_state((rsi2 < 10) & (c > sma(c, 200)), c > sma(c, 5),
                                                 (rsi2 > 90) & (c < sma(c, 200)), c < sma(c, 5)), group="mean-rev")
record("RSI 30/70 (long)", hold_state(rsi14 < 30, rsi14 > 70), group="repo")
record("RSI 30/70 (long/short)", hold_state(rsi14 < 30, rsi14 > 70, rsi14 > 70, rsi14 < 30), group="repo")

mid, sd = sma(c, 20), c.rolling(20).std()
record("Bollinger mean-rev (long)", hold_state(c < mid - 2 * sd, c >= mid), group="repo")
record("Bollinger mean-rev (long/short)", hold_state(c < mid - 2 * sd, c >= mid, c > mid + 2 * sd, c <= mid), group="repo")


def psar(af0=0.02, afmax=0.2):
    H, L = h.values, l.values
    sar, bull, af, ep = np.zeros(len(H)), True, af0, H[0]
    sar[0] = L[0]
    for i in range(1, len(H)):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if bull:
            s = min(s, L[i - 1], L[max(i - 2, 0)])
            if L[i] < s:
                bull, s, ep, af = False, ep, L[i], af0
            elif H[i] > ep:
                ep, af = H[i], min(af + af0, afmax)
        else:
            s = max(s, H[i - 1], H[max(i - 2, 0)])
            if H[i] > s:
                bull, s, ep, af = True, ep, H[i], af0
            elif L[i] < ep:
                ep, af = L[i], min(af + af0, afmax)
        sar[i] = s
    return pd.Series(sar, idx)


sar_up = c > psar()
record("Parabolic SAR (long)", sar_up.astype(float), group="repo")
record("Parabolic SAR (long/short)", np.where(sar_up, 1.0, -1.0) * one, group="repo")

ha_c = (o + h + l + c) / 4
ha_o = np.zeros(len(c))
ha_o[0] = (o.iloc[0] + c.iloc[0]) / 2
for i in range(1, len(c)):
    ha_o[i] = (ha_o[i - 1] + ha_c.iloc[i - 1]) / 2
record("Heikin-Ashi trend (long)", (ha_c > pd.Series(ha_o, idx)).astype(float), group="repo")

med = (h + l) / 2
ao = sma(med, 5) - sma(med, 34)
record("Awesome oscillator > 0 (long)", (ao > 0).astype(float), group="repo")

N, k = 4, 0.5
rng = pd.concat([h.rolling(N).max() - c.rolling(N).min(), c.rolling(N).max() - l.rolling(N).min()], axis=1).max(axis=1).shift(1)
up, dn = o + k * rng, o - k * rng
lh, sh = (h >= up), (l <= dn)
dt = pd.Series(np.where(lh & ~sh, c / np.maximum(up, o) - 1, np.where(sh & ~lh, 1 - c / np.minimum(dn, o), 0.0)), idx)
dt = dt - 2 * COST * (lh ^ sh)
record("Dual Thrust (intraday, daily-bar approx)", (lh ^ sh).astype(float), returns=dt, group="repo")

# ---------------- 2. pairs trading ----------------
print("\n# pairs trading (rolling hedge ratio, z>2 entry, exit at |z|<0.5)")
pair_rets = []
for a, b in [("KO", "PEP"), ("XOM", "CVX"), ("V", "MA"), ("HD", "LOW"), ("GLD", "GDX"), ("JPM", "BAC")]:
    A, B = dl(a), dl(b)
    pa, pb = np.log(align(A["close"])), np.log(align(B["close"]))
    beta = pa.rolling(252).cov(pb) / pb.rolling(252).var()
    spread = pa - beta * pb
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    pos = hold_state(z < -2, z > -0.5, z > 2, z < 0.5)
    ra = align(A["open"]).shift(-2) / align(A["open"]).shift(-1) - 1
    rb = align(B["open"]).shift(-2) / align(B["open"]).shift(-1) - 1
    gross = 1 + beta.abs()
    r = pos * (ra - beta * rb) / gross - 2 * COST * pos.diff().abs().fillna(0)
    pair_rets.append(r.fillna(0))
    record(f"Pair {a}/{b}", pos, returns=r.fillna(0), group="pairs")
record("Pairs basket (equal weight, 6 pairs)", one, returns=pd.concat(pair_rets, axis=1).mean(axis=1), group="pairs")

# ---------------- 3. ML ----------------
print("\n# current live model, emulated exactly (trained once on first 70% since 2015)")
ldf = live.load_data()
lmodel, lcols, ltest = live.train(ldf)
split_day = ltest.index[0]
lp = pd.Series(live.predict(lmodel, ldf[lcols]), ldf.index).reindex(idx)
lpos = ((lp >= 0.5) & ~(ext["^VIX"] > 35)).astype(float).where(idx >= split_day, 0.0)
lr = lpos * R - COST * lpos.diff().abs().fillna(0)
s_live, s_bh = stats(lr, lpos, split_day, idx[-1]), stats(R, one, split_day, idx[-1])
print(f"  live model OOS {split_day.date()}..: sharpe {s_live['sharpe']:.2f} cagr {s_live['cagr']:.1%} mdd {s_live['mdd']:.1%} expo {s_live['expo']:.0%}"
      f"  | buy&hold same window: sharpe {s_bh['sharpe']:.2f} cagr {s_bh['cagr']:.1%} mdd {s_bh['mdd']:.1%}")
pk = live.predict(lmodel, ldf[lcols].loc[split_day:].dropna())
print(f"  live model OOS prob: mean {pk.mean():.2f}  share >=0.5: {(pk >= 0.5).mean():.0%}")

base = live.add_features(spy, ext["^VIX"].rename("vix").to_frame())
X0 = base[live.FEATURE_COLS].astype(float)
X1 = X0.copy()
for t in ["TLT", "HYG", "GLD", "UUP", "QQQ", "IWM"]:
    s = ext[t]
    X1[f"{t}_r5"], X1[f"{t}_r20"] = s.pct_change(5), s.pct_change(20)
    X1[f"{t}_rel20"] = s.pct_change(20) - c.pct_change(20)
X1["vix_term"] = ext["^VIX"] / ext["^VIX3M"]
X1["t10y2y"], X1["t10y2y_chg20"] = t10y2y, t10y2y.diff(20)
X1["tnx_chg20"] = ext["^TNX"].diff(20)
X1["dd_52w"] = c / c.rolling(252).max() - 1
X1["rv_ratio"] = ret1.rolling(5).std() / ret1.rolling(60).std()
X1 = X1.astype(float)

LIVE_P = dict(objective="binary:logistic", eta=0.03, max_depth=5, seed=42, verbosity=0)
REG_P = dict(objective="binary:logistic", eta=0.05, max_depth=3, min_child_weight=20, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=5, seed=42, verbosity=0)


def walk_forward(X, horizon=5, params=LIVE_P, rounds=500, step=63):
    fwd = c.shift(-horizon) / c - 1
    y = (fwd > 0).astype(float).where(fwd.notna())
    preds = pd.Series(np.nan, idx)
    starts = idx[idx >= WF_START][::step]
    n = np.arange(len(idx))
    for i, t0 in enumerate(starts):
        loc0 = idx.get_loc(t0)
        tr = (n <= loc0 - horizon) & y.notna().values
        p = dict(params, scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        m = xgb.train(p, xgb.QuantileDMatrix(X[tr], label=y[tr]), rounds)
        end = idx.get_loc(starts[i + 1]) if i + 1 < len(starts) else len(idx)
        preds.iloc[loc0:end] = m.predict(xgb.DMatrix(X.iloc[loc0:end]))
    return preds


def ml_rules(name, p):
    blocked = ext["^VIX"] > 35
    record(f"{name} | long p>=.50", ((p >= 0.5) & ~blocked).astype(float), group="ml")
    record(f"{name} | long p>=.55", ((p >= 0.55) & ~blocked).astype(float), group="ml")
    record(f"{name} | L/S .55/.45", pd.Series(np.where(p >= 0.55, 1.0, np.where(p <= 0.45, -1.0, 0.0)), idx), group="ml")
    record(f"{name} | sized x vol target", ((p - 0.45) / 0.15).clip(0, 1) * vt, group="ml")


print("\n# walk-forward ML (quarterly retrain, 5-day embargo)")
for name, X, hz, prm, rounds in [
    ("XGB live params, live feats, 5d", X0, 5, LIVE_P, 500),
    ("XGB live params, +macro feats, 5d", X1, 5, LIVE_P, 500),
    ("XGB regularized, +macro feats, 5d", X1, 5, REG_P, 300),
    ("XGB regularized, +macro feats, 20d", X1, 20, REG_P, 300),
    ("XGB regularized, +macro feats, 1d", X1, 1, REG_P, 300),
]:
    t = time.time()
    p = walk_forward(X, hz, prm, rounds)
    print(f" {name}: {time.time() - t:.0f}s, mean prob {p.loc[WF_START:].mean():.2f}")
    ml_rules(name, p)

print("\n# pre-registered combinations (50/50 capital split)")
trend_vt = (c > sma(c, 200)) * vt
rsi2_pos = hold_state((rsi2 < 10) & (c > sma(c, 200)), c > sma(c, 5))
p_ml = walk_forward(X1, 5, REG_P, 300)
ml_vt = ((p_ml - 0.45) / 0.15).clip(0, 1) * vt
record("COMBO trend+VT / RSI2", 0.5 * trend_vt + 0.5 * rsi2_pos, group="combo")
record("COMBO trend+VT / ML sized", 0.5 * trend_vt + 0.5 * ml_vt, group="combo")
record("COMBO trend+VT gated by ML p>=.45", trend_vt * (p_ml >= 0.45), group="combo")

df = pd.DataFrame(RESULTS)
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 30)
cols = ["group", "name", "dev_sharpe", "dev_cagr", "dev_mdd", "dev_vol", "dev_expo", "dev_turns",
        "hold_sharpe", "hold_cagr", "hold_mdd"]
print("\n\n===== SORTED BY DEV SHARPE =====")
print(df.sort_values("dev_sharpe", ascending=False)[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
