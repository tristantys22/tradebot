exec(open("/tmp/nuai_swing_strategies.py").read().split("# ---------- ML")[0])
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
n = np.arange(len(idx)); first = max(idx.searchsorted(pd.Timestamp(DEV[0])), 60)
def run(hz, seed, step=21):
    fwd = c.shift(-hz) / c - 1
    y = (fwd > 0).astype(float).where(fwd.notna())
    P = dict(objective="binary:logistic", eta=0.05, max_depth=2, min_child_weight=10, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=5, seed=seed, verbosity=0)
    prob = pd.Series(np.nan, idx)
    for s in range(first, len(idx), step):
        tr = (n <= s - hz) & y.notna().to_numpy() & X["gap_sma50"].notna().to_numpy()
        if tr.sum() < 40: continue
        m = xgb.train(P, xgb.DMatrix(X[tr], label=y[tr]), 150)
        prob.iloc[s:s + step] = m.predict(xgb.DMatrix(X.iloc[s:s + step]))
    return prob
print("hz seed step | dev sharpe/total | hold sharpe/total | trades dev/hold")
res = []
for hz in (2, 3, 5):
    for seed in range(5):
        for step in ((21, 1) if seed == 0 else (21,)):
            ROWS.clear(); record("x", (run(hz, seed, step) > 0.5).astype(float), "ml"); x = ROWS[0]
            res.append((x["dev_sharpe"], x["hold_sharpe"]))
            print(f"{hz:2d} {seed:4d} {step:4d} | {x['dev_sharpe']:5.2f} {x['dev_total']:6.0%} | {x['hold_sharpe']:5.2f} {x['hold_total']:6.0%} | {x['dev_trades']}/{x['hold_trades']}", flush=True)
res = np.array(res)
print(f"positive in BOTH periods: {(res.min(axis=1) > 0).sum()}/{len(res)}; median dev {np.median(res[:,0]):.2f} hold {np.median(res[:,1]):.2f}")
