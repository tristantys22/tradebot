"""Zero-shot foundation models on SPY: every 5 trading days forecast the 5-day return, hold long
(or long/short) for the next 5 days. Same execution as research.py: decide at close, fill next open, 2 bp."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/kronos")
import numpy as np, pandas as pd, torch, yfinance as yf

torch.set_num_threads(8)
torch.manual_seed(0)
H, CTX, COST = 5, 512, 0.0002
EVAL_START, DEV_END, HOLD_START = "2015-01-01", "2024-11-13", "2024-11-14"

df = yf.download("SPY", start="2003-01-01", auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
idx, o, c = df.index, df["open"], df["close"]
R = o.shift(-2) / o.shift(-1) - 1
first = max(CTX, idx.searchsorted(pd.Timestamp(EVAL_START)))
dec = list(range(first, len(idx) - 1, H))
real5 = np.array([c.iloc[min(i + H, len(c) - 1)] / c.iloc[i] - 1 for i in dec])
print(f"{len(dec)} decision dates {idx[dec[0]].date()}..{idx[dec[-1]].date()}", flush=True)


def evaluate(name, pred_ret):
    pred_ret = np.asarray(pred_ret)
    out = {}
    for mode in ("long", "long/short"):
        pos = pd.Series(0.0, idx)
        for i, p in zip(dec, pred_ret):
            pos.iloc[i:i + H] = 1.0 if p > 0 else (-1.0 if mode == "long/short" else 0.0)
        r = pos * R - COST * pos.diff().abs().fillna(0)
        for label, a, b in (("dev", EVAL_START, DEV_END), ("hold", HOLD_START, idx[-1])):
            x = r.loc[a:b].dropna()
            eq = (1 + x).cumprod()
            out[f"{mode} {label}"] = (x.mean() / x.std() * np.sqrt(252), eq.iloc[-1] ** (252 / len(x)) - 1,
                                      (eq / eq.cummax() - 1).min(), pos.loc[a:b].abs().mean())
    done = idx[dec] <= pd.Timestamp(DEV_END)
    hit = (np.sign(pred_ret) == np.sign(real5))[done].mean()
    print(f"\n== {name}: 5d direction hit-rate (dev) {hit:.1%} | predicts UP {np.mean(pred_ret > 0):.0%} of the time | "
          f"corr(pred, real) {np.corrcoef(pred_ret[done], real5[done])[0, 1]:+.3f}")
    for k, (s, g, m, e) in out.items():
        print(f"   {k:16s} sharpe {s:5.2f}  cagr {g:6.1%}  maxDD {m:6.1%}  exposure {e:4.0%}")


evaluate("Buy & hold (always UP)", np.ones(len(dec)))
evaluate("Last 5d momentum (naive)", [c.iloc[i] / c.iloc[i - H] - 1 for i in dec])

from chronos import BaseChronosPipeline
import os
for repo in () if os.environ.get("ONLY_KRONOS") else ("amazon/chronos-bolt-small", "amazon/chronos-bolt-base"):
    t = time.time()
    pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32)
    preds = []
    for k in range(0, len(dec), 128):
        ctx = [torch.tensor(c.values[i - CTX + 1:i + 1], dtype=torch.float32) for i in dec[k:k + 128]]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        preds += list(q[:, -1, 0].numpy() / np.array([c.iloc[i] for i in dec[k:k + 128]]) - 1)
    print(f"[{repo}] {time.time() - t:.0f}s")
    evaluate(repo, preds)

from model import Kronos, KronosTokenizer, KronosPredictor
t = time.time()
predictor = KronosPredictor(Kronos.from_pretrained("NeoQuasar/Kronos-small"),
                            KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"),
                            device="cpu", max_context=CTX)
preds = []
for k in range(0, len(dec), 32):
    chunk = dec[k:k + 32]
    xs = [df.iloc[i - CTX + 1:i + 1].reset_index(drop=True) for i in chunk]
    xt = [pd.Series(idx[i - CTX + 1:i + 1]) for i in chunk]
    yt = [pd.Series(pd.bdate_range(idx[i] + pd.Timedelta(days=1), periods=H)) for i in chunk]
    out = predictor.predict_batch(xs, xt, yt, pred_len=H, T=1.0, top_p=0.9, sample_count=5, verbose=False)
    preds += [p["close"].iloc[-1] / c.iloc[i] - 1 for p, i in zip(out, chunk)]
    print(f"  kronos {k + len(chunk)}/{len(dec)} {time.time() - t:.0f}s", flush=True)
evaluate("Kronos-small (5 samples avg)", preds)
