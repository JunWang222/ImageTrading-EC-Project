"""Pretrained time-series foundation models (Chronos-2, TimesFM, Moirai 2, Kronos), zero-shot, on the
same rows as the chart models: daily bars (Step 1b) and 5-minute bars (Steps 2b / 3b).

Each model forecasts the next h closes of a series normalised by its last close, as 9 quantiles
(0.1 ... 0.9). Nothing is trained on our labels. The forecast is turned into a score per target:

    dir     median forecast return at step h                       q50[h] - 1
    xsdir   the same minus its cross-sectional median at that time
    vol     mean absolute forecast return over the 9 quantiles at step h, divided by the stock's
            training-period median |r| (the label's own threshold)

Inputs are point in time: every context window is put on its last bar's basis (daily: divided by
the anchor day's multipliers; 5-minute: chartcnn.images.point_in_time_window), so no later split
or dividend reaches it. Kronos also reads open/high/low/volume and the bar timestamps.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_compare.forecasters import QUANTILES, build

__version__ = "2026-11-23.1"

# name -> wrapper spec (tsfm_compare.forecasters.build); `default` = used unless the notebook says otherwise
MODELS = {
    "randomwalk": dict(kind="randomwalk", model_id="none", batch_size=4096, label="random walk, recent sigma (not pretrained)"),
    "chronos2": dict(kind="chronos", model_id="amazon/chronos-2", batch_size=512, label="Chronos-2"),
    "timesfm": dict(kind="timesfm", model_id="google/timesfm-3.0-500m-pytorch", batch_size=256, label="TimesFM"),
    "moirai2": dict(kind="moirai", model_id="Salesforce/moirai-2.0-R-small", batch_size=256, label="Moirai 2 small"),
    "kronos": dict(kind="kronos", model_id="NeoQuasar/Kronos-small", batch_size=64, samples=4,
                   tokenizer="NeoQuasar/Kronos-Tokenizer-base", label="Kronos-small"),
}


def label(name):
    if name == "randomwalk":
        return MODELS[name]["label"]
    return f"{MODELS[name]['label']} (zero-shot)" if name in MODELS else name


def _fill(x):
    """Forward-fill missing rows of a (L, 5) window, then back-fill the start."""
    f = pd.DataFrame(x).ffill().bfill().to_numpy()
    return f


# ----------------------------------------------------------------------------- contexts

def daily_contexts(store, frame, length, horizon):
    """Daily context of `length` sessions ending at each anchor (DailyStore from daily_jkx).

    Returns dict(close (n, L) normalised by the last close, ohlcv (n, L, 5), t_past, t_future)."""
    s, t = frame.security_index.to_numpy(), frame.t.to_numpy()
    n = len(frame)
    ohlcv = np.empty((n, length, 5))
    for i in range(n):
        a = t[i] - length + 1
        x = np.full((length, 5), np.nan)
        lo = max(a, 0)
        x[lo - a:] = store.ohlcv[s[i], lo:t[i] + 1]
        price, volume = store.mult[s[i], t[i]]
        x[:, :4] /= price
        x[:, 4] /= volume
        ohlcv[i] = _fill(x)
    days = pd.to_datetime(store.dates).values.astype("datetime64[D]")
    past = np.stack([days[np.clip(np.arange(ti - length + 1, ti + 1), 0, None)] for ti in t])
    future = np.stack([np.busday_offset(days[ti], np.arange(1, horizon + 1), roll="forward") for ti in t])
    to_close = lambda d: d.astype("datetime64[m]") + np.timedelta64(16 * 60, "m")
    close = (ohlcv[:, :, 3] / ohlcv[:, -1:, 3]).astype(np.float32)
    return dict(close=close, ohlcv=ohlcv.astype(np.float32), t_past=to_close(past), t_future=to_close(future))


def intraday_contexts(prepared, store, frame, length, horizon):
    """5-minute context of `length` bars ending at each decision bar (chartcnn Store / bundle)."""
    from chartcnn.images import point_in_time_window
    feats = np.load(Path(prepared) / "features.npy", mmap_mode="r")
    mult = np.load(Path(prepared) / "multipliers.npy", mmap_mode="r")
    s, t = frame.security_index.to_numpy(np.int64), frame.end_index.to_numpy(np.int64)
    ohlcv = np.empty((len(frame), length, 5), np.float32)
    order = np.argsort(s, kind="stable")                  # read one stock's memmap rows together
    for i in order:
        ohlcv[i] = point_in_time_window(feats[s[i]], mult[s[i]], t[i], length)
    clock = pd.to_datetime(store.manifest["decision_clock_ns"], unit="ns", utc=True)
    clock = clock.tz_convert("America/New_York").tz_localize(None).values.astype("datetime64[m]")
    past = clock[t[:, None] + np.arange(-length + 1, 1)[None, :]]
    future = clock[t][:, None] + np.arange(1, horizon + 1)[None, :] * np.timedelta64(5, "m")
    close = (ohlcv[:, :, 3] / ohlcv[:, -1:, 3]).astype(np.float32)
    return dict(close=close, ohlcv=ohlcv, t_past=past, t_future=future)


# ----------------------------------------------------------------------------- forecasting

def forecast_all(names, contexts_fn, keys, horizons, out_dir, device, specs=None, log=print):
    """Quantile forecasts of every model for every horizon, cached as npz in out_dir.

    contexts_fn(h) -> context dict for the rows `keys` (a frame; saved next to the forecasts so a
    resume can check it forecasts the same rows). Returns ({name: {h: (n, 9) quantiles at step h}},
    {name: parameters})."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = keys.reset_index(drop=True)
    kpath = out_dir / "forecast_rows.parquet"
    if kpath.exists():
        old = pd.read_parquet(kpath)
        assert old.equals(keys), "cached forecasts are for other rows: use a new RUN_ID"
    else:
        keys.to_parquet(kpath, index=False)
    ppath = out_dir / "parameters.json"
    params = json.loads(ppath.read_text()) if ppath.exists() else {}
    out = {}
    ctx_cache = {}
    for name in names:
        out[name] = {}
        todo = [h for h in horizons if not (out_dir / f"{name}_h{h}.npz").exists()]
        if todo:
            spec = dict((specs or MODELS)[name]); spec.pop("label", None)
            f = build(spec, device).load()
            params[f"tsfm_{name}"] = int(f.n_params)
            ppath.write_text(json.dumps(params, indent=2))
            for h in todo:
                if h not in ctx_cache:
                    ctx_cache[h] = contexts_fn(h)
                ctx = {k: v for k, v in ctx_cache[h].items() if k in f.needs}
                t0 = time.perf_counter()
                q = f.forecast(ctx, h)                                   # (n, h, 9)
                np.savez_compressed(out_dir / f"{name}_h{h}.npz", q=q[:, -1, :].astype(np.float32))
                log(f"  {name} h={h}: {len(q):,} series in {(time.perf_counter() - t0) / 60:.1f} min")
            f.unload()
        for h in horizons:
            out[name][h] = np.load(out_dir / f"{name}_h{h}.npz")["q"]
    return out, params


def scores(q, kind, keys, time_col, threshold=None):
    """Score per row from (n, 9) quantiles of close / last close at step h (see module docstring)."""
    q = np.asarray(q, np.float64)
    if kind == "vol":
        s = np.abs(q - 1.0).mean(1)
        if threshold is not None:
            s = s / np.asarray(threshold, float)
        return s
    s = q[:, QUANTILES.index(0.5)] - 1.0
    if kind == "xsdir":
        s = s - pd.Series(s).groupby(keys[time_col].to_numpy()).transform("median").to_numpy()
    return s


def attach(pred, keys, quantiles, kind, on, time_col, threshold=None):
    """Add score_tsfm_<name> columns to `pred` for the rows it shares with `keys` (inner join)."""
    k = keys[on].copy()
    for name, q in quantiles.items():
        k[f"score_tsfm_{name}"] = scores(q, kind, keys, time_col,
                                         None if threshold is None else keys.security_index.map(threshold).to_numpy())
    return pred.merge(k, on=on, how="inner")
