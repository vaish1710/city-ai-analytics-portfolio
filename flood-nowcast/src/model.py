"""Forecast the non-tidal water residual 1, 3 and 6 hours ahead, then judge it as a flood warning system.

Water level forecast = astronomical tide at t+h  +  forecast residual.
Methods compared at every horizon (all get the same margin tuning on the same validation data):
  tide_only    residual forecast = 0       (what you see if you only read the published tide table)
  persistence  residual stays as it is now (the standard hard-to-beat baseline)
  model        gradient boosting, mean forecast
  model_q90    gradient boosting, 90th percentile forecast (a deliberately cautious upper estimate)

Evaluation is chronological: train on the past, tune the alert margin on the next block, test on the last block,
with a gap of max-horizon hours between blocks so no target overlaps. Random splits would leak.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import make_features

HORIZONS = (1, 3, 6)
MARGINS = np.round(np.arange(0.0, 1.55, 0.1), 2)  # ft below the threshold at which we still raise an alert


def chrono_split(n, fracs=(0.6, 0.2, 0.2), gap=max(HORIZONS)):
    a = int(n * fracs[0]); b = a + int(n * fracs[1])
    return slice(0, a), slice(a + gap, b), slice(b + gap, n)


def event_scores(obs, score, threshold, margin):
    """obs, score: arrays of future observed level and forecast level. Event = obs >= threshold."""
    ok = ~(np.isnan(obs) | np.isnan(score))
    obs, score = obs[ok], score[ok]
    event, alert = obs >= threshold, score >= threshold - margin
    hits = int((event & alert).sum()); miss = int((event & ~alert).sum()); fa = int((~event & alert).sum())
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + miss + fa) if hits + miss + fa else np.nan
    return {"events": hits + miss, "hits": hits, "misses": miss, "false_alarms": fa,
            "POD": pod, "FAR": far, "CSI": csi}


def best_margin(obs, score, threshold):
    best = (0.0, -1.0)
    for m in MARGINS:
        csi = event_scores(obs, score, threshold, m)["CSI"]
        if not np.isnan(csi) and csi > best[1]:
            best = (float(m), csi)
    return best[0]


def _gb(**kw):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                                         random_state=0, **kw)


def fit(df, threshold):
    """Train on the first block, tune margins on the second, report on the third. Returns (bundle, results)."""
    tr, va, te = chrono_split(len(df))
    bundle = {"threshold": float(threshold), "models": {}, "margins": {}, "features": {}}
    rows = []
    for h in HORIZONS:
        X, y = make_features(df, h)
        cols = list(X.columns)
        bundle["features"][h] = cols
        fut_obs = df["observed"].asfreq("h").shift(-h).values
        pred_fut = X["pred_fut"].values
        resid_now = X["resid_lag0"].values

        # Learn the CHANGE from the current residual, so persistence is the starting point and the model
        # only has to learn what persistence misses. (Learning the raw residual made trees worse than persistence.)
        delta = y - X["resid_lag0"]
        Xtr, ytr = X.iloc[tr], delta.iloc[tr]
        m = ytr.notna() & Xtr.notna().any(axis=1)
        mean_m = _gb().fit(Xtr[m], ytr[m])
        q_m = _gb(loss="quantile", quantile=0.9).fit(Xtr[m], ytr[m])
        bundle["models"][h] = {"model": mean_m, "model_q90": q_m}

        scores = {
            "tide_only": pred_fut,
            "persistence": pred_fut + resid_now,
            "model": pred_fut + resid_now + mean_m.predict(X),
            "model_q90": pred_fut + resid_now + q_m.predict(X),
        }
        for name, s in scores.items():
            marg = best_margin(fut_obs[va], s[va], threshold)
            bundle["margins"][(h, name)] = marg
            res = event_scores(fut_obs[te], s[te], threshold, marg)
            err = (fut_obs[te] - s[te])
            err = err[~np.isnan(err)]
            rows.append({"horizon_h": h, "method": name, "margin_ft": marg,
                         "rmse_ft": float(np.sqrt(np.mean(err ** 2))) if name != "model_q90" and len(err) else np.nan, **res})
    return bundle, pd.DataFrame(rows)


def forecast_series(bundle, df, h, key="model"):
    """Level forecasts issued at every hour, indexed by the time each forecast is FOR (issue time + h)."""
    X, _ = make_features(df, h)
    Xc = X.reindex(columns=bundle["features"][h])
    ok = (X["pred_fut"].notna() & X["resid_lag0"].notna()).values
    fc = pd.Series(np.nan, index=X.index)
    fc[ok] = X["pred_fut"][ok] + X["resid_lag0"][ok] + bundle["models"][h][key].predict(Xc[ok])
    fc.index = fc.index + pd.Timedelta(hours=h)
    return fc


def forecast_row(bundle, X_row: pd.DataFrame, h, pred_fut):
    """Forecast for one feature row. Returns dict of mean and q90 level forecasts (ft)."""
    cols = bundle["features"][h]
    X = X_row.reindex(columns=cols)
    base = float(pred_fut + X["resid_lag0"].iloc[0])  # tide + current residual (persistence), then the learned change
    return {"mean": base + float(bundle["models"][h]["model"].predict(X)[0]),
            "q90": base + float(bundle["models"][h]["model_q90"].predict(X)[0])}
