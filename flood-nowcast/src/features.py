"""Leakage-safe features. The model forecasts the NON-TIDAL part of the water level (observed - predicted).

At time t a feature may use: observations at or before t, and astronomical tide predictions for ANY time
(those are published ahead of time, so they are legitimately known). It must never use observations after t.
tests/test_nowcast.py enforces this by corrupting the future and checking the features do not change.
"""
import pandas as pd

LAGS = (0, 1, 2, 3, 6, 12, 24)
MET_LAGS = (0, 3, 6)


def make_features(df: pd.DataFrame, horizon: int):
    """Return (X, y): features at each hour t, and the residual `horizon` hours later (y, NaN where unknown)."""
    d = df.asfreq("h")
    resid = d["observed"] - d["predicted"]
    X = pd.DataFrame(index=d.index)
    for k in LAGS:
        X[f"resid_lag{k}"] = resid.shift(k)
    X["resid_d1"] = resid - resid.shift(1)
    X["resid_d3"] = resid - resid.shift(3)
    X["pred_now"] = d["predicted"]
    X["pred_fut"] = d["predicted"].shift(-horizon)          # astronomical tide: known in advance
    X["pred_delta"] = X["pred_fut"] - X["pred_now"]
    for col in ("wind_speed", "pressure"):
        if col in d and d[col].notna().any():
            for k in MET_LAGS:
                X[f"{col}_lag{k}"] = d[col].shift(k)
    if "pressure" in d and d["pressure"].notna().any():
        X["pressure_d3"] = d["pressure"] - d["pressure"].shift(3)
    y = resid.shift(-horizon)
    return X, y
