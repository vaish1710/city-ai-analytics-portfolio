"""Parcel-level expected annual flood loss model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

NUMERIC = [
    "elevation_ft", "dist_to_water_m", "building_value", "sqft", "year_built",
    "flood_reports_311_5y", "prior_claims_5y",
]
CATEGORICAL = ["fema_zone", "foundation"]
TARGET = "observed_annual_loss"
# Deliberately EXCLUDED: tract_median_income and owner_occupied. Income is used later
# to audit and (optionally) weight outcomes, never to predict physical risk.


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)


def new_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
    )


def cross_validate_by_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    """Hold out whole neighborhoods so the score reflects generalisation to unseen areas."""
    X, y, groups = build_matrix(df), df[TARGET].values, df["neighborhood"].values
    rows = []
    for tr, te in GroupKFold(n_splits=4).split(X, y, groups):
        m = new_model().fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[te])
        rows.append({
            "held_out": ", ".join(sorted(set(groups[te]))),
            "r2": r2_score(y[te], pred),
            "mae_usd": mean_absolute_error(y[te], pred),
        })
    return pd.DataFrame(rows)


def fit_and_score(df: pd.DataFrame) -> tuple[pd.DataFrame, GradientBoostingRegressor]:
    X, y = build_matrix(df), df[TARGET].values
    model = new_model().fit(X, y)
    out = df.copy()
    out["pred_annual_loss"] = np.clip(model.predict(X), 0, None)
    return out, model
