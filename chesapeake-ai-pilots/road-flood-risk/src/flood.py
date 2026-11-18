"""Road-segment flood risk: rank segments before a storm; test on whole storms the model has never seen.

Label = a REPORTED flooding incident at a segment during a storm (not true flooding: some flooding is never reported).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["peak_fc", "total_fc", "tide_fc", "elev_ft", "drain_dist_ft", "tidal", "prior_rate"]


def make_data(n_seg=60, n_storms=80, seed=3, report_prob=0.7):
    """Synthetic city. Flooding depends on rain, elevation, drainage distance, tide (tidal roads) and a hidden local
    effect (blocked inlet etc.) that only shows up through past incidents. Forecasts are noisy."""
    rng = np.random.default_rng(seed)
    seg = pd.DataFrame({"segment": range(n_seg), "elev_ft": rng.gamma(2.0, 3.0, n_seg).clip(0.5, 25),
                        "drain_dist_ft": rng.uniform(50, 1500, n_seg), "tidal": rng.random(n_seg) < 0.35,
                        "local": rng.normal(0, 0.9, n_seg)})
    total = rng.gamma(2.0, 1.4, n_storms)
    peak = total * rng.uniform(0.3, 0.8, n_storms)
    tide = rng.normal(2.0, 0.8, n_storms)
    S = np.repeat(np.arange(n_storms), n_seg)
    G = np.tile(np.arange(n_seg), n_storms)
    sg = seg.iloc[G].reset_index(drop=True)
    z = (-6.2 + 0.9 * peak[S] + 0.25 * total[S] - 0.22 * sg.elev_ft + 0.0012 * sg.drain_dist_ft
         + np.where(sg.tidal, 1.8 * np.maximum(tide[S] - 3.0, 0), 0) + sg.local)
    p_true = 1 / (1 + np.exp(-z))
    flooded = rng.random(len(z)) < p_true
    df = pd.DataFrame({"storm": S, "segment": G, "elev_ft": sg.elev_ft, "drain_dist_ft": sg.drain_dist_ft,
                       "tidal": sg.tidal.astype(int),
                       "peak_fc": peak[S] * rng.lognormal(0, 0.25, len(z)),
                       "total_fc": total[S] * rng.lognormal(0, 0.25, len(z)),
                       "tide_fc": tide[S] + rng.normal(0, 0.3, len(z)),
                       "incident": (flooded & (rng.random(len(z)) < report_prob)).astype(int),
                       "p_true": p_true})   # synthetic only: used for the ceiling, never as a feature
    # forecasts are per-storm in reality: use one forecast per storm, not per row
    for c in ("peak_fc", "total_fc", "tide_fc"):
        df[c] = df.groupby("storm")[c].transform("first")
    return add_prior(df)


def add_prior(df):
    """Incidents at the same segment in EARLIER storms only (never the current or later ones)."""
    df = df.sort_values(["segment", "storm"]).copy()
    df["prior"] = df.groupby("segment")["incident"].cumsum() - df["incident"]
    # a count grows with time (train vs test drift), so the model gets a rate per earlier storm (shrunk toward 0 early on)
    df["prior_rate"] = df["prior"] / (df["storm"] + 5)
    return df.sort_values(["storm", "segment"]).reset_index(drop=True)


def split_by_storm(df, train_frac=0.7):
    cut = int(df.storm.max() * train_frac)
    return df[df.storm <= cut], df[df.storm > cut]


def fit(train, kind="gbm"):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(train[FEATURES], train.incident)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, random_state=0).fit(train[FEATURES], train.incident)


def recall_at_k(df, score, k=8):
    """Per storm: flag the k highest-scored segments. Returns (share of incidents caught, mean false alerts per storm).
    Storms with no incidents are skipped for recall (nothing to catch) but still count for false alerts."""
    d = df.assign(score=np.asarray(score))
    caught = total = 0
    fa = []
    for _, g in d.groupby("storm"):
        top = g.sort_values(["score", "segment"], ascending=[False, True]).head(k)
        hits = int(top.incident.sum())
        caught += hits
        total += int(g.incident.sum())
        fa.append(k - hits)
    return (caught / total if total else float("nan")), float(np.mean(fa))


def per_storm_hits(df, score, k=8):
    """Per storm: (hits among the k flagged segments, total incidents). Same rule as recall_at_k."""
    d = df.assign(score=np.asarray(score)); rows = []
    for st, g in d.groupby("storm"):
        top = g.sort_values(["score", "segment"], ascending=[False, True]).head(k)
        rows.append((st, int(top.incident.sum()), int(g.incident.sum())))
    return pd.DataFrame(rows, columns=["storm", "hits", "total"])


def baselines(test, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.random(len(test)) * 1e-6              # deterministic tie-break
    return {"hotspot history": test.prior.values + noise,
            "lowest elevation": -test.elev_ft.values + noise}


def reasons(row, ref):
    """Plain-language flags from simple thresholds on the training data. NOT a model explanation."""
    out = []
    if row.peak_fc >= ref["peak_q75"]: out.append("heavy forecast rain")
    if row.elev_ft <= ref["elev_q25"]: out.append("low elevation")
    if row.prior >= 2: out.append(f"{int(row.prior)} past incidents here")
    if row.tidal and row.tide_fc >= 3.0: out.append("high tide on a tidal road")
    return out or ["combined factors"]


def ref_thresholds(train):
    return {"peak_q75": float(train.peak_fc.quantile(0.75)), "elev_q25": float(train.elev_ft.quantile(0.25))}


def evaluate(seed, k=8, n_storms=80):
    df = make_data(seed=seed, n_storms=n_storms)
    tr, te = split_by_storm(df)
    scores = {"model (boosted trees)": fit(tr).predict_proba(te[FEATURES])[:, 1],
              "model (logistic)": fit(tr, "logit").predict_proba(te[FEATURES])[:, 1], **baselines(te),
              "ceiling (knows true risk; not achievable)": te.p_true.values}
    res = {}
    for name, s in scores.items():
        r, fa = recall_at_k(te, s, k)
        res[name] = {"recall_at_k": r, "false_alerts_per_storm": fa}
    return res
