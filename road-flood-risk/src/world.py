"""Simulator v2 and history features for the in-depth study. The original flood.py is unchanged.

What v2 adds, each switchable so the study can ask 'when does a model help?':
  * basins (watersheds): a hidden basin-level effect, and a basin id that is OBSERVABLE (the city publishes watershed maps)
  * drift: some segments change after a point in time (a blocked inlet is cleared, a new blockage appears), so old history goes stale
  * reporting bias: low-traffic roads get fewer reports, so 'reported incident' is not 'flooded'
  * forecast noise, and a limited history window (e.g. only the last 6 storms are on file)
Everything about history uses EARLIER storms only.
"""
import numpy as np
import pandas as pd

BASE_FEATS = ["peak_fc", "total_fc", "tide_fc", "elev_ft", "drain_dist_ft", "tidal"]
HIST_FEATS = ["prior_rate", "prior_decay", "basin_rate"]
FEATURES2 = BASE_FEATS + HIST_FEATS


def make_world(n_seg=60, n_storms=120, seed=3, n_basins=8, report_base=0.7, report_bias=False, fc_sigma=0.25,
               drift_frac=0.0, drift_at=0.5, window=None, half_life=8):
    rng = np.random.default_rng(seed)
    basin = rng.integers(0, n_basins, n_seg)
    seg = pd.DataFrame({"segment": range(n_seg), "basin": basin, "elev_ft": rng.gamma(2.0, 3.0, n_seg).clip(0.5, 25),
                        "drain_dist_ft": rng.uniform(50, 1500, n_seg), "tidal": (rng.random(n_seg) < 0.35).astype(int),
                        "local": rng.normal(0, 0.8, n_seg)})
    basin_eff = rng.normal(0, 0.6, n_basins)
    traffic = rng.lognormal(0, 0.6, n_seg)
    drifters = rng.random(n_seg) < drift_frac
    shift = rng.choice([-1.6, 1.6], n_seg) * drifters
    total = rng.gamma(2.0, 1.4, n_storms); peak = total * rng.uniform(0.3, 0.8, n_storms); tide = rng.normal(2.0, 0.8, n_storms)
    fc = {"peak_fc": peak * rng.lognormal(0, fc_sigma, n_storms), "total_fc": total * rng.lognormal(0, fc_sigma, n_storms),
          "tide_fc": tide + rng.normal(0, 0.3, n_storms)}
    S = np.repeat(np.arange(n_storms), n_seg); G = np.tile(np.arange(n_seg), n_storms)
    after = (S >= int(drift_at * n_storms)).astype(float)
    z = (-6.4 + 0.9 * peak[S] + 0.25 * total[S] - 0.22 * seg.elev_ft.values[G] + 0.0012 * seg.drain_dist_ft.values[G]
         + np.where(seg.tidal.values[G] == 1, 1.8 * np.maximum(tide[S] - 3.0, 0), 0) + seg.local.values[G] + basin_eff[basin[G]]
         + shift[G] * after)
    p_true = 1 / (1 + np.exp(-z))
    flooded = rng.random(len(z)) < p_true
    rank = pd.Series(traffic).rank(pct=True).values
    tf = 0.35 + 0.65 * rank
    rp = np.clip(report_base * tf / tf.mean(), 0, 1) if report_bias else np.full(n_seg, report_base)
    reported = flooded & (rng.random(len(z)) < rp[G])
    df = pd.DataFrame({"storm": S, "segment": G, "basin": basin[G], "elev_ft": seg.elev_ft.values[G], "drain_dist_ft": seg.drain_dist_ft.values[G],
                       "tidal": seg.tidal.values[G], "peak_fc": fc["peak_fc"][S], "total_fc": fc["total_fc"][S], "tide_fc": fc["tide_fc"][S],
                       "flooded": flooded.astype(int), "incident": reported.astype(int), "p_true": p_true,   # p_true/flooded: synthetic only, never features
                       "storm_total_obs": total[S]})
    return add_history(df, window=window, half_life=half_life)


def add_history(df, window=None, half_life=8):
    """Adds prior (count), prior_rate, prior_decay and basin_rate. All use storms strictly before the current one.
    prior: incidents in the last `window` storms (all earlier storms if None). prior_decay: recency-weighted count with the given half-life (in storms).
    basin_rate: average prior_rate of the OTHER segments in the same basin (never the segment itself)."""
    M = df.pivot(index="storm", columns="segment", values="incident").sort_index().sort_index(axis=1).fillna(0).values
    T, G = M.shape
    C = np.vstack([np.zeros((1, G)), np.cumsum(M, axis=0)])                      # C[t] = incidents before storm t
    t = np.arange(T)
    lo = np.zeros(T, dtype=int) if window is None else np.maximum(0, t - window)
    prior = C[t] - C[lo]                                                          # incidents in storms lo .. t-1
    rate = prior / ((t - lo)[:, None] + 5)
    gamma = 0.5 ** (1 / half_life)
    D = np.zeros((T, G))
    for i in range(1, T):
        D[i] = gamma * D[i - 1] + M[i - 1]
    basin = df.drop_duplicates("segment").sort_values("segment").basin.values
    peers = np.zeros((T, G))
    for b in np.unique(basin):
        idx = np.where(basin == b)[0]
        if len(idx) > 1:
            tot = rate[:, idx].sum(axis=1, keepdims=True)
            peers[:, idx] = (tot - rate[:, idx]) / (len(idx) - 1)
    out = df.sort_values(["storm", "segment"]).reset_index(drop=True).copy()
    out["prior"], out["prior_rate"], out["prior_decay"], out["basin_rate"] = prior.ravel(), rate.ravel(), D.ravel(), peers.ravel()
    return out


def _grid(df, score, label):
    """Scores and labels as (storms x segments) arrays, in (storm, segment) order, whatever the row order or index of df."""
    order = np.lexsort((df.segment.values, df.storm.values))
    T = df.storm.nunique(); G = len(df) // T
    assert T * G == len(df), "needs every segment in every storm"
    return np.asarray(score)[order].reshape(T, G), df[label].values[order].reshape(T, G), np.sort(df.storm.unique())


def recall_k(df, score, label="incident", k=8):
    """Share of `label` events at the k highest-scored segments of each storm (ties -> lower segment id), pooled over storms;
    plus mean false alerts per storm (flagged segments with no event)."""
    S, Y, _ = _grid(df, score, label)
    hits = tot = 0; fa = []
    for i in range(len(S)):
        top = np.argsort(-S[i], kind="stable")[:k]
        h = int(Y[i][top].sum()); hits += h; tot += int(Y[i].sum()); fa.append(k - h)
    return (hits / tot if tot else float("nan")), float(np.mean(fa))


def per_storm(df, score, label="incident", k=8):
    S, Y, storms = _grid(df, score, label)
    return pd.DataFrame({"storm": storms, "hits": [int(Y[i][np.argsort(-S[i], kind="stable")[:k]].sum()) for i in range(len(S))], "total": Y.sum(axis=1)})
