"""Intersection safety screening: rank sites for review using Empirical Bayes (Highway Safety Manual style).
Ranks on 2019-2021 crashes only; judged on crashes in 2022-2023. Synthetic data."""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

TRAIN_YEARS = (2019, 2020, 2021)
TEST_YEARS = (2022, 2023)
FEATS = ["log_aadt", "signal", "legs4", "speed_dev"]


def make_sites(n=300, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"site": range(n), "aadt": np.exp(rng.normal(9.0, 0.6, n)).round(),
                      "signal": (rng.random(n) < 0.4).astype(int), "legs4": (rng.random(n) < 0.6).astype(int),
                      "speed": rng.choice([25, 35, 45], n)})
    u = rng.normal(0, 0.6, n)                       # hidden site-specific risk (geometry, sight lines)
    d["true_rate"] = np.exp(-7.0 + 0.75 * np.log(d.aadt) + 0.3 * d.signal + 0.25 * d.legs4 + 0.02 * (d.speed - 35) + u)
    for y in TRAIN_YEARS + TEST_YEARS:
        d[f"crashes_{y}"] = rng.poisson(d.true_rate)
    return features(d)


def features(d):
    d = d.copy()
    d["log_aadt"] = np.log(d.aadt)
    d["speed_dev"] = d.speed - 35
    return d


def train_total(d):
    return d[[f"crashes_{y}" for y in TRAIN_YEARS]].sum(axis=1).values


def fit_spf(d):
    """Safety performance function: expected crashes per year from site features (Poisson regression)."""
    y = train_total(d) / len(TRAIN_YEARS)
    m = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(d[FEATS], y, sample_weight=np.full(len(d), len(TRAIN_YEARS)))
    return m


def overdispersion(y, mu):
    """Method of moments for Var = mu + k*mu^2. Floored at 0.01."""
    return float(max(0.01, (((y - mu) ** 2 - mu).sum()) / (mu ** 2).sum()))


def eb_estimate(y, mu, k):
    """Empirical Bayes expected crashes: w*predicted + (1-w)*observed, w = 1/(1+k*mu)."""
    w = 1.0 / (1.0 + k * mu)
    return w * mu + (1 - w) * y


def rank_scores(d):
    y = train_total(d)
    spf = fit_spf(d)
    mu = spf.predict(d[FEATS]) * len(TRAIN_YEARS)
    k = overdispersion(y, mu)
    return {"raw crash count": y.astype(float), "crashes per vehicle": y / d.aadt.values, "model only (no history)": mu,
            "empirical Bayes": eb_estimate(y, mu, k)}, k


def top_idx(score, frac=0.10):
    n = max(1, int(round(len(score) * frac)))
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))      # ties -> lower site id first
    return order[:n]


def future_share(d, score, frac=0.10):
    """Share of 2022-2023 crashes that occurred at the top `frac` of sites."""
    fut = d[[f"crashes_{y}" for y in TEST_YEARS]].sum(axis=1).values
    return fut[top_idx(score, frac)].sum() / fut.sum()


def true_overlap(d, score, frac=0.10):
    """Synthetic only: share of the truly riskiest sites that the ranking found."""
    t = set(top_idx(d.true_rate.values, frac)); r = set(top_idx(score, frac))
    return len(t & r) / len(t)


def regression_to_mean(d, frac=0.10):
    """Sites picked for the most crashes in 2019-21: average crashes/yr then vs 2022-23, with nothing done to them."""
    idx = top_idx(train_total(d).astype(float), frac)
    pre = train_total(d)[idx].mean() / len(TRAIN_YEARS)
    post = d[[f"crashes_{y}" for y in TEST_YEARS]].sum(axis=1).values[idx].mean() / len(TEST_YEARS)
    return pre, post
