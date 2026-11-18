"""python src/study.py  -> outputs/study/*.csv  (about 3-5 minutes on 2 CPUs)

Question: when, if ever, does a model beat 'check the segments that flooded before'? Every number is from the synthetic worlds in world.py.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import world as W

K_LIST = (2, 4, 6, 8, 12, 16, 20)
SEEDS = range(1, 17)


def systems(tr, te, seed):
    Xtr, ytr = tr[W.FEATURES2], tr.incident
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=100, random_state=0).fit(Xtr, ytr)
    return {"model (logistic)": logit.predict_proba(te[W.FEATURES2])[:, 1], "model (boosted trees)": gbm.predict_proba(te[W.FEATURES2])[:, 1],
            "hotspot (count)": te.prior.values.astype(float), "hotspot (recency-weighted)": te.prior_decay.values,
            "lowest elevation": -te.elev_ft.values, "random": np.random.default_rng(seed).random(len(te)),
            "ceiling (true risk; unattainable)": te.p_true.values}


def one_world(scenario, seed, curves=False, **wk):
    d = W.make_world(seed=seed, **wk)
    cut = int(d.storm.max() * 0.7); tr, te = d[d.storm <= cut], d[d.storm > cut]
    sc = systems(tr, te, seed)
    rows, extra = [], []
    ps = te.groupby("storm").agg(total_fc=("total_fc", "first"), n=("incident", "sum"))
    for name, s in sc.items():
        ri, fa = W.recall_k(te, s, "incident", 8); rf, _ = W.recall_k(te, s, "flooded", 8)
        rows.append({"scenario": scenario, "seed": seed, "system": name, "recall_reported": ri, "recall_true_flooding": rf, "false_alerts": fa})
        if curves:
            for k in K_LIST:
                extra.append({"kind": "k_curve", "seed": seed, "system": name, "k": k, "recall": W.recall_k(te, s, "incident", k)[0]})
            terc = pd.qcut(ps.total_fc, 3, labels=["small storms", "medium storms", "large storms"])
            for lab in terc.cat.categories:
                sub = te[te.storm.isin(ps.index[terc == lab])]
                idx = np.where(te.storm.isin(ps.index[terc == lab]).values)[0]
                extra.append({"kind": "strata", "seed": seed, "system": name, "stratum": lab, "recall": W.recall_k(sub, np.asarray(s)[idx], "incident", 8)[0]})
    rho = spearmanr(ps.total_fc, ps.n)[0]
    extra.append({"kind": "forecast_corr", "seed": seed, "scenario": scenario, "spearman_forecast_vs_incident_count": rho})
    return rows, extra


def main():
    jobs = []
    for w in (3, 6, 12, 24, None):
        for s in SEEDS: jobs.append((f"history window = {'all' if w is None else w} storms", s, w is None, dict(window=w)))
    for w in (None, 12):
        for s in SEEDS: jobs.append((f"drift (40% of segments change), window = {'all' if w is None else w}", s, False, dict(window=w, drift_frac=0.4)))
    for s in SEEDS: jobs.append(("reporting bias (quiet roads under-reported)", s, False, dict(report_bias=True)))
    for sg in (0.1, 0.5, 0.8):
        for s in SEEDS: jobs.append((f"forecast noise sigma = {sg}", s, False, dict(fc_sigma=sg)))
    res = Parallel(n_jobs=2)(delayed(one_world)(sc, s, cv, **wk) for sc, s, cv, wk in jobs)
    out = Path(__file__).resolve().parent.parent / "outputs" / "study"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(out / "by_world.csv", index=False)
    ex = pd.DataFrame([e for _, ex in res for e in ex])
    for kind in ("k_curve", "strata", "forecast_corr"):
        ex[ex.kind == kind].dropna(axis=1, how="all").drop(columns="kind").to_csv(out / f"{kind}.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
