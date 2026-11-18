"""python src/run_real.py --storms s.csv --segments g.csv --incidents i.csv --mapping m.json [--k 8]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import flood as F
from real_data import load_real, load_mapping

ap = argparse.ArgumentParser()
ap.add_argument("--storms", required=True); ap.add_argument("--segments", required=True); ap.add_argument("--incidents", required=True)
ap.add_argument("--mapping", required=True); ap.add_argument("--k", type=int, default=8); ap.add_argument("--boot", type=int, default=500); ap.add_argument("--out", default="outputs_real")
a = ap.parse_args()
df, notes = load_real(a.storms, a.segments, a.incidents, load_mapping(a.mapping))
[print("note:", n) for n in notes]
tr, te = F.split_by_storm(df)
print(f"train: first {tr.storm.nunique()} storms; test: last {te.storm.nunique()} storms (never seen in training). Crews check the top {a.k} of {df.segment.nunique()} segments per storm.\n")
scores = {"model (logistic)": F.fit(tr, "logit").predict_proba(te[F.FEATURES])[:, 1], "model (boosted trees)": F.fit(tr).predict_proba(te[F.FEATURES])[:, 1], **F.baselines(te)}
per = {n: F.per_storm_hits(te, s, a.k) for n, s in scores.items()}
rows = []
for n, p in per.items():
    r, fa = F.recall_at_k(te, scores[n], a.k); rows.append({"system": n, "share_of_incidents_caught": r, "false_alerts_per_storm": fa})
res = pd.DataFrame(rows).sort_values("share_of_incidents_caught", ascending=False); print(res.round(3).to_string(index=False))
rng = np.random.default_rng(0); ids = per["model (logistic)"].storm.values; n = len(ids)
def pooled(p, idx): q = p.iloc[idx]; return q.hits.sum() / max(1, q.total.sum())
d = []
for _ in range(a.boot):
    idx = rng.integers(0, n, n); d.append(pooled(per["model (logistic)"], idx) - pooled(per["hotspot history"], idx))
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\nLogistic model minus hotspot-history baseline: {np.mean(d):+.3f}  (95% bootstrap over test storms: {lo:+.3f} to {hi:+.3f}; spanning 0 means no demonstrated gain)")
print(f"Test set has {int(te.incident.sum())} incidents in {te.storm.nunique()} storms, so a few large storms dominate. Check outputs for per-storm results.")
out = Path(a.out); out.mkdir(exist_ok=True); res.to_csv(out / "results.csv", index=False)
per["model (logistic)"].to_csv(out / "per_storm_model.csv", index=False)
