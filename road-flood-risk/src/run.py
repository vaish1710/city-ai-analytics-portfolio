"""python src/run.py   -> results over 10 simulated cities, chart, and an example flagged list."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import flood

ROOT = Path(__file__).resolve().parent.parent
rows = []
for n_storms in (80, 200):
    for seed in range(1, 11):
        for name, m in flood.evaluate(seed, n_storms=n_storms).items():
            rows.append({"history_storms": n_storms, "seed": seed, "system": name, **m})
df = pd.DataFrame(rows)
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
df.to_csv(out / "results_by_seed.csv", index=False)
s = df.groupby(["history_storms", "system"]).agg(recall_mean=("recall_at_k", "mean"), recall_min=("recall_at_k", "min"),
                              recall_max=("recall_at_k", "max"), false_alerts=("false_alerts_per_storm", "mean")).round(3)
s.to_csv(out / "summary.csv")
print("Share of reported flood incidents caught when crews check the top 8 of 60 segments per storm")
print("(synthetic data, 10 simulated cities per row, last 30% of storms held out)\n"); print(s.sort_values(["history_storms", "recall_mean"], ascending=[True, False]))

# example: flagged list for one held-out storm, with reasons
d = flood.make_data(seed=1); tr, te = flood.split_by_storm(d); m = flood.fit(tr, 'logit'); ref = flood.ref_thresholds(tr)
st = te.storm.min(); g = te[te.storm == st].copy(); g["risk"] = m.predict_proba(g[flood.FEATURES])[:, 1]
g = g.sort_values("risk", ascending=False).head(8)
g["why"] = [", ".join(flood.reasons(r, ref)) for r in g.itertuples()]
g[["storm", "segment", "risk", "incident", "why"]].round(3).to_csv(out / "example_storm_flags.csv", index=False)
print(f"\nExample, storm {st} (incident=1 means it was reported):"); print(g[["segment", "risk", "incident", "why"]].round(2).to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
o = s.loc[200].sort_values("recall_mean"); fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(o.index, o.recall_mean, xerr=[o.recall_mean - o.recall_min, o.recall_max - o.recall_mean], color="#4C78A8", capsize=3)
ax.set_xlabel("Share of incidents caught (bar = mean, whisker = range over 10 simulated cities)"); ax.set_xlim(0, 1)
ax.set_title("Synthetic data, 200 storms of history: top-8 segments per storm", fontsize=10); fig.tight_layout(); fig.savefig(out / "recall.png", dpi=150)
