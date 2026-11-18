"""python src/run.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import safety as S

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
rows, rtm = [], []
for hist in ((2019, 2020, 2021), (2021,)):                  # 3 years of history, then only 1
    S.TRAIN_YEARS = hist
    for seed in range(1, 31):
        d = S.make_sites(seed=seed)
        sc, k = S.rank_scores(d)
        for name, s in sc.items():
            rows.append({"history_years": len(hist), "seed": seed, "ranking": name, "future_crash_share_top10pct": S.future_share(d, s), "true_top10_found": S.true_overlap(d, s)})
        if len(hist) == 3:
            rtm.append(S.regression_to_mean(d))
S.TRAIN_YEARS = (2019, 2020, 2021)
df = pd.DataFrame(rows); df.to_csv(out / "results_by_seed.csv", index=False)
paired = []
for h, gh in df.groupby("history_years"):
    p = gh.pivot(index="seed", columns="ranking", values="future_crash_share_top10pct"); dd = p["empirical Bayes"] - p["raw crash count"]
    paired.append(f"{h} yr history: EB minus raw count = {dd.mean():+.4f} (std error {dd.std() / np.sqrt(len(dd)):.4f}); EB better in {(dd > 0).sum()}/{len(dd)} cities")
df3 = df[df.history_years == 3]
g = df3.groupby("ranking").agg(future_share_mean=("future_crash_share_top10pct", "mean"), future_share_min=("future_crash_share_top10pct", "min"),
                              future_share_max=("future_crash_share_top10pct", "max"), true_top10_found=("true_top10_found", "mean")).round(3)
g = g.sort_values("future_share_mean", ascending=False); g.to_csv(out / "summary.csv")
pre, post = np.mean(rtm, axis=0)
print("300 simulated intersections x 30 simulated cities. Flag the top 10% on 2019-21 crashes; judge on 2022-23 crashes.\n")
print(g.to_string())
print("  (a random 10% of sites would capture 0.10 of future crashes)")
print("\nPaired comparison (same simulated cities):"); [print("  " + x) for x in paired]
g1 = df[df.history_years == 1].groupby("ranking").future_crash_share_top10pct.mean().round(3).sort_values(ascending=False)
print("\nWith only 1 year of history, share of future crashes found:"); print(g1.to_string())
print(f"\nRegression to the mean: the 10% of sites with the most 2019-21 crashes averaged {pre:.2f} crashes/yr then and {post:.2f}/yr in 2022-23,"
      f"\nwith nothing done to them. Treating that drop as the effect of a fix would be a mistake.")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
o = g.sort_values("future_share_mean"); fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(o.index, o.future_share_mean, xerr=[o.future_share_mean - o.future_share_min, o.future_share_max - o.future_share_mean], color="#4C78A8", capsize=3)
ax.axvline(0.10, color="gray", ls=":"); ax.set_xlabel("Share of 2022-23 crashes at the flagged 10% (mean; whisker = range over 30 simulated cities; dotted = random)")
ax.set_title("Synthetic data: which ranking finds future crash sites?", fontsize=10); fig.tight_layout(); fig.savefig(out / "ranking_comparison.png", dpi=150)
