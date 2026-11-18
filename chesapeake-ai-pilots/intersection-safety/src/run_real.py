"""python src/run_real.py --crashes crashes.csv --sites sites.csv --mapping data/mapping.json [--train 2019 2020 2021 --test 2022 2023]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import safety as S
from real_data import load_real, load_mapping

ap = argparse.ArgumentParser()
ap.add_argument("--crashes", required=True); ap.add_argument("--sites", required=True); ap.add_argument("--mapping", required=True)
ap.add_argument("--train", type=int, nargs="+", default=[2019, 2020, 2021]); ap.add_argument("--test", type=int, nargs="+", default=[2022, 2023])
ap.add_argument("--top", type=float, default=0.10); ap.add_argument("--boot", type=int, default=500); ap.add_argument("--out", default="outputs_real")
a = ap.parse_args()
S.TRAIN_YEARS, S.TEST_YEARS = tuple(a.train), tuple(a.test)
d, notes = load_real(a.crashes, a.sites, load_mapping(a.mapping), a.train + a.test)
d = S.features(d)
print(f"{len(d)} intersections loaded.");  [print("  note:", n) for n in notes]
if len(d) < 100: print("  WARNING: fewer than 100 sites; results will be very noisy")
sc, k = S.rank_scores(d)
fut = d[[f"crashes_{y}" for y in S.TEST_YEARS]].sum(axis=1).values
rows = [{"ranking": n, "future_crash_share": S.future_share(d, s, a.top)} for n, s in sc.items()]
res = pd.DataFrame(rows).sort_values("future_crash_share", ascending=False)
print(f"\nFlag the top {a.top:.0%} on {a.train} crashes; share of {a.test} crashes at those sites (random = {a.top:.2f}). Overdispersion k = {k:.2f}\n")
print(res.round(3).to_string(index=False))
rng = np.random.default_rng(0); diffs = []
for _ in range(a.boot):                                     # resample sites, re-rank inside each resample
    i = rng.integers(0, len(d), len(d)); sub = d.iloc[i].reset_index(drop=True)
    diffs.append(S.future_share(sub, sc["empirical Bayes"][i], a.top) - S.future_share(sub, sc["raw crash count"][i], a.top))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nEmpirical Bayes minus raw count: {np.mean(diffs):+.3f}  (95% bootstrap interval {lo:+.3f} to {hi:+.3f}; if it spans 0, the data cannot tell them apart)")
pre, post = S.regression_to_mean(d, a.top)
print(f"Regression to the mean: top {a.top:.0%} by history averaged {pre:.2f} crashes/yr then and {post:.2f}/yr later.")
out = Path(a.out); out.mkdir(exist_ok=True); res.to_csv(out / "ranking_results.csv", index=False)
top = d.assign(eb_expected=sc["empirical Bayes"], crashes_train=S.train_total(d), model_expected=sc["model only (no history)"])
top.sort_values("eb_expected", ascending=False).head(max(10, int(len(d) * a.top)))[["site", "aadt", "signal", "crashes_train", "model_expected", "eb_expected"]].round(2).to_csv(out / "sites_to_review.csv", index=False)
print(f"Wrote {out}/ranking_results.csv and sites_to_review.csv")
