"""python src/run_real.py --tickets tickets.csv --mapping mapping.json"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import classification_report
sys.path.insert(0, str(Path(__file__).parent))
import triage as T
from real_data import load_tickets, load_mapping

ap = argparse.ArgumentParser()
ap.add_argument("--tickets", required=True); ap.add_argument("--mapping", required=True); ap.add_argument("--target", type=float, default=0.95)
ap.add_argument("--out", default="outputs_real")
a = ap.parse_args()
m = load_mapping(a.mapping)
df, notes, has_inc = load_tickets(a.tickets, m)
[print("note:", n) for n in notes]
if len(df) < 1000: print("WARNING: fewer than 1,000 tickets; treat results as a rough guide only")
tr, va, te = T.chrono_split(df)
print(f"train {len(tr)} (oldest) / validation {len(va)} / test {len(te)} (newest), split by time")
vec, clf = T.fit_router(tr)
pv, cv = T.predict(vec, clf, va.text); tau, reached = T.choose_threshold(cv, pv == va.label.values, target=a.target)
pt, ct = T.predict(vec, clf, te.text); y = te.label.values; auto = ct >= tau
maj = tr.label.value_counts().index[0]
res = {"majority class ('%s') accuracy" % maj: float((y == maj).mean()), "model, route everything": float((pt == y).mean()),
       "confidence cutoff": tau, f"{a.target:.0%} target reached on validation": "yes" if reached else "NO - strictest cutoff used",
       "model auto-routes (share)": float(auto.mean()), "  accuracy on auto-routed": float((pt[auto] == y[auto]).mean()) if auto.any() else float("nan")}
rules = m.get("keyword_rules")
if rules:
    def kw(t):
        t = t.lower()
        for team, words in rules:
            if any(w in t for w in words): return team
        return "__none__"
    k = te.text.map(kw).values
    res["your keyword rules, correct (no match counts as wrong)"] = float((k == y).mean())
    res["  keyword rules on the auto-routed tickets"] = float((k[auto] == y[auto]).mean()) if auto.any() else float("nan")
else:
    print("note: no keyword_rules in the mapping, so there is no comparison with current practice. Add your intake rules to get one.")
if has_inc:
    truth = T.true_pairs(te)
    for name, f in (("space+time+team", T.candidate_pairs(te, pt, vec)), ("... + text >= 0.15", T.candidate_pairs(te, pt, vec, min_cos=0.15))):
        p, r = T.pr(f, truth); res[f"duplicates {name}: precision / recall"] = f"{p:.2f} / {r:.2f} ({len(f)} flagged, {len(truth)} true)"
print()
for kk, v in res.items(): print(f"{kk:<62} {v:.3f}" if isinstance(v, float) else f"{kk:<62} {v}")
print("\nPer-team results on the test set (precision/recall):"); print(classification_report(y, pt, zero_division=0))
out = Path(a.out); out.mkdir(exist_ok=True)
pd.Series(res).to_csv(out / "results.csv", header=["value"])
pd.DataFrame({"label": y, "suggested": pt, "confidence": ct.round(3), "route": np.where(auto, "auto", "person")}).to_csv(out / "test_predictions.csv", index=False)
print(f"Wrote {out}/results.csv and test_predictions.csv (no ticket text, to keep personal details out)")
