"""python src/run.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import triage as T

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
df = T.make_tickets()
tr, va, te = T.chrono_split(df)
vec, clf = T.fit_router(tr)
pv, cv = T.predict(vec, clf, va.text)
tau, reached = T.choose_threshold(cv, pv == va.label.values)
pt, ct = T.predict(vec, clf, te.text)
kw = te.text.map(T.keyword_route).values
auto = ct >= tau
res = {
    "tickets (train/val/test)": f"{len(tr)}/{len(va)}/{len(te)}",
    "keyword rules: correct first assignment": (kw == te.label.values).mean(),
    "model, route everything: correct first assignment": (pt == te.label.values).mean(),
    "confidence cutoff (chosen on validation)": tau,
    "  95% target reached on validation?": "yes" if reached else "NO - used the strictest cutoff",
    "model auto-routes (share of tickets)": auto.mean(),
    "  accuracy on auto-routed": (pt[auto] == te.label.values[auto]).mean(),
    "  sent to a person (share)": 1 - auto.mean(),
    "  keyword-rule accuracy on those same auto-routed": (kw[auto] == te.label.values[auto]).mean(),
    "vague tickets sent to a person (share)": (~auto[te.vague.values]).mean(),
    "clear tickets sent to a person (share)": (~auto[~te.vague.values]).mean(),
}
truth = T.true_pairs(te)
st = T.candidate_pairs(te, pt, vec)
tx = T.candidate_pairs(te, pt, vec, min_cos=0.15)
for name, f in (("space+time+same team", st), ("... plus text similarity>=0.15", tx)):
    p, r = T.pr(f, truth); res[f"duplicates, {name}: precision / recall"] = f"{p:.2f} / {r:.2f}  ({len(f)} pairs flagged, {len(truth)} true)"
s = pd.Series(res)
tab = pd.DataFrame([{"cutoff": c, "auto_routed_share": (ct >= c).mean(), "accuracy_on_auto_routed": (pt[ct >= c] == te.label.values[ct >= c]).mean()}
                    for c in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)]).round(3)
tab.to_csv(out / "coverage_vs_accuracy.csv", index=False)
print("Synthetic tickets, chronological split (train oldest 60%, validate next 15%, test newest 25%)\n")
for k, v in s.items(): print(f"{k:<58} {v:.3f}" if isinstance(v, (float, np.floating)) else f"{k:<58} {v}")
s.to_csv(out / "results.csv", header=["value"])
print("\nCoverage vs accuracy on the test set:"); print(tab.to_string(index=False))
pd.DataFrame({"text": te.text.values, "label": te.label.values, "suggested": pt, "confidence": ct.round(3),
              "route": np.where(auto, "auto", "person")}).head(40).to_csv(out / "example_routing.csv", index=False)
