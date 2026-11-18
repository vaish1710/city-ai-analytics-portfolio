"""python src/run.py"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import permit as P, simulate as S

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
rules = P.load_rules()
subs = S.make_submissions(rules)
rows = []
for s in subs:
    r = P.check(s["permit_type"], s["fields"], s["files"], rules)
    nv = P.naive_check(s["permit_type"], s["fields"], s["files"], rules)
    truth = set(s["truly_missing"])
    rows.append({"permit": s["permit_type"], "truly_incomplete": bool(truth), "status": r["status"], "naive_status": nv["status"],
                 "claimed_missing": len(r["missing"]), "claimed_missing_wrong": len(set(r["missing"]) - truth),
                 "naive_claimed": len(nv["missing"]), "naive_wrong": len(set(nv["missing"]) - truth), "naive_status_wrong_complete": nv["status"] == "complete" and bool(truth)})
d = pd.DataFrame(rows)
inc = d[d.truly_incomplete]
res = {
    "submissions": len(d), "truly incomplete": int(d.truly_incomplete.sum()),
    "ASSISTANT status: complete": (d.status == "complete").mean(), "  incomplete": (d.status == "incomplete").mean(), "  staff_review": (d.status == "staff_review").mean(),
    "ASSISTANT said 'complete' but it was incomplete (share of incomplete)": ((inc.status == "complete").mean()),
    "ASSISTANT 'missing' claims that were wrong (share of claims)": d.claimed_missing_wrong.sum() / max(1, d.claimed_missing.sum()),
    "BASELINE (no uncertainty handling) 'missing' claims that were wrong": d.naive_wrong.sum() / max(1, d.naive_claimed.sum()),
    "BASELINE said 'complete' but it was incomplete (share of incomplete)": (inc.naive_status == "complete").mean(),
}
s = pd.Series(res)
print("Synthetic submissions with known truth\n")
for k, v in s.items(): print(f"{k:<72} {v:.3f}" if isinstance(v, float) else f"{k:<72} {v}")
s.to_csv(out / "results.csv", header=["value"])
ex = []
for s_ in subs[:6]:
    r = P.check(s_["permit_type"], s_["fields"], s_["files"], rules)
    ex.append(f"--- {s_['permit_type']} | files: {s_['files']} | truly missing: {s_['truly_missing']} | status: {r['status']}\n{P.message(r, s_['permit_type'], rules)}\n")
(out / "example_messages.txt").write_text("\n".join(ex))
