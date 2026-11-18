"""python src/run_real.py --submissions subs.csv --mapping mapping.json [--rules checklists.json]"""
import argparse, sys
from collections import Counter
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import permit as P
from real_data import load_submissions, load_mapping

ap = argparse.ArgumentParser()
ap.add_argument("--submissions", required=True); ap.add_argument("--mapping", required=True); ap.add_argument("--rules", default=str(P.DEFAULT_RULES)); ap.add_argument("--out", default="outputs_real")
a = ap.parse_args()
rules = P.load_rules(a.rules)
if "FICTIONAL" in rules.get("_note", ""): print("WARNING: the checklist file is still the FICTIONAL example. Replace data/checklists.json with the real requirements first.\n")
subs, notes, has_truth = load_submissions(a.submissions, load_mapping(a.mapping), rules)
[print("note:", n) for n in notes]
rec = [(s, P.check(s["permit_type"], s["fields"], s["files"], rules), P.naive_check(s["permit_type"], s["fields"], s["files"], rules)) for s in subs]
allfiles = [f for s in subs for f in s["files"]]
kinds = Counter("unknown" if not P.recognize(f, rules) else ("ambiguous" if len(P.recognize(f, rules)) > 1 else "recognized") for f in allfiles)
print(f"\nFile names: {dict(kinds)}  ({kinds['unknown'] / max(1, len(allfiles)):.0%} unrecognized)")
unk = Counter(P.normalize(f) for f in allfiles if not P.recognize(f, rules)).most_common(15)
print("Most common UNRECOGNIZED names (add keywords for these first):"); [print(f"  {n:>5}  {name}") for name, n in unk]
st = Counter(r["status"] for _, r, _ in rec); print("\nStatus:", {k: f"{v / len(rec):.1%}" for k, v in st.items()})
rows = {"applications": len(rec), **{f"status {k}": v / len(rec) for k, v in st.items()}}
if has_truth:
    inc = [(s, r, n) for s, r, n in rec if s["truly_missing"]]
    wrong = sum(len(set(r["missing"]) - set(s["truly_missing"])) for s, r, _ in rec); claims = sum(len(r["missing"]) for _, r, _ in rec)
    nwrong = sum(len(set(n["missing"]) - set(s["truly_missing"])) for s, _, n in rec); nclaims = sum(len(n["missing"]) for _, _, n in rec)
    rows.update({"staff-labelled incomplete": len(inc), "assistant 'missing' claims that staff disagree with": wrong / max(1, claims), "baseline 'missing' claims that staff disagree with": nwrong / max(1, nclaims),
                 "assistant said complete but staff found missing (share of incomplete)": sum(r["status"] == "complete" for _, r, _ in inc) / max(1, len(inc))})
    print(); [print(f"{k:<70} {v:.3f}" if isinstance(v, float) else f"{k:<70} {v}") for k, v in rows.items()]
else:
    print("\nNo staff labels, so accuracy cannot be measured. Add a staff_missing column to evaluate.")
out = Path(a.out); out.mkdir(exist_ok=True); pd.Series(rows).to_csv(out / "results.csv", header=["value"])
pd.DataFrame(unk, columns=["normalized_file_name", "count"]).to_csv(out / "unrecognized_names.csv", index=False)
