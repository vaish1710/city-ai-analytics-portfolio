"""Permit completeness check: which required documents are missing, and what a person should look at.
Filename-based recognition + a checklist. Staff keep every decision. Checklist data here is FICTIONAL."""
import json
import re
from pathlib import Path

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "data" / "checklists.json"


def load_rules(path=DEFAULT_RULES):
    rules = json.loads(Path(path).read_text())
    for t, d in rules["documents"].items():
        d["_re"] = [re.compile(k) for k in d["keywords"]]
    return rules


def normalize(name):
    stem = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", name.lower())
    return re.sub(r"[_\-\.]+", " ", stem)


def recognize(filename, rules):
    """Document types whose keywords appear in the filename. 0 = unknown, 1 = recognized, 2+ = ambiguous."""
    n = normalize(filename)
    return [t for t, d in rules["documents"].items() if any(r.search(n) for r in d["_re"])]


def required_docs(permit_type, fields, rules):
    p = rules["permits"][permit_type]
    req = list(p["required"])
    for c in p["conditional"]:
        v = fields.get(c["field"])
        hit = (v is not None and v > c["value"]) if c["op"] == "gt" else (v == c["value"])
        if hit and c["require"] not in req:
            req.append(c["require"])
    return req


def check(permit_type, fields, filenames, rules):
    """Status: 'complete' | 'incomplete' | 'staff_review'.
    A required document is CONFIRMED only if some file matches that type and nothing else.
    If it is not confirmed it is MISSING only when no unknown or ambiguous file could be it; otherwise it is UNCERTAIN."""
    req = required_docs(permit_type, fields, rules)
    matches = [recognize(f, rules) for f in filenames]
    confirmed = {m[0] for m in matches if len(m) == 1}
    unknown = sum(1 for m in matches if not m)
    missing, uncertain = [], []
    for t in req:
        if t in confirmed:
            continue
        could_be = unknown > 0 or any(len(m) > 1 and t in m for m in matches)
        (uncertain if could_be else missing).append(t)
    status = "incomplete" if missing else ("staff_review" if uncertain else "complete")
    return {"status": status, "required": req, "missing": missing, "uncertain": uncertain,
            "unrecognized_files": [f for f, m in zip(filenames, matches) if not m],
            "ambiguous_files": [f for f, m in zip(filenames, matches) if len(m) > 1]}


def naive_check(permit_type, fields, filenames, rules):
    """Baseline without uncertainty handling: any required type not matched by some file is 'missing'."""
    req = required_docs(permit_type, fields, rules)
    seen = {t for f in filenames for t in recognize(f, rules)}
    missing = [t for t in req if t not in seen]
    return {"status": "incomplete" if missing else "complete", "missing": missing}


def message(result, permit_type, rules):
    """Draft text for the applicant. Staff review it before it is sent."""
    lab = lambda t: rules["documents"][t]["label"]
    if result["status"] == "complete":
        return "All required documents for your %s permit appear to be attached. Staff will complete their review." % rules["permits"][permit_type]["label"].lower()
    lines = ["Your %s permit application is missing:" % rules["permits"][permit_type]["label"].lower()] + [" - " + lab(t) for t in result["missing"]]
    if result["uncertain"]:
        lines += ["We could not tell from the file names whether you attached:"] + [" - " + lab(t) for t in result["uncertain"]]
        lines += ["Please give files descriptive names (for example 'Site Plan.pdf')."]
    return "\n".join(lines)
