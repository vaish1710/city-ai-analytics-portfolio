"""Load REAL permit submissions from a CSV with your own column names.

    python src/run_real.py --submissions subs.csv --mapping data/mapping.example.json [--rules data/checklists.json]

One row per application. 'files' holds the uploaded file names separated by ';'. Optional 'staff_missing' holds the document types
staff recorded as missing (same ids as checklists.json, ';'-separated; blank = nothing missing). Without it, the run still reports
how many file names are unrecognized, which is the first thing to fix.
PRIVACY: file names can contain people's names. Scrub before sharing outputs.
"""
import json
from pathlib import Path
import pandas as pd

TRUE = {"1", "y", "yes", "true", "t"}


def read_mapped(path, mapping, required, optional=()):
    df = pd.read_csv(path, low_memory=False)
    missing = [k for k in required if mapping.get(k) not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column found for {missing}. Mapping says {({k: mapping.get(k) for k in missing})}; file has {list(df.columns)}")
    keep = list(required) + [o for o in optional if mapping.get(o) in df.columns]
    return df[[mapping[k] for k in keep]].set_axis(keep, axis=1)


def load_submissions(path, mapping, rules):
    notes = []
    df = read_mapped(path, mapping, ["permit_type", "files"], ["staff_missing"])
    fmap = mapping.get("fields", {})
    raw = pd.read_csv(path, low_memory=False)
    out, unknown_types, bad_docs = [], {}, set()
    for i, r in df.iterrows():
        pt = str(r.permit_type).strip()
        if pt not in rules["permits"]:
            unknown_types[pt] = unknown_types.get(pt, 0) + 1
            continue
        files = [f.strip() for f in str(r.files).split(";") if f.strip() and str(r.files) != "nan"]
        fields = {}
        for name, col in fmap.items():
            if col not in raw.columns: raise ValueError(f"{path}: fields mapping {name!r} -> {col!r} not found in {list(raw.columns)}")
            v = raw.loc[i, col]
            fields[name] = float(pd.to_numeric(v, errors="coerce")) if name == "height_ft" else str(v).strip().lower() in TRUE
        truth = None
        if "staff_missing" in df and pd.notna(r.staff_missing):
            truth = sorted({t.strip() for t in str(r.staff_missing).split(";") if t.strip()})
            bad_docs |= {t for t in truth if t not in rules["documents"]}
        elif "staff_missing" in df:
            truth = []
        out.append({"permit_type": pt, "fields": fields, "files": files, "truly_missing": truth})
    if unknown_types: notes.append(f"skipped {sum(unknown_types.values())} applications with permit types not in the checklist file: {unknown_types}")
    if bad_docs: notes.append(f"staff_missing mentions document ids not in the checklist file: {sorted(bad_docs)}")
    nf = sum(1 for o in out if not o["files"])
    if nf: notes.append(f"{nf} applications have no file names")
    has_truth = "staff_missing" in df
    notes.append(f"{len(out)} applications loaded; staff labels {'present' if has_truth else 'NOT present'}")
    return out, notes, has_truth


def load_mapping(path):
    return json.loads(Path(path).read_text())
