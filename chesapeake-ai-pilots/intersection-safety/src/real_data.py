"""Load REAL crash records + intersection attributes from CSV files with your own column names.

    python src/run_real.py --crashes crashes.csv --sites sites.csv --mapping data/mapping.example.json

Crash file: one row per crash. Sites file: one row per intersection. The mapping JSON says which of YOUR columns is which.
Everything dropped or suspicious is reported, because real data is where the surprises are.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TRUE = {"1", "y", "yes", "true", "t", "signal", "signalized"}


def read_mapped(path, mapping, required, optional=()):
    df = pd.read_csv(path, low_memory=False)
    missing = [k for k in required if mapping.get(k) not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column found for {missing}. Mapping says {({k: mapping.get(k) for k in missing})}; file has {list(df.columns)}")
    keep = list(required) + [o for o in optional if mapping.get(o) in df.columns]
    return df[[mapping[k] for k in keep]].set_axis(keep, axis=1)


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(TRUE).astype(int)


def load_real(crashes_csv, sites_csv, mapping, years):
    """Returns (frame with crashes_<year> columns, list of notes about what was dropped or looks wrong)."""
    notes = []
    s = read_mapped(sites_csv, mapping["sites"], ["site", "aadt", "signal", "legs", "speed"])
    c = read_mapped(crashes_csv, mapping["crashes"], ["site", "date"], ["severity"])
    keep = mapping["crashes"].get("severity_keep")
    if keep and "severity" in c:
        n0 = len(c); c = c[c.severity.astype(str).isin([str(k) for k in keep])]
        notes.append(f"kept severities {keep}: {len(c)} of {n0} crash rows")
    n0 = len(s); s = s.drop_duplicates("site")
    if len(s) < n0: notes.append(f"dropped {n0 - len(s)} duplicate site rows")
    s["aadt"] = pd.to_numeric(s.aadt, errors="coerce"); s["speed"] = pd.to_numeric(s.speed, errors="coerce"); s["legs"] = pd.to_numeric(s.legs, errors="coerce")
    bad = s.aadt.isna() | (s.aadt <= 0) | s.speed.isna() | s.legs.isna()
    if bad.any(): notes.append(f"dropped {int(bad.sum())} sites with missing/zero volume, speed or legs")
    s = s[~bad].copy()
    s["signal"] = to_bool(s.signal); s["legs4"] = (s.legs >= 4).astype(int)
    c["year"] = pd.to_datetime(c.date, errors="coerce").dt.year
    if c.year.isna().any(): notes.append(f"dropped {int(c.year.isna().sum())} crashes with unreadable dates")
    c = c.dropna(subset=["year"])
    unmatched = ~c.site.isin(s.site)
    if unmatched.any(): notes.append(f"dropped {int(unmatched.sum())} crashes at sites not in the sites file")
    c = c[~unmatched]
    counts = c.groupby(["site", "year"]).size().unstack(fill_value=0)
    out = s[["site", "aadt", "signal", "legs4", "speed"]].reset_index(drop=True)
    for y in years:
        col = counts[y] if y in counts.columns else pd.Series(0, index=counts.index)
        out[f"crashes_{y}"] = out.site.map(col).fillna(0).astype(int)
    totals = {y: int(out[f"crashes_{y}"].sum()) for y in years}
    typical = float(np.median(list(totals.values())))
    for y, t in totals.items():
        if t == 0: notes.append(f"WARNING: no crashes at all in {y}; that year is missing or incomplete")
        elif t < 0.5 * typical: notes.append(f"WARNING: {y} has far fewer crashes ({t}) than typical ({typical:.0f}); check for an incomplete year")
    return out, notes


def load_mapping(path):
    return json.loads(Path(path).read_text())
