"""Load REAL storms, road segments and flood incident reports from three CSV files with your own column names.

    python src/run_real.py --storms storms.csv --segments segments.csv --incidents incidents.csv --mapping data/mapping.example.json

IMPORTANT: peak_fc / total_fc / tide_fc must be FORECAST values that were available BEFORE each storm. Using observed rainfall
instead makes the results look much better than the tool could really do.
"""
import json
from pathlib import Path
import pandas as pd
from flood import add_prior


def read_mapped(path, mapping, required, optional=()):
    df = pd.read_csv(path, low_memory=False)
    missing = [k for k in required if mapping.get(k) not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column found for {missing}. Mapping says {({k: mapping.get(k) for k in missing})}; file has {list(df.columns)}")
    keep = list(required) + [o for o in optional if mapping.get(o) in df.columns]
    return df[[mapping[k] for k in keep]].set_axis(keep, axis=1)


def load_real(storms_csv, segments_csv, incidents_csv, mapping):
    notes = []
    st = read_mapped(storms_csv, mapping["storms"], ["storm", "start", "peak_fc", "total_fc"], ["tide_fc"])
    sg = read_mapped(segments_csv, mapping["segments"], ["segment", "elev_ft", "drain_dist_ft"], ["tidal"])
    inc = read_mapped(incidents_csv, mapping["incidents"], ["storm", "segment"])
    st["start"] = pd.to_datetime(st.start, errors="coerce", utc=True)
    for c in ("peak_fc", "total_fc") + (("tide_fc",) if "tide_fc" in st else ()):
        st[c] = pd.to_numeric(st[c], errors="coerce")
    bad = st.isna().any(axis=1)
    if bad.any(): notes.append(f"dropped {int(bad.sum())} storms with missing date or forecast values")
    st = st[~bad].drop_duplicates("storm").sort_values("start").reset_index(drop=True)
    st["storm_idx"] = st.index
    if "tide_fc" not in st: st["tide_fc"] = 0.0; notes.append("no tide forecast column: tide_fc set to 0 (tidal roads will not be modeled)")
    for c in ("elev_ft", "drain_dist_ft"):
        sg[c] = pd.to_numeric(sg[c], errors="coerce")
    badg = sg[["elev_ft", "drain_dist_ft"]].isna().any(axis=1)
    if badg.any(): notes.append(f"dropped {int(badg.sum())} segments missing elevation or drainage distance")
    sg = sg[~badg].drop_duplicates("segment").reset_index(drop=True)
    sg["tidal"] = to_int(sg["tidal"]) if "tidal" in sg else 0
    sg["seg_idx"] = sg.index
    inc = inc.drop_duplicates()
    ok = inc.storm.isin(st.storm) & inc.segment.isin(sg.segment)
    if (~ok).any(): notes.append(f"dropped {int((~ok).sum())} incident rows referencing unknown storms or segments")
    inc = inc[ok].assign(incident=1)
    full = st.merge(sg, how="cross").merge(inc, on=["storm", "segment"], how="left")
    full["incident"] = full.incident.fillna(0).astype(int)
    df = full.rename(columns={"storm_idx": "storm_i", "seg_idx": "seg_i"})[["storm_i", "seg_i", "elev_ft", "drain_dist_ft", "tidal", "peak_fc", "total_fc", "tide_fc", "incident"]]
    df = df.rename(columns={"storm_i": "storm", "seg_i": "segment"})
    df = add_prior(df)
    per = df.groupby("storm").incident.sum()
    notes.append(f"{len(st)} storms x {len(sg)} segments; {int(df.incident.sum())} reported incidents ({df.incident.mean():.1%} of storm-segment pairs); {int((per == 0).sum())} storms had none")
    if len(st) < 40: notes.append("WARNING: fewer than 40 storms; the model will have very little to learn from")
    if df.incident.sum() < 100: notes.append("WARNING: fewer than 100 incidents; expect very noisy results")
    return df, notes


def to_int(s):
    return s.astype(str).str.strip().str.lower().isin({"1", "y", "yes", "true", "t"}).astype(int)


def load_mapping(path):
    return json.loads(Path(path).read_text())
