"""Load REAL 311/service tickets from a CSV with your own column names.

    python src/run_real.py --tickets tickets.csv --mapping data/mapping.example.json

The label must be the team that FINALLY handled the ticket (not the first assignment, which is what you are trying to improve).
PRIVACY: ticket text often contains names, phone numbers and addresses. Scrub before sharing outputs or sending text to any outside service.
"""
import json
from pathlib import Path
import pandas as pd


def read_mapped(path, mapping, required, optional=()):
    df = pd.read_csv(path, low_memory=False)
    missing = [k for k in required if mapping.get(k) not in df.columns]
    if missing:
        raise ValueError(f"{path}: no column found for {missing}. Mapping says {({k: mapping.get(k) for k in missing})}; file has {list(df.columns)}")
    keep = list(required) + [o for o in optional if mapping.get(o) in df.columns]
    return df[[mapping[k] for k in keep]].set_axis(keep, axis=1)


def load_tickets(path, mapping, min_label_count=30):
    df = read_mapped(path, mapping, ["text", "label", "created", "lat", "lon"], ["incident"])
    notes = []
    df["created"] = pd.to_datetime(df.created, errors="coerce", utc=True)
    df["lat"] = pd.to_numeric(df.lat, errors="coerce"); df["lon"] = pd.to_numeric(df.lon, errors="coerce")
    df["text"] = df.text.fillna("").astype(str).str.strip(); df["label"] = df.label.fillna("").astype(str).str.strip()
    bad = df.created.isna() | (df.text == "") | (df.label == "")
    if bad.any(): notes.append(f"dropped {int(bad.sum())} of {len(df)} tickets with missing text, label or date")
    df = df[~bad].copy()
    nog = df.lat.isna() | df.lon.isna()
    if nog.any(): notes.append(f"{int(nog.sum())} tickets have no usable location (they cannot be matched as duplicates)")
    df = df.sort_values("created").reset_index(drop=True)
    df["hours"] = (df.created - df.created.min()).dt.total_seconds() / 3600
    df["ticket"] = df.index
    has_inc = "incident" in df
    if not has_inc: df["incident"] = df.ticket
    df["category"] = df.label; df["vague"] = False
    counts = df.label.value_counts()
    rare = counts[counts < min_label_count]
    if len(rare): notes.append(f"{len(rare)} labels have fewer than {min_label_count} tickets and will be hard to learn: {dict(rare.head(8))}")
    notes.append(f"{len(df)} tickets, {df.label.nunique()} teams, {df.created.min():%Y-%m-%d} to {df.created.max():%Y-%m-%d}; duplicates {'known' if has_inc else 'NOT known (no incident column)'}")
    return df, notes, has_inc


def load_mapping(path):
    return json.loads(Path(path).read_text())
