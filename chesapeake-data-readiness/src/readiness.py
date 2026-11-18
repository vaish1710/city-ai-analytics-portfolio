"""Profile a dataset and compute a transparent AI-readiness score (0-100).

Every weight below is the author's assumption, not a standard. Change them in WEIGHTS and say why.

  completeness  share of non-system fields that are filled in (sample-based)
  key           uniqueness of the best identifier-like field (sample-based, see limits)
  documentation share of fields with a real alias + whether the layer has a description
  freshness     1.0 if edited today, falling linearly to 0 at one year; 0.5 if no edit date is published
  spatial       share of sampled features that have geometry (1.0 for tables with no geometry)
"""
import re
from datetime import datetime, timezone

import pandas as pd

WEIGHTS = {"completeness": 0.35, "key": 0.20, "documentation": 0.15, "freshness": 0.20, "spatial": 0.10}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must total 1.0 or scores can exceed 100"
SYSTEM = ("OBJECTID", "SHAPE", "GLOBALID", "CREATED", "LAST_EDIT", "EDITOR", "CREATOR")
KEY_HINT = re.compile(r"(^|_)(ID|PIN|GPIN|ACCT|ACCOUNT|NUMBER|NO|NUM|CODE)($|_)|ID$", re.I)
STALE_DAYS = 365


def is_system(name: str) -> bool:
    u = name.upper()
    return any(u.startswith(p) or u == p for p in SYSTEM)


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def profile_fields(meta: dict, records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    rows = []
    for f in meta.get("fields", []):
        name = f["name"]
        if is_system(name):
            continue
        col = df[name] if name in df else pd.Series([None] * len(df), dtype=object)
        blank = col.map(_blank)
        nonnull = col[~blank]
        alias = (f.get("alias") or "").strip()
        norm = lambda s: re.sub(r"[\s_]+", "", s).lower()
        rows.append({"field": name, "type": f.get("type", "").replace("esriFieldType", ""),
                     "n_sampled": len(col), "null_share": float(blank.mean()) if len(col) else 1.0,
                     "distinct": int(nonnull.astype(str).nunique()),
                     "constant": bool(nonnull.astype(str).nunique() == 1 and len(nonnull) > 1),
                     "has_real_alias": bool(alias) and norm(alias) != norm(name)})
    return pd.DataFrame(rows)


def best_key(prof: pd.DataFrame):
    """Pick the identifier-like field with the highest uniqueness. Returns (name, unique_share) or (None, None)."""
    best = (None, None)
    for _, r in prof.iterrows():
        nn = r["n_sampled"] * (1 - r["null_share"])
        if nn > 0 and KEY_HINT.search(r["field"]):
            u = r["distinct"] / nn
            if best[1] is None or u > best[1]:
                best = (r["field"], u)
    return best


def days_since_edit(meta: dict, now=None):
    ei = meta.get("editingInfo") or {}
    ms = ei.get("lastEditDate") or ei.get("dataLastEditDate")
    if not ms:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - datetime.fromtimestamp(ms / 1000, tz=timezone.utc)).days


def score_layer(meta, count, prof, null_geom, now=None) -> dict:
    flags = []
    if count == 0 or prof.empty:
        return {"score": 0.0, "readiness_1_to_5": 1.0, "flags": "EMPTY or no usable fields", "components": {}}
    comp = {"completeness": float((1 - prof["null_share"]).mean())}
    k, u = best_key(prof)
    if k is None:
        comp["key"] = 0.5
        flags.append("no identifier-like field found (key credit set to 0.5)")
    else:
        comp["key"] = float(min(u, 1.0))
        if u < 0.99:
            flags.append(f"key '{k}' only {u:.1%} unique in sample")
    desc = bool((meta.get("description") or "").strip())
    comp["documentation"] = 0.5 * float(prof["has_real_alias"].mean()) + 0.5 * float(desc)
    if not desc:
        flags.append("no layer description")
    d = days_since_edit(meta, now)
    if d is None:
        comp["freshness"] = 0.5
        flags.append("no edit date published: freshness cannot be verified")
    else:
        comp["freshness"] = max(0.0, 1 - d / STALE_DAYS)
        if d > STALE_DAYS:
            flags.append(f"not edited for {d} days")
    has_geom = meta.get("geometryType") is not None
    comp["spatial"] = (1 - null_geom) if (has_geom and null_geom is not None) else 1.0
    if has_geom and null_geom:
        flags.append(f"{null_geom:.0%} of sampled features have no geometry")
    hi = prof[prof["null_share"] > 0.5]["field"].tolist()
    if hi:
        flags.append("over half empty: " + ", ".join(hi[:6]) + ("..." if len(hi) > 6 else ""))
    const = prof[prof["constant"]]["field"].tolist()
    if const:
        flags.append("constant value: " + ", ".join(const[:6]))
    score = 100 * sum(WEIGHTS[c] * comp[c] for c in WEIGHTS)
    return {"score": round(score, 1), "readiness_1_to_5": round(1 + 4 * score / 100, 2),
            "flags": "; ".join(flags), "components": comp, "key_field": k}
