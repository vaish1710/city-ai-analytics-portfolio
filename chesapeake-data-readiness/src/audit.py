"""AI-readiness audit of an ArcGIS MapServer (default: City of Chesapeake open data).

    python src/audit.py                               # audit every layer in the default service
    python src/audit.py --layers 8 15 --sample 3000   # only city facilities and parcels
    python src/audit.py --service http://127.0.0.1:8765/MapServer   # any other server (used by the tests)

Writes to outputs/: dataset_readiness.csv, field_profile.csv, REPORT.md, and appends run_history.csv
so repeated runs show whether counts and freshness drift (a simple reliability monitor).
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from arcgis import ArcGISError, Client
from readiness import WEIGHTS, days_since_edit, profile_fields, score_layer

# The layer ids 8 (City Facilities) and 15 (Parcels) are confirmed by data.gov records for this host and path.
# The service root listing itself has not been confirmed from the build environment.
DEFAULT_SERVICE = "https://gis.cityofchesapeake.net/mapping/rest/services/OpenData/OpenData/MapServer"


def audit(service, sample=2000, seed=42, only=None, client=None, now=None, keep=None):
    """`keep`, if a dict, receives {layer_id: (layer_name, profile, score_result, record_count)} for the AI-docs step."""
    c = client or Client()
    rows, fields = [], []
    for lid, name in c.layers(service):
        if only and lid not in only:
            continue
        url = f"{service}/{lid}"
        try:
            meta = c.meta(url)
            n = c.count(url)
            recs, null_geom, how = c.sample(url, n=sample, seed=seed)
            prof = profile_fields(meta, recs)
            res = score_layer(meta, n, prof, null_geom, now=now)
            if keep is not None:
                keep[lid] = (name, prof, res, n)
            for _, r in prof.iterrows():
                fields.append({"layer_id": lid, "layer": name, **r.to_dict()})
            rows.append({"layer_id": lid, "layer": name, "record_count": n, "fields_profiled": len(prof),
                         "score": res["score"], "readiness_1_to_5": res["readiness_1_to_5"],
                         **{f"c_{k}": round(v, 3) for k, v in res.get("components", {}).items()},
                         "days_since_edit": days_since_edit(meta, now), "key_field": res.get("key_field"),
                         "sampling": how, "flags": res["flags"]})
        except Exception as e:  # keep auditing the other layers
            rows.append({"layer_id": lid, "layer": name, "score": float("nan"), "flags": f"ERROR: {e}"})
    return pd.DataFrame(rows), pd.DataFrame(fields)


def write_outputs(res, fields, out_dir: Path, run_utc: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "dataset_readiness.csv", index=False)
    fields.to_csv(out_dir / "field_profile.csv", index=False)

    hist_path = out_dir / "run_history.csv"
    new = res[["layer_id", "layer", "record_count", "days_since_edit", "score"]].copy()
    new.insert(0, "run_utc", run_utc)
    changes = []
    if hist_path.exists():
        old = pd.read_csv(hist_path)
        prev = old.sort_values("run_utc").groupby("layer_id").tail(1).set_index("layer_id")
        for _, r in new.iterrows():
            if r["layer_id"] in prev.index and pd.notna(r["record_count"]):
                d = r["record_count"] - prev.loc[r["layer_id"], "record_count"]
                if d != 0:
                    changes.append(f"{r['layer']}: record count changed by {int(d):+d} since {prev.loc[r['layer_id'], 'run_utc']}")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(hist_path, index=False)

    ok = res.dropna(subset=["score"]).sort_values("score", ascending=False)
    lines = [f"# Data AI-readiness report\n\nRun: {run_utc} UTC. Scores are 0-100; readiness is the same score on a 1-5 scale "
             "(feeds the 'Data readiness' column of the AI use-case scorecard).\n",
             "Weights (author's assumption): " + ", ".join(f"{k} {v:.0%}" for k, v in WEIGHTS.items()) + "\n",
             "| Layer | Records | Score | 1-5 | Key flags |", "|---|---|---|---|---|"]
    for _, r in ok.iterrows():
        lines.append(f"| {r['layer']} | {int(r['record_count']):,} | {r['score']} | {r['readiness_1_to_5']} | {r['flags'] or 'none'} |")
    bad = res[res["score"].isna()]
    if len(bad):
        lines += ["", "## Layers that could not be audited", *[f"- {r['layer']}: {r['flags']}" for _, r in bad.iterrows()]]
    if changes:
        lines += ["", "## Changes since the previous run", *[f"- {c}" for c in changes]]
    lines += ["", "## Limits", "- Completeness, key uniqueness and geometry are measured on a random sample, not every record; "
              "duplicates in the full layer can be missed.",
              "- A high score means the data is tidy and documented, not that it is correct, unbiased or fit for a given decision.",
              "- Freshness uses the layer's published edit date; many layers publish none.",
              "- Parcels are listed on data.gov under CC BY-NC-ND 4.0 (non-commercial, no derivatives). Check the license of every "
              "dataset before publishing anything derived from it."]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default=DEFAULT_SERVICE)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    ap.add_argument("--ai-docs", action="store_true",
                    help="draft documentation with an LLM (needs `pip install anthropic` and ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    a = ap.parse_args()
    keep = {}
    res, fields = audit(a.service, a.sample, a.seed, set(a.layers) if a.layers else None, keep=keep)
    run = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    changes = write_outputs(res, fields, Path(a.out), run)
    if a.ai_docs:
        from ai_docs import anthropic_llm, draft_layer
        llm = anthropic_llm(a.model)
        for lid, (name, prof, result, n) in keep.items():
            status, bad = draft_layer(lid, name, prof, result, n, llm, Path(a.out), model_label=a.model)
            print(f"AI draft for {name}: {status}")
    show = [c for c in ("layer", "record_count", "score", "readiness_1_to_5", "days_since_edit") if c in res]
    print(res[show].sort_values("score", ascending=False).to_string(index=False))
    for c in changes:
        print("CHANGE:", c)
    print(f"\nwrote {a.out}/REPORT.md, dataset_readiness.csv, field_profile.csv, run_history.csv")


if __name__ == "__main__":
    main()
