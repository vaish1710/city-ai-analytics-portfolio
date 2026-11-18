"""End-to-end run: data -> model -> prioritisation -> outputs.

Usage (from project root):  python src/run.py [--data path/to/parcels.csv] [--budget 25000000]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_data import make_parcels
from model import cross_validate_by_neighborhood, fit_and_score
from prioritize import add_benefit_cost, compare_rules

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Parcel CSV with the same columns as the synthetic file")
    ap.add_argument("--budget", type=float, default=25_000_000)
    ap.add_argument("--physical", action="store_true", help="force physical EAL even if loss labels exist")
    args = ap.parse_args()

    synthetic = args.data is None
    df = make_parcels() if synthetic else pd.read_csv(args.data)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    if synthetic:
        df.to_csv(ROOT / "data" / "parcels_synthetic.csv", index=False)

    # Real data has no parcel-level loss labels, so estimate loss physically (see physical.py).
    # Train the ML model only when a real observed_annual_loss column is supplied.
    has_labels = "observed_annual_loss" in df and df["observed_annual_loss"].notna().any()
    if args.physical or not has_labels:
        from physical import estimate_eal
        mode = "physical"
        cv = None
        scored = df.copy()
        scored["pred_annual_loss"] = estimate_eal(df)
        print("Mode: physical depth-damage EAL (no ML; no parcel-level labels)")
        # physical.py invents BFE = ground + 1 ft for flood-zone parcels without a published BFE,
        # which makes their risk independent of elevation. Say so, and how many are affected.
        sfha = df["fema_zone"].astype(str).str.upper().str.match(r"^(A|V)")
        no_bfe = sfha & (df["bfe_ft"].isna() if "bfe_ft" in df else True)
        if no_bfe.any():
            print(f"WARNING: {int(no_bfe.sum())} of {int(sfha.sum())} flood-zone parcels have no BFE; "
                  "their loss uses an assumed BFE and is NOT elevation-sensitive. Treat their ranks as unreliable.")
        print(f"Parcels in non-A/V zones (e.g. X-shaded) get zero modelled loss: {int((~sfha).sum())}")
    else:
        mode = "ml"
        cv = cross_validate_by_neighborhood(df)
        print("Neighbourhood-held-out validation\n", cv.round(3).to_string(index=False))
        scored, _ = fit_and_score(df)
    scored = add_benefit_cost(scored)
    summary, picks = compare_rules(scored, args.budget)
    print("\nRule comparison (budget ${:,.0f})\n".format(args.budget), summary.round(3).to_string(index=False))

    # Sanity check available only on synthetic data: does the model recover the hidden truth?
    if "_true_annual_loss" in scored:
        corr = np.corrcoef(scored["pred_annual_loss"], scored["_true_annual_loss"])[0, 1]
        print(f"\n[synthetic only] correlation with hidden true loss: {corr:.3f}")

    best = picks["bcr_equity"].sort_values("score", ascending=False)
    cols = ["parcel_id", "neighborhood", "income_quartile", "elevation_ft", "pred_annual_loss",
            "cost_usd", "benefit_usd", "bcr", "score"]
    best[cols].round(2).to_csv(ROOT / "outputs" / "ranked_parcels.csv", index=False)
    summary.to_csv(ROOT / "outputs" / "rule_comparison.csv", index=False)
    if cv is not None:
        cv.to_csv(ROOT / "outputs" / "validation.csv", index=False)
    else:  # don't leave a stale validation file from an earlier ML run next to physical results
        (ROOT / "outputs" / "validation.csv").unlink(missing_ok=True)
    (ROOT / "outputs" / "summary.json").write_text(json.dumps({
        "synthetic_data": synthetic, "mode": mode, "budget": args.budget, "rows": len(df),
        "cv_mean_r2": None if cv is None else float(cv["r2"].mean()),
        "cv_mean_mae_usd": None if cv is None else float(cv["mae_usd"].mean()),
    }, indent=2))

    # Figure: who gets funded under each rule
    order = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    labels = {"highest_loss": "Highest loss first", "bcr": "Benefit-cost", "bcr_equity": "Benefit-cost + equity"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"highest_loss": "#9aa5b1", "bcr": "#3b7dd8", "bcr_equity": "#e08a1e"}
    x = np.arange(len(order)); w = 0.26
    for i, (rule, c) in enumerate(picks.items()):
        counts = c["income_quartile"].value_counts().reindex(order, fill_value=0)
        ax[0].bar(x + (i - 1) * w, counts.values, w, label=labels[rule], color=colors[rule])
    ax[0].set_xticks(x, order); ax[0].set_ylabel("Homes funded"); ax[0].set_title("Funded homes by income quartile")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].bar([labels[r] for r in summary["rule"]], summary["avoided_annual_loss_usd"] / 1e6,
              color=[colors[r] for r in summary["rule"]])
    ax[1].set_ylabel("Avoided annual loss ($M)"); ax[1].set_title("Loss avoided per rule (same budget)")
    ax[1].tick_params(axis="x", labelrotation=15, labelsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    note = "SYNTHETIC DATA: illustrates method only" if synthetic else "Real data run"
    fig.suptitle(note, fontsize=9, color="#a33", y=0.99)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "rule_comparison.png", dpi=160)
    print("\nwrote outputs/ranked_parcels.csv, rule_comparison.csv/png, validation.csv, summary.json")


if __name__ == "__main__":
    main()
