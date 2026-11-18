"""Who pays for Chesapeake's multi-year water/sewer rate increases?

Compares three ways of raising the SAME revenue and shows the effect by income group.

  A  as approved       : every charge rises 4.9% a year (2026-2032)
  B  lifeline blocks   : fixed charge and first block rise at half the pace; the upper block
                         is solved each year so total revenue equals A (revenue-neutral)
  C  assistance credit : same increase as A, plus a bill credit for the lowest-income group,
                         paid for by a uniform surcharge on everyone else (revenue-neutral)

REAL inputs (source: council vote summary, July 2025, check the meeting video):
  * 4.9% annual increase 2026-2027, projected 4.9% a year through 2032
  * a typical household uses 5,000 gallons a month
PLACEHOLDERS you must replace (the tariff and the household data are NOT real):
  * BASE_FIXED, BASE_PER_KGAL: calibrated roughly to one unreliable staff figure
  * the household income and usage distributions are synthetic
Run:  python src/analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RATE = 0.049                 # annual increase (real, see above)
YEARS = list(range(2025, 2033))
BASE_FIXED = 20.0            # $ per bimonthly bill   (PLACEHOLDER)
BASE_PER_KGAL = 4.30         # $ per 1,000 gal        (PLACEHOLDER)
BLOCK1_KGAL = 6.0            # lifeline block size per bimonthly bill (design choice)
LIFELINE_GROWTH = RATE / 2   # growth of fixed + block 1 under option B
CREDIT = 0.25                # share of bill credited under option C
THRESHOLDS = (0.02, 0.045)   # burden screening levels as share of income (verify before citing)
QUARTILES = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]


def make_households(n=20000, seed=7, median_income=85000):
    """SYNTHETIC households. Replace with ACS tract incomes + the city's billing distribution."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(np.log(median_income), 0.65, n)
    kgal = 10 * (income / median_income) ** 0.25 * rng.lognormal(0, 0.45, n)  # bimonthly usage; 10 = 5,000 gal/mo
    df = pd.DataFrame({"income": income, "kgal": kgal})
    df["quartile"] = pd.qcut(df["income"], 4, labels=QUARTILES)
    return df


def tariff_A(df, t):
    g = (1 + RATE) ** t
    return (BASE_FIXED + BASE_PER_KGAL * df["kgal"]) * g


def tariff_B(df, t, revenue_target):
    fixed = BASE_FIXED * (1 + LIFELINE_GROWTH) ** t
    r1 = BASE_PER_KGAL * (1 + LIFELINE_GROWTH) ** t
    u1 = df["kgal"].clip(upper=BLOCK1_KGAL)
    u2 = (df["kgal"] - BLOCK1_KGAL).clip(lower=0)
    r2 = (revenue_target - fixed * len(df) - r1 * u1.sum()) / u2.sum()  # revenue-neutral upper block
    return fixed + r1 * u1 + r2 * u2, r2


def tariff_C(df, t):
    base = tariff_A(df, t)
    recip = df["quartile"] == "Q1 lowest"
    credit_total = (CREDIT * base[recip]).sum()
    surcharge = credit_total / base[~recip].sum()  # uniform % on non-recipients
    return np.where(recip, base * (1 - CREDIT), base * (1 + surcharge)), surcharge


def run(df):
    rows = []
    for t, year in enumerate(YEARS):
        a = tariff_A(df, t)
        b, r2 = tariff_B(df, t, a.sum())
        c, _ = tariff_C(df, t)
        for name, bill in [("A as approved", a), ("B lifeline blocks", b), ("C assistance credit", c)]:
            ann = np.asarray(bill) * 6
            burden = ann / df["income"].values
            d = df.assign(burden=burden, annual=ann)
            for q, g in d.groupby("quartile", observed=True):
                rows.append({"year": year, "option": name, "quartile": q,
                             "mean_annual_bill": g["annual"].mean(),
                             "mean_burden_pct": 100 * g["burden"].mean(),
                             "share_over_2pct": (g["burden"] > THRESHOLDS[0]).mean(),
                             "share_over_4_5pct": (g["burden"] > THRESHOLDS[1]).mean()})
            rows.append({"year": year, "option": name, "quartile": "ALL (revenue $)",
                         "mean_annual_bill": np.asarray(bill).sum() * 6, "mean_burden_pct": np.nan,
                         "share_over_2pct": np.nan, "share_over_4_5pct": np.nan})
    return pd.DataFrame(rows)


def checks(df, res):
    rev = res[res.quartile == "ALL (revenue $)"].pivot(index="year", columns="option", values="mean_annual_bill")
    assert np.allclose(rev["A as approved"], rev["B lifeline blocks"], rtol=1e-9), "B not revenue-neutral"
    assert np.allclose(rev["A as approved"], rev["C assistance credit"], rtol=1e-9), "C not revenue-neutral"
    growth = rev["A as approved"].iloc[-1] / rev["A as approved"].iloc[0]
    assert np.isclose(growth, (1 + RATE) ** (len(YEARS) - 1)), "compounding wrong"
    print(f"checks passed: B and C raise the same revenue as A every year; A compounds to x{growth:.3f} by 2032")


def chart(res):
    last = res[(res.year == YEARS[-1]) & (res.quartile.isin(QUARTILES))]
    first = res[(res.year == YEARS[0]) & (res.option == "A as approved") & (res.quartile.isin(QUARTILES))]
    colors = {"A as approved": "#9aa5b1", "B lifeline blocks": "#3b7dd8", "C assistance credit": "#e08a1e"}
    fig, ax = plt.subplots(figsize=(8.5, 4.4)); x = np.arange(4); w = 0.26
    ax.scatter(x, first.set_index("quartile").loc[QUARTILES, "mean_burden_pct"], marker="_", s=900,
               color="k", linewidths=2, zorder=3, label="2025 baseline")
    for i, (opt, g) in enumerate(last.groupby("option")):
        ax.bar(x + (i - 1) * w, g.set_index("quartile").loc[QUARTILES, "mean_burden_pct"], w, label=opt, color=colors[opt])
    ax.set_xticks(x, QUARTILES); ax.set_ylabel("Water + sewer bill, % of household income")
    ax.set_title("2032 bill burden by income group (same total revenue)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SYNTHETIC households and placeholder tariff: shows method only", fontsize=8, color="#a33", y=0.99)
    fig.tight_layout(); fig.savefig(ROOT / "outputs" / "burden_by_income_2032.png", dpi=160)


if __name__ == "__main__":
    hh = make_households()
    res = run(hh)
    checks(hh, res)
    (ROOT / "outputs").mkdir(exist_ok=True)
    res.to_csv(ROOT / "outputs" / "burden_results.csv", index=False)
    chart(res)
    pv = res[(res.year == 2032) & res.quartile.isin(QUARTILES)].pivot(
        index="quartile", columns="option", values="mean_burden_pct").loc[QUARTILES]
    print("\nMean bill burden in 2032 (% of income):\n", pv.round(2).to_string())
    base = res[(res.year == 2025) & (res.option == "A as approved") & res.quartile.isin(QUARTILES)].set_index("quartile")
    print("\n2025 baseline (% of income):\n", base.loc[QUARTILES, "mean_burden_pct"].round(2).to_string())
