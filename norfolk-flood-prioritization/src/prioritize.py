"""Choose which homes to elevate under a fixed budget, comparing three rules."""
import numpy as np
import pandas as pd

HORIZON_YEARS = 30
DISCOUNT = 0.03
ANNUITY = (1 - (1 + DISCOUNT) ** -HORIZON_YEARS) / DISCOUNT
LOSS_REDUCTION = 0.85  # share of expected loss removed by elevation (assumption)


def elevation_cost(df: pd.DataFrame) -> pd.Series:
    """Rough cost model (assumption, replace with program bid data)."""
    per_sqft = df["foundation"].map({"slab": 140, "crawlspace": 95, "basement": 120}).fillna(110)
    return 45000 + per_sqft * df["sqft"]


def add_benefit_cost(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_usd"] = elevation_cost(out)
    out["benefit_usd"] = out["pred_annual_loss"] * LOSS_REDUCTION * ANNUITY
    out["bcr"] = out["benefit_usd"] / out["cost_usd"]
    inc = out["tract_median_income"]
    out["income_quartile"] = pd.qcut(inc, 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    # Equity weight: up to 1.5x priority for the lowest-income homes, tapering to 1.0x.
    out["equity_weight"] = 1.5 - 0.5 * (inc.rank(pct=True))
    return out


def select(df: pd.DataFrame, budget: float, rule: str) -> pd.DataFrame:
    """Greedy selection by score per dollar until the budget is spent."""
    d = df[df["bcr"] > 1.0].copy()  # only fund projects whose benefits exceed costs
    if rule == "highest_loss":
        d["score"] = d["pred_annual_loss"]  # naive baseline: worst-hit homes first, ignores cost
    elif rule == "bcr":
        d["score"] = d["bcr"]
    elif rule == "bcr_equity":
        d["score"] = d["bcr"] * d["equity_weight"]
    else:
        raise ValueError(rule)
    d = d.sort_values("score", ascending=False)
    d["cum_cost"] = d["cost_usd"].cumsum()
    chosen = d[d["cum_cost"] <= budget].copy()
    chosen["rule"] = rule
    return chosen


def compare_rules(df: pd.DataFrame, budget: float) -> tuple[pd.DataFrame, dict]:
    rows, picks = [], {}
    for rule in ["highest_loss", "bcr", "bcr_equity"]:
        c = select(df, budget, rule)
        picks[rule] = c
        share_q1 = (c["income_quartile"] == "Q1 lowest").mean() if len(c) else np.nan
        rows.append({
            "rule": rule,
            "homes_funded": len(c),
            "spent_usd": c["cost_usd"].sum(),
            "net_benefit_usd": (c["benefit_usd"] - c["cost_usd"]).sum(),
            "avoided_annual_loss_usd": (c["pred_annual_loss"] * LOSS_REDUCTION).sum(),
            "share_lowest_income_quartile": share_q1,
        })
    return pd.DataFrame(rows), picks
