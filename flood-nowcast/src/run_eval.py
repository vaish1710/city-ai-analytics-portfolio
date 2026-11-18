"""Train, evaluate and save the nowcast model.

    python src/run_eval.py                                   # synthetic demo (works offline)
    python src/run_eval.py --source noaa --station 8639348 --begin 20240101 --end 20260101 --threshold-ft 4.0

For real data you MUST pass --threshold-ft from the official flood level for the station and datum you use.
Without it a 99th-percentile threshold is used and a warning is printed: that is a stand-in, not a flood level.
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import noaa
from features import make_features
from model import HORIZONS, chrono_split, fit, forecast_series
from sim import make_series

ROOT = Path(__file__).resolve().parent.parent


def load(args):
    if args.source == "synthetic":
        return make_series().drop(columns=["_true_surge"])
    return noaa.build_dataset(args.station, args.begin, args.end, args.datum, with_met=not args.no_met)


def chart(bundle, df, h, path, synthetic):
    _, _, te = chrono_split(len(df))
    test_idx = df.index[te]
    peak = df["observed"].loc[test_idx].idxmax()
    w = slice(peak - pd.Timedelta(days=3), peak + pd.Timedelta(days=3))
    X, _ = make_features(df, h)
    tide = df["predicted"]
    persist = (X["pred_fut"] + X["resid_lag0"]).copy(); persist.index = persist.index + pd.Timedelta(hours=h)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(tide.loc[w], color="#9aa5b1", lw=1.2, label="Published tide only")
    ax.plot(persist.loc[w], color="#3b7dd8", lw=1.2, ls="--", label=f"Persistence ({h} h ahead)")
    ax.plot(forecast_series(bundle, df, h).loc[w], color="#e08a1e", lw=1.8, label=f"Model ({h} h ahead)")
    ax.plot(df["observed"].loc[w], color="black", lw=1.4, label="Observed")
    ax.axhline(bundle["threshold"], color="#c0392b", lw=1, ls=":", label="Flood threshold")
    ax.set_ylabel("Water level (ft)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Largest test-period event: forecasts issued {h} h before each time")
    if synthetic:
        fig.suptitle("SYNTHETIC data: shows the method only", fontsize=8, color="#a33", y=0.99)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "noaa"], default="synthetic")
    ap.add_argument("--station", default="8639348")  # Money Point, VA
    ap.add_argument("--begin", default="20240101")
    ap.add_argument("--end", default="20260101")
    ap.add_argument("--datum", default="MLLW")
    ap.add_argument("--threshold-ft", type=float, default=None)
    ap.add_argument("--no-met", action="store_true", help="skip wind and pressure")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--model-path", default=str(ROOT / "models" / "nowcast.joblib"))
    a = ap.parse_args()

    df = load(a)
    coverage = df["observed"].notna().mean()
    print(f"{len(df)} hourly rows, observed coverage {coverage:.1%}")
    if a.threshold_ft is None:
        a.threshold_ft = float(df["observed"].iloc[: int(len(df) * 0.6)].quantile(0.99))
        print(f"WARNING: no --threshold-ft given; using the 99th percentile of the training period "
              f"({a.threshold_ft:.2f} ft). This is NOT an official flood level.")
    bundle, res = fit(df, a.threshold_ft)
    bundle["meta"] = {"source": a.source, "station": a.station, "datum": a.datum,
                      "met": bool(df.get("wind_speed", pd.Series(dtype=float)).notna().any())}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Path(a.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, a.model_path)
    res.to_csv(out / "metrics.csv", index=False)
    chart(bundle, df, 3, out / "event_h3.png", a.source == "synthetic")
    pd.set_option("display.width", 200)
    print(f"\nTest-period results (threshold {a.threshold_ft:.2f} ft):")
    print(res[res.method != "tide_only"].round(3).to_string(index=False))
    print(f"\nsaved {a.model_path}, {out}/metrics.csv, {out}/event_h3.png")


if __name__ == "__main__":
    main()
