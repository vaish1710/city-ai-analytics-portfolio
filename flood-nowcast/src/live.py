"""Live forecast: fetch the latest NOAA data, forecast 1, 3 and 6 hours ahead, and print an alert status.

    python src/live.py                  # one forecast now
    python src/live.py --watch 900      # repeat every 15 minutes (Ctrl-C to stop)

Needs a model trained on real data first:  python src/run_eval.py --source noaa --threshold-ft <official level>
Status per horizon: ALERT  = model forecast is within its tuned margin of the flood threshold
                    WATCH  = only the cautious (90th percentile) forecast is
This is decision support and a portfolio project, NOT an official warning. Use NWS/NOAA and the city's alerts.
"""
import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

import noaa
from features import make_features
from model import HORIZONS, forecast_row

ROOT = Path(__file__).resolve().parent.parent
MAX_STALE_HOURS = 3


def latest_forecast(bundle, station, datum, now=None, base_url=noaa.BASE, session=None):
    now = (pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)).floor("h")
    df = noaa.build_dataset(station, now - pd.Timedelta(hours=72), now + pd.Timedelta(hours=12), datum,
                            with_met=bundle["meta"]["met"], base_url=base_url, session=session)
    # Real feeds have no future observations; make that explicit so nothing from after `now` can be used.
    future = df.index > now
    for c in ("observed", "wind_speed", "pressure"):
        if c in df:
            df.loc[future, c] = float("nan")
    seen = df["observed"].dropna()
    if seen.empty or now - seen.index[-1] > pd.Timedelta(hours=MAX_STALE_HOURS):
        raise RuntimeError(f"stale or missing observations (latest: {seen.index[-1] if len(seen) else 'none'}); no forecast issued")
    t0 = seen.index[-1]
    thr = bundle["threshold"]
    out = {"issued_utc": str(now), "last_obs_utc": str(t0), "observed_ft": float(seen.iloc[-1]), "threshold_ft": thr,
           "datum": datum, "horizons": {}}
    order = {"none": 0, "WATCH": 1, "ALERT": 2}
    overall = "none"
    for h in HORIZONS:
        X, _ = make_features(df, h)
        row = X.loc[[t0]]
        pf = row["pred_fut"].iloc[0]
        if pd.isna(pf):
            out["horizons"][h] = {"status": "no forecast (tide prediction missing)"}
            continue
        fc = forecast_row(bundle, row, h, pf)
        status = "none"
        if fc["q90"] >= thr - bundle["margins"][(h, "model_q90")]:
            status = "WATCH"
        if fc["mean"] >= thr - bundle["margins"][(h, "model")]:
            status = "ALERT"
        out["horizons"][h] = {"for_utc": str(t0 + pd.Timedelta(hours=h)), "forecast_ft": round(fc["mean"], 2),
                              "upper_ft": round(fc["q90"], 2), "status": status}
        if order.get(status, 0) > order[overall]:
            overall = status
    out["status"] = overall
    return out


def log_row(res, path: Path):
    rows = [{"issued_utc": res["issued_utc"], "last_obs_utc": res["last_obs_utc"], "observed_ft": res["observed_ft"],
             "horizon_h": h, **v} for h, v in res["horizons"].items()]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def show(res):
    print(f"\n{res['issued_utc']} UTC  last obs {res['last_obs_utc']}  {res['observed_ft']:.2f} ft ({res['datum']})  "
          f"threshold {res['threshold_ft']:.2f} ft  OVERALL: {res['status']}")
    for h, v in res["horizons"].items():
        print(f"  +{h} h: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(ROOT / "models" / "nowcast.joblib"))
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    ap.add_argument("--base-url", default=noaa.BASE)
    a = ap.parse_args()
    bundle = joblib.load(a.model_path)
    if bundle["meta"]["source"] == "synthetic":
        print("NOTE: this model was trained on SYNTHETIC data and means nothing for a real station.")
    m = bundle["meta"]
    while True:
        try:
            res = latest_forecast(bundle, m["station"], m["datum"], base_url=a.base_url)
            show(res); log_row(res, ROOT / "outputs" / "live_log.csv")
        except Exception as e:  # keep a watch loop alive through a bad poll
            print(f"{pd.Timestamp.now(tz='UTC')}: no forecast this cycle: {e}")
            if not a.watch:
                raise
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
