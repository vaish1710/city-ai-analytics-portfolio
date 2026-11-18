"""Build storms.csv for road-flood-risk from the collected snapshots, observed rain and NOAA tides.

    python src/build_storm_table.py --db data/chesapeake.db --begin 2026-01-01 --end 2026-11-01 --out data/
Observed rain: from the collector's database (default), or an hourly CSV (--rain-csv with columns `time`, `precip_in`; `time` = END of
the hour). If the CSV is in local time, pass --rain-csv-tz (e.g. America/New_York); it is converted to UTC.
Forecast snapshots exist only from the day the collector started, so storms before that have no forecast and are excluded.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import noaa, storms, store


def rain_from_csv(path, tz="UTC"):
    df = pd.read_csv(path)
    for c in ("time", "precip_in"):
        if c not in df.columns: raise ValueError(f"{path}: needs columns 'time' and 'precip_in'; has {list(df.columns)}")
    t = pd.to_datetime(df.time, errors="coerce")
    t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") if t.dt.tz is None else t.dt.tz_convert(tz)
    s = pd.Series(pd.to_numeric(df.precip_in, errors="coerce").values, index=t.dt.tz_convert("UTC")).dropna()
    s = s[~s.index.isna()]
    return s.groupby(level=0).sum().sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--begin", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--tide-station", default="8638610"); ap.add_argument("--datum", default="MLLW"); ap.add_argument("--base-url", default=noaa.BASE)
    ap.add_argument("--rain-csv"); ap.add_argument("--rain-csv-tz", default="UTC"); ap.add_argument("--rain-station")
    ap.add_argument("--lead-hours", type=float, default=6); ap.add_argument("--tide-threshold-ft", type=float, default=None); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    conn = store.connect(a.db)
    rain = rain_from_csv(a.rain_csv, a.rain_csv_tz) if a.rain_csv else store.hourly_rain_inches(conn, a.rain_station)
    rain = rain.loc[pd.Timestamp(a.begin, tz="UTC"):pd.Timestamp(a.end, tz="UTC")]
    if rain.empty: raise SystemExit("no observed rain in that period; run the collector, or pass --rain-csv")
    tide = noaa.build_dataset(a.tide_station, a.begin, a.end, a.datum, with_met=False, base_url=a.base_url)
    full, road, notes = storms.build_storm_table(conn, rain, tide, lead_hours=a.lead_hours, tide_thr_ft=a.tide_threshold_ft)
    [print("note:", n) for n in notes]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "storms_full.csv", index=False); road.to_csv(out / "storms.csv", index=False)
    print(f"wrote {out}/storms.csv ({len(road)} storms) and storms_full.csv ({len(full)} storms)")


if __name__ == "__main__":
    main()
