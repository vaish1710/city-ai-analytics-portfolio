"""Save NWS forecast snapshots and observed rain, so a forecast archive exists for future storms.

    python src/collect.py --db data/chesapeake.db --contact you@example.com once
    python src/collect.py --db data/chesapeake.db --contact you@example.com loop --every 3600     # or run `once` from cron hourly

The NWS API does not keep past forecasts, so the only way to have 'the forecast that existed before this storm' is to save it now.
Every snapshot records WHEN IT WAS FETCHED; later steps never use a snapshot fetched after a storm began.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import nws, store

CHESAPEAKE = (36.7682, -76.2875)      # the coordinates in the supplied example; adjust to the watershed you care about


def _location(conn, s, lat, lon, base):
    key = f"loc:{lat:.4f},{lon:.4f}"
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    loc = nws.locate(s, lat, lon, base)
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(loc))); conn.commit()
    return loc


def collect_once(conn, contact, lat=CHESAPEAKE[0], lon=CHESAPEAKE[1], base=nws.BASE, station=None, now=None, obs_hours=12, session=None):
    s = session or nws.session(contact)
    loc = _location(conn, s, lat, lon, base)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    grid = nws.get_json(s, loc["grid_url"])
    qpf = nws.grid_series(grid, "quantitativePrecipitation", "wmoUnit:mm")
    pop = nws.grid_series(grid, "probabilityOfPrecipitation", "wmoUnit:percent")
    store.insert_forecast(conn, now, nws.updated_at(grid), qpf, pop)
    st = station or nws.station_ids(s, loc["stations_url"])[0]
    obs = nws.observations(s, st, now - pd.Timedelta(hours=obs_hours), now, base)
    store.insert_obs(conn, st, obs)
    return {"fetched_at": now, "qpf_rows": len(qpf), "pop_rows": len(pop), "obs_rows": len(obs), "station": st,
            "note": None if obs else f"station {st} returned no rain observations; consider --station or a rain CSV (see README)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--contact", required=True, help="email or URL for the NWS User-Agent")
    ap.add_argument("--lat", type=float, default=CHESAPEAKE[0]); ap.add_argument("--lon", type=float, default=CHESAPEAKE[1])
    ap.add_argument("--station", default=None); ap.add_argument("--base-url", default=nws.BASE)
    ap.add_argument("mode", choices=["once", "loop"]); ap.add_argument("--every", type=int, default=3600)
    a = ap.parse_args()
    conn = store.connect(a.db)
    while True:
        try:
            r = collect_once(conn, a.contact, a.lat, a.lon, a.base_url, a.station)
            print(f"{r['fetched_at']:%Y-%m-%d %H:%M}Z  forecast rows {r['qpf_rows']}/{r['pop_rows']}  observations {r['obs_rows']} ({r['station']})" + (f"\n  note: {r['note']}" if r["note"] else ""), flush=True)
        except Exception as e:                       # keep a long-running collector alive; the error is printed, not hidden
            print("collect failed:", repr(e), flush=True)
            if a.mode == "once":
                raise SystemExit(1)
        if a.mode == "once":
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
