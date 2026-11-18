"""SQLite store for forecast snapshots and observed rain. Timestamps are ISO-8601 UTC strings."""
import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_qpf (fetched_at TEXT, issued_at TEXT, valid_start TEXT, hours REAL, mm REAL, PRIMARY KEY (fetched_at, valid_start));
CREATE TABLE IF NOT EXISTS forecast_pop (fetched_at TEXT, issued_at TEXT, valid_start TEXT, hours REAL, pct REAL, PRIMARY KEY (fetched_at, valid_start));
CREATE TABLE IF NOT EXISTS obs_rain (station TEXT, ts TEXT, mm_last_hour REAL, PRIMARY KEY (station, ts));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
iso = lambda t: None if t is None else pd.Timestamp(t).tz_convert("UTC").isoformat()


def connect(path):
    c = sqlite3.connect(path)
    c.executescript(SCHEMA)
    return c


def insert_forecast(c, fetched_at, issued_at, qpf, pop):
    f, i = iso(fetched_at), iso(issued_at)
    c.executemany("INSERT OR IGNORE INTO forecast_qpf VALUES (?,?,?,?,?)", [(f, i, iso(t), h, v) for t, h, v in qpf])
    c.executemany("INSERT OR IGNORE INTO forecast_pop VALUES (?,?,?,?,?)", [(f, i, iso(t), h, v) for t, h, v in pop])
    c.commit()


def insert_obs(c, station, obs):
    c.executemany("INSERT OR IGNORE INTO obs_rain VALUES (?,?,?)", [(station, iso(t), v) for t, v in obs])
    c.commit()


def snapshot_times(c):
    return [pd.Timestamp(r[0]) for r in c.execute("SELECT DISTINCT fetched_at FROM forecast_qpf ORDER BY fetched_at")]


def snapshot_rows(c, fetched_at):
    return [(pd.Timestamp(a), h, v) for a, h, v in c.execute("SELECT valid_start, hours, mm FROM forecast_qpf WHERE fetched_at=? ORDER BY valid_start", (iso(fetched_at),))]


def hourly_rain_inches(c, station=None):
    """Observed rain per hour, indexed by hour END (UTC). An observation at 12:53 reporting the last hour is assigned to 13:00."""
    q, args = "SELECT ts, mm_last_hour FROM obs_rain", ()
    if station:
        q, args = q + " WHERE station=?", (station,)
    df = pd.DataFrame(c.execute(q, args).fetchall(), columns=["ts", "mm"])
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    df["hour"] = pd.to_datetime(df.ts, utc=True).dt.round("h")
    return (df.groupby("hour").mm.max() / 25.4).sort_index()
