"""Detect storms, attach the last forecast issued BEFORE each storm, and export the table road-flood-risk reads.

Forecast features come only from a snapshot that was fetched at least `lead_hours` before the storm began (never later, never observed
rain), so the table does not leak. Storms with no usable snapshot are left out and counted, not filled in.
"""
import numpy as np
import pandas as pd

import store

MM_PER_IN = 25.4


def detect_storms(rain_in, tide=None, rain_thr=0.10, gap_h=12, min_total_in=0.5, tide_thr_ft=None):
    """rain_in: hourly rain (inches), indexed by hour END. tide: optional hourly observed water level (ft).
    A storm is a run of 'active' hours (rain >= rain_thr, or tide >= tide_thr_ft) with gaps shorter than gap_h merged."""
    idx = rain_in.index if tide is None else rain_in.index.union(tide.index)
    r = rain_in.reindex(idx).fillna(0.0)
    active = r >= rain_thr
    if tide is not None and tide_thr_ft is not None:
        active = active | (tide.reindex(idx) >= tide_thr_ft)
    hrs = list(idx[active.values])
    groups, cur = [], []
    for t in hrs:
        if cur and (t - cur[-1]) > pd.Timedelta(hours=gap_h):
            groups.append(cur); cur = []
        cur.append(t)
    if cur: groups.append(cur)
    rows = []
    for g in groups:
        first, last = g[0], g[-1]
        win = r.loc[first:last]
        tide_peak = float(tide.loc[first - pd.Timedelta(hours=1):last].max()) if tide is not None and not tide.loc[first - pd.Timedelta(hours=1):last].dropna().empty else np.nan
        keep = win.sum() >= min_total_in or (tide_thr_ft is not None and tide_peak >= tide_thr_ft)
        if keep:
            rows.append({"start": first - pd.Timedelta(hours=1), "end": last, "peak_obs_in_hr": float(win.max()), "total_obs_in": float(win.sum()), "tide_obs_peak_ft": tide_peak})
    return pd.DataFrame(rows, columns=["start", "end", "peak_obs_in_hr", "total_obs_in", "tide_obs_peak_ft"])


def pick_snapshot(snapshot_times, start, lead_hours=6, max_age_hours=36):
    """Latest snapshot fetched at or before start - lead_hours, and not older than max_age_hours before that cutoff."""
    cutoff = pd.Timestamp(start) - pd.Timedelta(hours=lead_hours)
    ok = [t for t in snapshot_times if t <= cutoff]
    if not ok:
        return None
    best = max(ok)
    return best if (cutoff - best) <= pd.Timedelta(hours=max_age_hours) else None


def qpf_features(rows, start, end):
    """From forecast rows (valid_start, hours, mm): (peak inches/hour, total inches, coverage 0..1) over [start, end].
    Each interval's rain is spread evenly over its hours and prorated by overlap with the storm window."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total_mm, peak_mm_hr, covered = 0.0, 0.0, 0.0
    for vs, hours, mm in rows:
        ve = vs + pd.Timedelta(hours=hours)
        ov = (min(ve, end) - max(vs, start)).total_seconds() / 3600
        if ov <= 0:
            continue
        total_mm += mm * ov / hours
        peak_mm_hr = max(peak_mm_hr, mm / hours)
        covered += ov
    span = (end - start).total_seconds() / 3600
    return peak_mm_hr / MM_PER_IN, total_mm / MM_PER_IN, (min(1.0, covered / span) if span > 0 else 0.0)


def build_storm_table(conn, rain_in, tide_df=None, lead_hours=6, max_age_hours=36, min_coverage=0.8, **detect_kw):
    """Returns (full table, road-flood-risk table, notes). tide_df has hourly 'observed' and 'predicted' columns (ft)."""
    notes = []
    tide_obs = tide_df["observed"].dropna() if tide_df is not None else None
    storms = detect_storms(rain_in, tide_obs, **detect_kw)
    snaps = store.snapshot_times(conn)
    rows = []
    for s in storms.itertuples():
        snap = pick_snapshot(snaps, s.start, lead_hours, max_age_hours)
        row = {"storm": f"S{s.start:%Y%m%d%H}", "start": s.start, "end": s.end, "peak_obs_in_hr": s.peak_obs_in_hr, "total_obs_in": s.total_obs_in,
               "tide_obs_peak_ft": s.tide_obs_peak_ft, "snapshot": snap, "peak_fc": np.nan, "total_fc": np.nan, "coverage": np.nan}
        if snap is not None:
            pk, tot, cov = qpf_features(store.snapshot_rows(conn, snap), s.start, s.end)
            row.update(peak_fc=pk, total_fc=tot, coverage=cov)
        row["tide_fc"] = float(tide_df["predicted"].loc[s.start:s.end].max()) if tide_df is not None and not tide_df["predicted"].loc[s.start:s.end].dropna().empty else np.nan
        row["has_forecast"] = snap is not None and row["coverage"] >= min_coverage
        rows.append(row)
    full = pd.DataFrame(rows)
    if full.empty:
        return full, full, ["no storms detected"]
    n_ok = int(full.has_forecast.sum())
    notes.append(f"{len(full)} storms detected; {n_ok} have a forecast snapshot fetched at least {lead_hours} h before the storm; {len(full) - n_ok} excluded (no usable snapshot or coverage < {min_coverage:.0%})")
    if n_ok < 40: notes.append("WARNING: fewer than 40 storms with forecasts. Keep the collector running; the road-flood model needs far more.")
    if tide_df is None: notes.append("no tide data: tide_fc is empty")
    elif full.tide_fc.isna().any(): notes.append(f"{int(full.tide_fc.isna().sum())} storms have no tide prediction")
    notes.append("tide_fc is the predicted ASTRONOMICAL tide peak; it does not include any storm-surge forecast")
    road = full[full.has_forecast][["storm", "start", "peak_fc", "total_fc", "tide_fc"]].copy()
    return full, road, notes
