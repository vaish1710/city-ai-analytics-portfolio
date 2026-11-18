import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, pytest
import nws, store, storms, collect, noaa, mock_nws, mock_noaa
from build_storm_table import rain_from_csv

T = lambda s: pd.Timestamp(s, tz="UTC")


def snap(conn, fetched, rows):
    store.insert_forecast(conn, T(fetched), T(fetched), [(T(a), h, v) for a, h, v in rows], [])


def test_duration_and_valid_time():
    assert nws.parse_duration_hours("PT6H") == 6 and nws.parse_duration_hours("P1DT12H") == 36 and nws.parse_duration_hours("PT30M") == 0.5
    with pytest.raises(nws.NwsError): nws.parse_duration_hours("6 hours")
    assert nws.parse_valid_time("2026-01-01T06:00:00+00:00/PT6H") == (T("2026-01-01 06:00"), 6)
    assert nws.parse_valid_time("2026-01-01T01:00:00-05:00/PT1H")[0] == T("2026-01-01 06:00")


def test_session_requires_identifying_contact():
    with pytest.raises(ValueError): nws.session("")
    assert "me@example.org" in nws.session("me@example.org").headers["User-Agent"]


def _mock_state():
    return dict(qpf=[{"validTime": "2026-03-01T00:00:00+00:00/PT6H", "value": 12.7}, {"validTime": "2026-03-01T06:00:00+00:00/PT6H", "value": None}, {"validTime": "2026-03-01T12:00:00+00:00/PT6H", "value": 0.0}],
                pop=[{"validTime": "2026-03-01T00:00:00+00:00/PT12H", "value": 80}],
                obs=[("2026-03-01T00:53:00+00:00", 2.5), ("2026-03-01T01:53:00+00:00", None), ("2026-03-01T02:53:00+00:00", 0.0)], update_time="2026-02-28T22:10:00+00:00")


def test_collect_once_stores_rows_and_is_idempotent(tmp_path):
    srv, url, st = mock_nws.start(**_mock_state())
    conn = store.connect(tmp_path / "d.db")
    r = collect.collect_once(conn, "me@example.org", base=url, now=T("2026-03-01 03:00"))
    assert (r["qpf_rows"], r["pop_rows"], r["obs_rows"], r["station"]) == (2, 1, 2, "KAAA")           # null QPF and null obs skipped
    assert all(q["ua"] and "me@example.org" in q["ua"] for q in st["requests"])
    n = lambda t: conn.execute(f"select count(*) from {t}").fetchone()[0]
    assert (n("forecast_qpf"), n("forecast_pop"), n("obs_rain")) == (2, 1, 2)
    collect.collect_once(conn, "me@example.org", base=url, now=T("2026-03-01 03:00"))
    assert (n("forecast_qpf"), n("obs_rain")) == (2, 2)                                               # same fetch time: nothing duplicated
    collect.collect_once(conn, "me@example.org", base=url, now=T("2026-03-01 04:00"))
    assert n("forecast_qpf") == 4 and len(store.snapshot_times(conn)) == 2                            # new fetch time: new snapshot kept
    assert conn.execute("select mm from forecast_qpf where valid_start=?", (store.iso(T("2026-03-01 00:00")),)).fetchone()[0] == 12.7
    assert sum(1 for q in st["requests"] if q["path"].startswith("/points/")) == 1                    # location cached after first call
    srv.shutdown()


def test_station_choice_and_no_rain_reported(tmp_path):
    s = _mock_state(); s["obs"] = []
    srv, url, st = mock_nws.start(**s)
    r = collect.collect_once(store.connect(tmp_path / "d.db"), "me@example.org", base=url, station="KCPK", now=T("2026-03-01 03:00"))
    assert r["station"] == "KCPK" and r["obs_rows"] == 0 and "no rain observations" in r["note"]
    assert any(q["path"] == "/stations/KCPK/observations" for q in st["requests"])
    srv.shutdown()


def test_unexpected_units_fail_loudly(tmp_path):
    srv, url, _ = mock_nws.start(**_mock_state(), qpf_uom="wmoUnit:in")
    with pytest.raises(nws.NwsError, match="expected unit wmoUnit:mm"):
        collect.collect_once(store.connect(tmp_path / "d.db"), "me@example.org", base=url, now=T("2026-03-01 03:00"))
    srv.shutdown()


def test_retries_a_503(tmp_path):
    srv, url, st = mock_nws.start(**_mock_state(), fail_next=1)
    r = collect.collect_once(store.connect(tmp_path / "d.db"), "me@example.org", base=url, now=T("2026-03-01 03:00"))
    assert r["qpf_rows"] == 2 and sum(1 for q in st["requests"] if q["path"].startswith("/points/")) == 2
    srv.shutdown()


def hourly(start, vals):
    return pd.Series(vals, index=pd.date_range(T(start), periods=len(vals), freq="h"))


def test_detect_storms_hand_example():
    r = np.zeros(120); r[10] = r[11] = 0.3; r[15] = 0.4            # one storm: two bursts 3 h apart (merged)
    r[50] = r[51] = 0.2                                              # total 0.4 < 0.5: dropped
    r[80] = 0.6                                                      # kept
    s = storms.detect_storms(hourly("2026-03-01 00:00", r))
    assert len(s) == 2
    a, b = s.iloc[0], s.iloc[1]
    assert a.start == T("2026-03-01 09:00") and a.end == T("2026-03-01 15:00")                       # start = first active hour-end minus 1 h
    assert a.total_obs_in == pytest.approx(1.0) and a.peak_obs_in_hr == pytest.approx(0.4)
    assert b.start == T("2026-03-04 07:00") and b.total_obs_in == pytest.approx(0.6)
    r[30] = 0.5                                                      # 15 h after hour 15: separate storm, not merged
    assert len(storms.detect_storms(hourly("2026-03-01 00:00", r))) == 3


def test_detect_storms_by_tide():
    tide = hourly("2026-03-01 00:00", np.r_[np.full(10, 1.0), [4.0, 4.2], np.full(10, 1.0)])
    s = storms.detect_storms(hourly("2026-03-01 00:00", np.zeros(22)), tide, tide_thr_ft=3.5)
    assert len(s) == 1 and s.iloc[0].tide_obs_peak_ft == pytest.approx(4.2)


def test_qpf_features_proration():
    rows = [(T("2026-03-01 06:00"), 6, 12.0), (T("2026-03-01 12:00"), 6, 6.0)]
    pk, tot, cov = storms.qpf_features(rows, T("2026-03-01 09:00"), T("2026-03-01 15:00"))
    assert tot == pytest.approx(9.0 / 25.4) and pk == pytest.approx(2.0 / 25.4) and cov == 1.0
    assert storms.qpf_features(rows[:1], T("2026-03-01 09:00"), T("2026-03-01 15:00"))[2] == pytest.approx(0.5)


def test_pick_snapshot_never_uses_future_or_stale_forecasts():
    start = T("2026-03-10 12:00"); h = lambda x: start + pd.Timedelta(hours=x)
    assert storms.pick_snapshot([h(-10), h(-5), h(1)], start, 6) == h(-10)           # -5 is inside the lead window, +1 is after the start
    assert storms.pick_snapshot([h(-5), h(1)], start, 6) is None
    assert storms.pick_snapshot([h(-50)], start, 6, max_age_hours=36) is None        # too old
    assert storms.pick_snapshot([h(-6)], start, 6) == h(-6)                            # exactly at the cutoff is allowed


def _tide_df():
    idx = pd.date_range(T("2026-03-01"), periods=24 * 40, freq="h")
    pred = 2 + 1.5 * np.sin(np.arange(len(idx)) * 2 * np.pi / 12.42)
    return pd.DataFrame({"observed": pred + 0.1, "predicted": pred}, index=idx)


def _rain():
    r = np.zeros(24 * 30); r[48] = 0.5; r[49] = 0.4; r[200] = 0.7; r[400] = 0.9
    return hourly("2026-03-01 01:00", r)


def test_table_uses_only_prior_snapshots_and_not_later_ones(tmp_path):
    conn = store.connect(tmp_path / "d.db")
    snap(conn, "2026-03-02 12:00", [("2026-03-02 12:00", 6, 6.0), ("2026-03-02 18:00", 6, 6.0), ("2026-03-03 00:00", 6, 6.0), ("2026-03-03 06:00", 6, 6.0)])
    full1, road1, notes = storms.build_storm_table(conn, _rain(), _tide_df())
    first = full1[full1.start == T("2026-03-03 00:00")].iloc[0]
    assert first.has_forecast and first.snapshot == T("2026-03-02 12:00") and first.total_fc > 0
    snap(conn, "2026-03-03 00:00", [("2026-03-03 00:00", 6, 600.0)])      # fetched AT storm start: inside the lead window
    snap(conn, "2026-03-03 06:00", [("2026-03-03 00:00", 6, 600.0)])      # fetched after the storm began
    full2, road2, _ = storms.build_storm_table(conn, _rain(), _tide_df())
    f2 = full2[full2.start == T("2026-03-03 00:00")].iloc[0]
    assert (f2.peak_fc, f2.total_fc, f2.snapshot) == (first.peak_fc, first.total_fc, first.snapshot)    # later snapshots changed nothing


def test_storms_without_a_snapshot_are_excluded_and_counted(tmp_path):
    conn = store.connect(tmp_path / "d.db")
    snap(conn, "2026-03-02 12:00", [("2026-03-02 12:00", 24, 12.0)])
    full, road, notes = storms.build_storm_table(conn, _rain(), _tide_df())
    assert len(full) == 3 and len(road) == 1 and road.iloc[0].start == T("2026-03-03 00:00")
    t = " | ".join(notes)
    assert "3 storms detected; 1 have a forecast" in t and "WARNING: fewer than 40" in t and "ASTRONOMICAL" in t
    assert list(road.columns) == ["storm", "start", "peak_fc", "total_fc", "tide_fc"] and road.tide_fc.notna().all()


def test_road_table_loads_in_the_road_flood_loader(tmp_path):
    loader = Path(__file__).resolve().parents[2] / "road-flood-risk" / "src"
    if not loader.exists(): pytest.skip("road-flood-risk not alongside")
    conn = store.connect(tmp_path / "d.db")
    snap(conn, "2026-03-02 12:00", [("2026-03-02 12:00", 24, 12.0), ("2026-03-03 00:00", 24, 12.0)])
    snap(conn, "2026-03-08 12:00", [("2026-03-09 00:00", 24, 12.0)])
    snap(conn, "2026-03-17 00:00", [("2026-03-17 00:00", 48, 12.0)])
    _, road, _ = storms.build_storm_table(conn, _rain(), _tide_df())
    assert len(road) >= 2
    road.to_csv(tmp_path / "storms.csv", index=False)
    pd.DataFrame({"segment": ["a", "b", "c"], "elev": [3, 6, 9], "drain": [100, 200, 300]}).to_csv(tmp_path / "seg.csv", index=False)
    pd.DataFrame({"storm": [road.storm.iloc[0]], "segment": ["a"]}).to_csv(tmp_path / "inc.csv", index=False)
    sys.path.insert(0, str(loader))
    import importlib; rd = importlib.import_module("real_data")
    df, notes = rd.load_real(tmp_path / "storms.csv", tmp_path / "seg.csv", tmp_path / "inc.csv",
        {"storms": {k: k for k in ("storm", "start", "peak_fc", "total_fc", "tide_fc")}, "segments": {"segment": "segment", "elev_ft": "elev", "drain_dist_ft": "drain"}, "incidents": {"storm": "storm", "segment": "segment"}})
    assert len(df) == 3 * len(road) and df.incident.sum() == 1


def test_rain_csv_timezone_conversion(tmp_path):
    p = tmp_path / "r.csv"; pd.DataFrame({"time": ["2026-03-10 08:00", "2026-03-10 09:00"], "precip_in": [0.2, "bad"]}).to_csv(p, index=False)
    s = rain_from_csv(p, "America/New_York")                                     # EDT began 2026-03-08, so 08:00 local = 12:00 UTC
    assert list(s.index) == [T("2026-03-10 12:00")] and s.iloc[0] == 0.2
    with pytest.raises(ValueError, match="needs columns"):
        pd.DataFrame({"x": [1]}).to_csv(p, index=False); rain_from_csv(p)


def test_cli_collect_and_build_end_to_end(tmp_path):
    srv, url, _ = mock_nws.start(**_mock_state())
    db = tmp_path / "d.db"
    r = subprocess.run([sys.executable, str(SRC / "collect.py"), "--db", str(db), "--contact", "me@example.org", "--base-url", url, "once"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "observations 2 (KAAA)" in r.stdout
    r2 = subprocess.run([sys.executable, str(SRC / "collect.py"), "--db", str(db), "--base-url", url, "once"], capture_output=True, text=True)
    assert r2.returncode != 0 and "--contact" in r2.stderr                      # contact is mandatory
    srv.shutdown()
    tsrv, turl, _ = mock_noaa.start(_tide_df())
    _rain().rename("precip_in").rename_axis("time").reset_index().to_csv(tmp_path / "rain.csv")
    store.connect(db); c = store.connect(db); snap(c, "2026-03-02 12:00", [("2026-03-02 12:00", 24, 12.0), ("2026-03-03 00:00", 24, 12.0)])
    r3 = subprocess.run([sys.executable, str(SRC / "build_storm_table.py"), "--db", str(db), "--begin", "2026-03-01", "--end", "2026-03-30", "--rain-csv", str(tmp_path / "rain.csv"),
                         "--base-url", turl + "/api/prod/datagetter", "--out", str(tmp_path / "out")], capture_output=True, text=True)
    assert r3.returncode == 0, r3.stderr + r3.stdout
    assert (tmp_path / "out" / "storms.csv").exists() and "storms detected" in r3.stdout
    tsrv.shutdown()


def test_hourly_rain_assignment(tmp_path):
    conn = store.connect(tmp_path / "d.db")
    store.insert_obs(conn, "K1", [(T("2026-03-01 00:53"), 2.54), (T("2026-03-01 01:10"), 5.08), (T("2026-03-01 02:53"), 25.4)])   # 00:53 and 01:10 both round to 01:00 -> keep the larger
    store.insert_obs(conn, "K2", [(T("2026-03-01 02:53"), 50.8)])
    s = store.hourly_rain_inches(conn, "K1")
    assert list(s.index) == [T("2026-03-01 01:00"), T("2026-03-01 03:00")] and list(s.round(3)) == [0.2, 1.0]
    assert store.hourly_rain_inches(conn).loc[T("2026-03-01 03:00")] == pytest.approx(2.0)                                     # all stations: max
    assert store.hourly_rain_inches(store.connect(tmp_path / "empty.db")).empty


def test_forecast_covering_only_part_of_the_storm_is_not_used(tmp_path):
    conn = store.connect(tmp_path / "d.db")
    snap(conn, "2026-03-02 12:00", [("2026-03-02 18:00", 6, 6.0)])            # forecast ends at 00:00, storm runs 00:00-02:00: zero coverage
    full, road, _ = storms.build_storm_table(conn, _rain(), _tide_df())
    f = full[full.start == T("2026-03-03 00:00")].iloc[0]
    assert f.coverage == 0.0 and not f.has_forecast and road.empty
