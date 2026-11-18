import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, pytest
import flood as F
from real_data import load_real

MAP = {"storms": {"storm": "SID", "start": "T0", "peak_fc": "PK", "total_fc": "TOT", "tide_fc": "TIDE"},
       "segments": {"segment": "SEG", "elev_ft": "ELEV", "drain_dist_ft": "DRAIN", "tidal": "TIDAL"},
       "incidents": {"storm": "SID", "segment": "SEG"}}


def export(d, tmp):
    base = pd.Timestamp("2024-01-01", tz="UTC")
    s = d.groupby("storm").first().reset_index()
    pd.DataFrame({"SID": "S" + s.storm.astype(str), "T0": (base + pd.to_timedelta(s.storm * 7, unit="D")).dt.strftime("%Y-%m-%d"),
                  "PK": s.peak_fc, "TOT": s.total_fc, "TIDE": s.tide_fc}).sample(frac=1, random_state=1).to_csv(tmp / "storms.csv", index=False)
    g = d.groupby("segment").first().reset_index()
    pd.DataFrame({"SEG": "R" + g.segment.astype(str), "ELEV": g.elev_ft, "DRAIN": g.drain_dist_ft, "TIDAL": np.where(g.tidal == 1, "Y", "N")}).to_csv(tmp / "segments.csv", index=False)
    i = d[d.incident == 1]
    pd.DataFrame({"SID": "S" + i.storm.astype(str), "SEG": "R" + i.segment.astype(str)}).to_csv(tmp / "incidents.csv", index=False)
    return tmp / "storms.csv", tmp / "segments.csv", tmp / "incidents.csv"


def test_round_trip_matches_synthetic_path(tmp_path):
    d = F.make_data(seed=5)
    got, notes = load_real(*export(d, tmp_path), MAP)
    assert len(got) == len(d) and got.incident.sum() == d.incident.sum()
    assert (got.prior.values == d.prior.values).all()                       # same storm order despite the shuffled storms file
    for kind in ("logit",):
        a_tr, a_te = F.split_by_storm(d); b_tr, b_te = F.split_by_storm(got)
        ra = F.recall_at_k(a_te, F.fit(a_tr, kind).predict_proba(a_te[F.FEATURES])[:, 1])
        rb = F.recall_at_k(b_te, F.fit(b_tr, kind).predict_proba(b_te[F.FEATURES])[:, 1])
        assert ra == pytest.approx(rb)


def test_per_storm_hits_agree_with_recall_at_k():
    d = F.make_data(seed=2); _, te = F.split_by_storm(d)
    s = np.random.default_rng(0).random(len(te)); p = F.per_storm_hits(te, s, 8)
    assert p.hits.sum() / p.total.sum() == pytest.approx(F.recall_at_k(te, s, 8)[0])


def test_missing_column_and_unknown_ids(tmp_path):
    d = F.make_data(seed=5); st, sg, inc = export(d, tmp_path)
    with pytest.raises(ValueError, match="no column found for \\['elev_ft'\\]"):
        load_real(st, sg, inc, {**MAP, "segments": {**MAP["segments"], "elev_ft": "NOPE"}})
    x = pd.read_csv(inc); x.loc[len(x)] = ["S999", "R1"]; x.loc[len(x)] = ["S1", "R999"]; x = pd.concat([x, x.iloc[[0]]]); x.to_csv(inc, index=False)
    got, notes = load_real(st, sg, inc, MAP)
    assert "dropped 2 incident rows" in " | ".join(notes) and got.incident.sum() == d.incident.sum()     # duplicate row removed too


def test_missing_tide_is_disclosed(tmp_path):
    d = F.make_data(seed=5); st, sg, inc = export(d, tmp_path)
    m = json.loads(json.dumps(MAP)); del m["storms"]["tide_fc"]
    got, notes = load_real(st, sg, inc, m)
    assert any("tide_fc set to 0" in n for n in notes) and (got.tide_fc == 0).all()


def test_cli_end_to_end(tmp_path):
    d = F.make_data(seed=7); st, sg, inc = export(d, tmp_path); (tmp_path / "m.json").write_text(json.dumps(MAP))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "run_real.py"), "--storms", str(st), "--segments", str(sg), "--incidents", str(inc),
                        "--mapping", str(tmp_path / "m.json"), "--boot", "50", "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "hotspot history" in r.stdout and "bootstrap" in r.stdout and (tmp_path / "o" / "per_storm_model.csv").exists()
