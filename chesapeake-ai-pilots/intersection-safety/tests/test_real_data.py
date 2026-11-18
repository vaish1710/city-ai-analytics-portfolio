import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, pytest
import safety as S
from real_data import load_real, read_mapped

MAP = {"crashes": {"site": "ID", "date": "WHEN"}, "sites": {"site": "ID", "aadt": "ADT", "signal": "SIG", "legs": "LEGS", "speed": "SPD"}}
YEARS = list(S.TRAIN_YEARS + S.TEST_YEARS)


def export(d, tmp):
    s = pd.DataFrame({"ID": d.site, "ADT": d.aadt, "SIG": np.where(d.signal == 1, "Y", "N"), "LEGS": np.where(d.legs4 == 1, 4, 3), "SPD": d.speed})
    rows = [{"ID": r.site, "WHEN": f"{y}-06-15"} for r in d.itertuples() for y in YEARS for _ in range(getattr(r, f"crashes_{y}"))]
    s.to_csv(tmp / "sites.csv", index=False); pd.DataFrame(rows).to_csv(tmp / "crashes.csv", index=False)
    return tmp / "crashes.csv", tmp / "sites.csv"


def test_round_trip_matches_the_synthetic_path(tmp_path):
    d = S.make_sites(seed=2)
    c, s = export(d, tmp_path)
    got, notes = load_real(c, s, MAP, YEARS)
    got = S.features(got)
    for y in YEARS:
        assert (got.sort_values("site")[f"crashes_{y}"].values == d.sort_values("site")[f"crashes_{y}"].values).all()
    a, _ = S.rank_scores(d); b, _ = S.rank_scores(got)
    for n in a:
        assert S.future_share(d, a[n]) == pytest.approx(S.future_share(got, b[n]))
    assert not [n for n in notes if "WARNING" in n]


def test_missing_mapping_column_gives_clear_error(tmp_path):
    d = S.make_sites(seed=2); c, s = export(d, tmp_path)
    bad = json.loads(json.dumps(MAP)); bad["sites"]["aadt"] = "NOPE"
    with pytest.raises(ValueError, match="no column found for \\['aadt'\\]"):
        load_real(c, s, bad, YEARS)


def test_problems_are_reported_not_hidden(tmp_path):
    d = S.make_sites(seed=2); c, s = export(d, tmp_path)
    cr = pd.read_csv(c)
    cr = cr[~cr.WHEN.str.startswith("2023")]                                      # an incomplete/missing year
    cr = pd.concat([cr, pd.DataFrame({"ID": [999999, 0], "WHEN": ["2021-01-01", "not a date"]})]); cr.to_csv(c, index=False)
    st = pd.read_csv(s); st.loc[0, "ADT"] = 0; st.to_csv(s, index=False)
    got, notes = load_real(c, s, MAP, YEARS)
    text = " | ".join(notes)
    assert "no crashes at all in 2023" in text and "not in the sites file" in text and "unreadable dates" in text and "missing/zero volume" in text
    assert len(got) == len(d) - 1


def test_severity_filter(tmp_path):
    d = S.make_sites(seed=2); c, s = export(d, tmp_path)
    cr = pd.read_csv(c); cr["SEV"] = np.where(np.arange(len(cr)) % 4 == 0, "K", "O"); cr.to_csv(c, index=False)
    m = json.loads(json.dumps(MAP)); m["crashes"]["severity"] = "SEV"; m["crashes"]["severity_keep"] = ["K"]
    got, _ = load_real(c, s, m, YEARS)
    assert abs(sum(got[f"crashes_{y}"].sum() for y in YEARS) - len(cr) / 4) <= 1


def test_cli_end_to_end(tmp_path):
    d = S.make_sites(seed=3); c, s = export(d, tmp_path)
    (tmp_path / "m.json").write_text(json.dumps(MAP))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "run_real.py"), "--crashes", str(c), "--sites", str(s),
                        "--mapping", str(tmp_path / "m.json"), "--boot", "50", "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "empirical Bayes" in r.stdout and (tmp_path / "o" / "sites_to_review.csv").exists()
