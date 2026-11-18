import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, pytest
import triage as T
from real_data import load_tickets

MAP = {"text": "DESC", "label": "DEPT", "created": "WHEN", "lat": "LAT", "lon": "LON", "incident": "INC"}


def export(df, tmp):
    base = pd.Timestamp("2025-01-01", tz="UTC")
    out = pd.DataFrame({"DESC": df.text, "DEPT": df.label, "WHEN": (base + pd.to_timedelta(df.hours.round(3), unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                        "LAT": df.lat, "LON": df.lon, "INC": df.incident})
    p = tmp / "t.csv"; out.to_csv(p, index=False); return p


def test_round_trip_gives_same_model_results(tmp_path):
    d = T.make_tickets(n=1500, seed=4)
    got, notes, has_inc = load_tickets(export(d, tmp_path), MAP)
    assert has_inc and len(got) == len(d) and list(got.text) == list(d.text) and list(got.label) == list(d.label)
    def acc(df):
        tr, va, te = T.chrono_split(df); vec, clf = T.fit_router(tr); p, c = T.predict(vec, clf, te.text)
        return (p == te.label.values).mean(), c.mean()
    a, b = acc(d), acc(got)
    assert a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])


def test_missing_column_error(tmp_path):
    d = T.make_tickets(n=200, seed=1); p = export(d, tmp_path)
    with pytest.raises(ValueError, match="no column found for \\['label'\\]"):
        load_tickets(p, {**MAP, "label": "NOPE"})


def test_problems_are_reported(tmp_path):
    d = T.make_tickets(n=400, seed=1); p = export(d, tmp_path)
    df = pd.read_csv(p); df.loc[0, "DESC"] = ""; df.loc[1, "WHEN"] = "garbage"; df.loc[2:20, "LAT"] = np.nan; df.to_csv(p, index=False)
    got, notes, _ = load_tickets(p, MAP, min_label_count=10_000)
    t = " | ".join(notes)
    assert "dropped 2 of 400" in t and "19 tickets have no usable location" in t and "fewer than 10000 tickets" in t
    assert len(got) == 398


def test_no_incident_column_is_stated(tmp_path):
    d = T.make_tickets(n=200, seed=1); p = export(d, tmp_path)
    got, notes, has_inc = load_tickets(p, {k: v for k, v in MAP.items() if k != "incident"})
    assert not has_inc and "NOT known" in notes[-1] and got.incident.tolist() == got.ticket.tolist()


def test_cli_end_to_end_with_keyword_rules(tmp_path):
    d = T.make_tickets(n=1500, seed=6); p = export(d, tmp_path)
    m = {**MAP, "keyword_rules": [["streets", ["pothole"]], ["waste", ["trash"]]]}; (tmp_path / "m.json").write_text(json.dumps(m))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "run_real.py"), "--tickets", str(p), "--mapping", str(tmp_path / "m.json"), "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "model, route everything" in r.stdout and "your keyword rules" in r.stdout and (tmp_path / "o" / "test_predictions.csv").exists()
    assert "text" not in pd.read_csv(tmp_path / "o" / "test_predictions.csv").columns


def test_unsorted_file_is_sorted_by_time_before_splitting(tmp_path):
    d = T.make_tickets(n=600, seed=2); p = export(d, tmp_path)
    pd.read_csv(p).sample(frac=1, random_state=0).to_csv(p, index=False)
    got, _, _ = load_tickets(p, MAP)
    assert got.hours.is_monotonic_increasing
    assert list(got.text) == list(d.text)
