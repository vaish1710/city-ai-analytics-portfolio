import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd, pytest
import permit as P, simulate as S
from real_data import load_submissions

R = P.load_rules()
MAP = {"permit_type": "KIND", "files": "FILES", "staff_missing": "MISSING", "fields": {"height_ft": "H", "in_flood_zone": "FLOOD", "uses_contractor": "CONTR"}}


def export(subs, tmp, truth=True):
    rows = [{"KIND": s["permit_type"], "FILES": "; ".join(s["files"]), "H": s["fields"]["height_ft"], "FLOOD": "Y" if s["fields"]["in_flood_zone"] else "N",
             "CONTR": "yes" if s["fields"]["uses_contractor"] else "no", "MISSING": "; ".join(s["truly_missing"])} for s in subs]
    df = pd.DataFrame(rows)
    if not truth: df = df.drop(columns="MISSING")
    p = tmp / "s.csv"; df.to_csv(p, index=False); return p


def test_round_trip_same_results(tmp_path):
    subs = S.make_submissions(R, n=300, seed=4)
    got, notes, has_truth = load_submissions(export(subs, tmp_path), MAP, R)
    assert has_truth and len(got) == 300
    for a, b in zip(subs, got):
        assert a["permit_type"] == b["permit_type"] and a["files"] == b["files"] and a["truly_missing"] == b["truly_missing"] and a["fields"] == b["fields"]
        assert P.check(a["permit_type"], a["fields"], a["files"], R) == P.check(b["permit_type"], b["fields"], b["files"], R)


def test_errors_and_notes(tmp_path):
    subs = S.make_submissions(R, n=50, seed=1); p = export(subs, tmp_path)
    with pytest.raises(ValueError, match="no column found for \\['files'\\]"):
        load_submissions(p, {**MAP, "files": "NOPE"}, R)
    with pytest.raises(ValueError, match="fields mapping 'height_ft'"):
        load_submissions(p, {**MAP, "fields": {"height_ft": "NOPE"}}, R)
    df = pd.read_csv(p); df.loc[0, "KIND"] = "pool"; df.loc[1, "MISSING"] = "made_up_doc"; df.loc[2, "FILES"] = ""; df.to_csv(p, index=False)
    got, notes, _ = load_submissions(p, MAP, R)
    t = " | ".join(notes)
    assert "skipped 1 applications" in t and "made_up_doc" in t and "no file names" in t and len(got) == 49


def test_without_staff_labels(tmp_path):
    subs = S.make_submissions(R, n=30, seed=1); p = export(subs, tmp_path, truth=False)
    got, notes, has_truth = load_submissions(p, {k: v for k, v in MAP.items() if k != "staff_missing"}, R)
    assert not has_truth and "NOT present" in notes[-1] and got[0]["truly_missing"] is None


def test_cli_end_to_end_and_fictional_warning(tmp_path):
    subs = S.make_submissions(R, n=400, seed=2); p = export(subs, tmp_path); (tmp_path / "m.json").write_text(json.dumps(MAP))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "run_real.py"), "--submissions", str(p), "--mapping", str(tmp_path / "m.json"), "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "FICTIONAL" in r.stdout and "UNRECOGNIZED" in r.stdout and "assistant 'missing' claims" in r.stdout and (tmp_path / "o" / "unrecognized_names.csv").exists()
