import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import permit as P, simulate as S

R = P.load_rules()


def test_checklist_is_consistent():
    docs = R["documents"]
    for pt, p in R["permits"].items():
        for t in p["required"] + [c["require"] for c in p["conditional"]]:
            assert t in docs and docs[t]["keywords"], (pt, t)
    assert "FICTIONAL" in R["_note"]


def test_recognize_examples():
    assert P.recognize("Site_Plan_FINAL.pdf", R) == ["site_plan"]
    assert P.recognize("scan0012.pdf", R) == []
    assert sorted(P.recognize("framing plan.pdf", R)) == ["framing_detail", "structural_drawings"]
    assert P.recognize("EC-2024.pdf", R) == ["elevation_certificate"]
    assert P.recognize("because.pdf", R) == []                      # 'ec' inside a word must not match


def test_required_docs_conditionals():
    assert P.required_docs("deck", {"height_ft": 6, "in_flood_zone": False}, R) == ["site_plan", "framing_detail", "owner_authorization"]
    assert "engineer_letter" in P.required_docs("deck", {"height_ft": 10, "in_flood_zone": False}, R)
    assert "engineer_letter" not in P.required_docs("deck", {"height_ft": 8, "in_flood_zone": False}, R)   # 'greater than 8', not 8
    assert "elevation_certificate" in P.required_docs("deck", {"height_ft": 6, "in_flood_zone": True}, R)
    assert "contractor_license" in P.required_docs("roof_replacement", {"uses_contractor": True}, R)


def test_complete():
    r = P.check("fence", {}, ["Site Plan.pdf", "Owner Authorization.pdf"], R)
    assert r["status"] == "complete" and r["missing"] == [] and r["uncertain"] == []


def test_certain_missing():
    r = P.check("fence", {}, ["Site Plan.pdf", "photos.pdf"], R)       # 'photos.pdf' is unknown, so owner letter is only uncertain
    assert r["status"] == "staff_review" and r["uncertain"] == ["owner_authorization"]
    r = P.check("fence", {}, ["Site Plan.pdf", "Engineer Letter.pdf"], R)   # every file is recognized: nothing could hide the letter
    assert r["status"] == "incomplete" and r["missing"] == ["owner_authorization"]


def test_ambiguous_file_is_not_counted_as_confirmed():
    r = P.check("residential_addition", {}, ["Site Plan.pdf", "Floor Plan.pdf", "Owner Authorization.pdf", "framing plan.pdf"], R)
    assert r["status"] == "staff_review" and r["uncertain"] == ["structural_drawings"] and r["ambiguous_files"] == ["framing plan.pdf"]
    assert P.naive_check("residential_addition", {}, ["Site Plan.pdf", "Floor Plan.pdf", "Owner Authorization.pdf", "framing plan.pdf"], R)["status"] == "complete"


def test_message_lists_labels_not_ids():
    r = P.check("fence", {}, ["Site Plan.pdf", "Engineer Letter.pdf"], R)
    m = P.message(r, "fence", R)
    assert "Owner authorization letter" in m and "owner_authorization" not in m


def test_known_limit_misleading_name_fools_the_check():
    """Documented limitation: a photo named 'site plan photos.pdf' is accepted as the site plan."""
    r = P.check("fence", {}, ["site plan photos.pdf", "Owner Authorization.pdf"], R)
    assert r["status"] == "complete"


def test_simulation_assistant_never_wrongly_claims_missing_and_beats_baseline():
    subs = S.make_submissions(R, n=800, seed=3)
    wrong = claims = nwrong = nclaims = 0
    for s in subs:
        truth = set(s["truly_missing"])
        a = P.check(s["permit_type"], s["fields"], s["files"], R)["missing"]
        b = P.naive_check(s["permit_type"], s["fields"], s["files"], R)["missing"]
        wrong += len(set(a) - truth); claims += len(a); nwrong += len(set(b) - truth); nclaims += len(b)
    assert claims > 0 and wrong == 0
    assert nwrong / nclaims > 0.2
