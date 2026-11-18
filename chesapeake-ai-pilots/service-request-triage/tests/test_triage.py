import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
import triage as T


def test_split_is_chronological_and_disjoint():
    df = T.make_tickets(n=500, seed=1)
    a, b, c = T.chrono_split(df)
    assert a.hours.max() <= b.hours.min() and b.hours.max() <= c.hours.min()
    assert len(a) + len(b) + len(c) == len(df)
    assert not (set(a.ticket) & set(c.ticket))


def test_keyword_rules_examples():
    assert T.keyword_route("Huge pothole on Main") == "streets"
    assert T.keyword_route("traffic light is out") == "traffic"
    assert T.keyword_route("garbage truck skipped us") == "waste"
    assert T.keyword_route("ditch is full") == "stormwater"
    assert T.keyword_route("loud party") == "parks_other"


def test_haversine():
    assert abs(T.haversine_m(0, 0, 0, 0.001) - 111.19) < 0.5
    assert T.haversine_m(36.8, -76.3, 36.8, -76.3) == 0


def test_choose_threshold_and_fallback():
    p = np.r_[np.full(30, 0.5), np.full(30, 0.9)]
    ok = np.r_[np.r_[np.ones(15), np.zeros(15)], np.ones(30)].astype(bool)      # 50% accurate at 0.5, 100% at 0.9
    t, reached = T.choose_threshold(p, ok, target=0.95, min_n=20)
    assert reached and 0.5 < t <= 0.9
    t2, reached2 = T.choose_threshold(p, np.zeros(60, bool), target=0.95, min_n=20)
    assert not reached2 and t2 == 0.95


def _tiny():
    df = pd.DataFrame({"ticket": [0, 1, 2, 3], "hours": [0, 10, 200, 5], "lat": [36.8, 36.8001, 36.8, 36.9], "lon": [-76.3] * 4,
                       "text": ["pothole on main", "pothole on main", "pothole on main", "pothole on main"],
                       "incident": [0, 0, 2, 3]})
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer().fit(df.text)
    return df, v


def test_candidate_pairs_space_time_team():
    df, v = _tiny()
    cats = ["streets"] * 4
    # 0-1: ~11 m, 10 h -> pair. 0-2: 200 h apart -> no. 0-3: ~11 km away -> no.
    assert T.candidate_pairs(df, cats, v) == {(0, 1)}
    assert T.candidate_pairs(df, ["streets", "traffic", "streets", "streets"], v) == set()     # different team


def test_true_pairs_and_pr():
    df = pd.DataFrame({"ticket": [0, 1, 2], "incident": [5, 5, 7]})
    tp = T.true_pairs(df)
    assert tp == {(0, 1)}
    assert T.pr({(0, 1), (0, 2)}, tp) == (0.5, 1.0)


def test_typo_changes_one_word_keeps_length():
    rng = np.random.default_rng(0)
    t = "broken signal light"
    t2 = T.typo(t, rng)
    assert t2 != t and len(t2) == len(t) and sorted(t2) == sorted(t)


def test_model_beats_keywords_and_vague_tickets_go_to_people():
    df = T.make_tickets(seed=11)
    tr, va, te = T.chrono_split(df)
    vec, clf = T.fit_router(tr)
    pv, cv = T.predict(vec, clf, va.text); tau, _ = T.choose_threshold(cv, pv == va.label.values)
    pt, ct = T.predict(vec, clf, te.text)
    kw = te.text.map(T.keyword_route).values
    assert (pt == te.label.values).mean() > (kw == te.label.values).mean() + 0.03
    auto = ct >= tau
    assert (~auto[te.vague.values]).mean() > (~auto[~te.vague.values]).mean()
