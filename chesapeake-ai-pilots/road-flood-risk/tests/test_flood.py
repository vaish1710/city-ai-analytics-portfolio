import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
import flood


def test_split_has_no_storm_overlap_and_is_chronological():
    d = flood.make_data(seed=1)
    tr, te = flood.split_by_storm(d)
    assert set(tr.storm).isdisjoint(set(te.storm))
    assert tr.storm.max() < te.storm.min()


def test_prior_uses_only_earlier_storms():
    d = flood.make_data(seed=2)
    s = 30
    e = d.copy()
    e.loc[e.storm >= s, "incident"] = 1 - e.loc[e.storm >= s, "incident"]     # corrupt the present and the future
    e = flood.add_prior(e.drop(columns=["prior", "prior_rate"]))
    a, b = d[d.storm <= s].sort_values(["storm", "segment"]), e[e.storm <= s].sort_values(["storm", "segment"])
    assert (a.prior.values == b.prior.values).all()          # unchanged through storm s, even though storm s itself was flipped


def test_prior_hand_example():
    df = pd.DataFrame({"storm": [0, 1, 2, 3], "segment": [7] * 4, "incident": [1, 0, 1, 0]})
    out = flood.add_prior(df)
    assert out.prior.tolist() == [0, 1, 1, 2]
    assert out.prior_rate.tolist() == [0 / 5, 1 / 6, 1 / 7, 2 / 8]


def test_recall_at_k_hand_check():
    df = pd.DataFrame({"storm": [0, 0, 0, 1, 1, 1], "segment": [0, 1, 2, 0, 1, 2], "incident": [1, 0, 1, 0, 0, 0]})
    r, fa = flood.recall_at_k(df, [0.9, 0.8, 0.1, 0.5, 0.4, 0.3], k=2)
    assert r == 0.5            # storm 0: flagged segs 0,1 -> caught 1 of 2 incidents; storm 1 has none
    assert fa == (1 + 2) / 2   # 1 false alert in storm 0, 2 in storm 1


def test_ties_broken_by_segment_not_input_order():
    df = pd.DataFrame({"storm": [0] * 3, "segment": [2, 0, 1], "incident": [0, 1, 0]})
    r, _ = flood.recall_at_k(df, [1, 1, 1], k=1)
    assert r == 1.0            # all tied: segment 0 wins


def test_reasons():
    ref = {"peak_q75": 2.0, "elev_q25": 3.0}
    row = pd.Series({"peak_fc": 2.5, "elev_ft": 1.0, "prior": 3, "tidal": 1, "tide_fc": 3.5}).rename("r")
    out = flood.reasons(type("R", (), row.to_dict())(), ref)
    assert out == ["heavy forecast rain", "low elevation", "3 past incidents here", "high tide on a tidal road"]
    calm = type("R", (), {"peak_fc": 0.5, "elev_ft": 9, "prior": 0, "tidal": 0, "tide_fc": 1})()
    assert flood.reasons(calm, ref) == ["combined factors"]


def test_model_better_than_a_random_ranking():
    d = flood.make_data(seed=4)
    tr, te = flood.split_by_storm(d)
    m = flood.fit(tr, "logit")
    r_model, _ = flood.recall_at_k(te, m.predict_proba(te[flood.FEATURES])[:, 1])
    r_rand, _ = flood.recall_at_k(te, np.random.default_rng(0).random(len(te)))
    assert r_model > 1.5 * r_rand


def test_ceiling_is_not_a_feature():
    assert "p_true" not in flood.FEATURES and "incident" not in flood.FEATURES
