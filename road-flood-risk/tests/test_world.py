import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, pytest
import world, flood
W = world


def grid(M, basins):
    T, G = M.shape
    return pd.DataFrame({"storm": np.repeat(np.arange(T), G), "segment": np.tile(np.arange(G), T), "basin": np.tile(basins, T), "incident": M.ravel().astype(int)})


M = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]])
BASINS = [0, 0, 1]


def col(df, name, seg):
    return df[df.segment == seg].sort_values("storm")[name].values


def test_history_hand_computed_all_history():
    h = world.add_history(grid(M, BASINS), half_life=1)
    assert list(col(h, "prior", 0)) == [0, 1, 1, 2, 2, 2] and list(col(h, "prior", 1)) == [0, 0, 0, 0, 1, 1]
    assert col(h, "prior_rate", 0)[3] == pytest.approx(2 / 8)
    assert col(h, "prior_decay", 0) == pytest.approx([0, 1, 0.5, 1.25, 0.625, 0.3125])
    assert col(h, "basin_rate", 0) == pytest.approx(col(h, "prior_rate", 1))        # basin 0 = segments 0 and 1: each other's peer
    assert col(h, "basin_rate", 1) == pytest.approx(col(h, "prior_rate", 0))
    assert (col(h, "basin_rate", 2) == 0).all()                                      # alone in its basin


def test_history_window():
    h = world.add_history(grid(M, BASINS), window=2)
    assert list(col(h, "prior", 0)) == [0, 1, 1, 1, 1, 0]                            # last two earlier storms only
    assert col(h, "prior_rate", 0)[3] == pytest.approx(1 / 7) and col(h, "prior_rate", 0)[1] == pytest.approx(1 / 6)


def test_no_future_leakage_in_any_history_feature():
    d = world.make_world(seed=2)
    s = 50
    e = d.copy(); e.loc[e.storm >= s, "incident"] = 1 - e.loc[e.storm >= s, "incident"]
    a = world.add_history(d.drop(columns=world.HIST_FEATS + ["prior"]), window=None)
    b = world.add_history(e.drop(columns=world.HIST_FEATS + ["prior"]), window=None)
    for c in world.HIST_FEATS + ["prior"]:
        assert np.allclose(a[a.storm <= s][c].values, b[b.storm <= s][c].values), c


def test_basin_rate_excludes_the_segment_itself():
    base = grid(M, BASINS); flipped = base.copy(); flipped.loc[flipped.segment == 0, "incident"] = 1 - flipped.loc[flipped.segment == 0, "incident"]
    a, b = world.add_history(base), world.add_history(flipped)
    assert np.allclose(col(a, "basin_rate", 0), col(b, "basin_rate", 0))            # own history does not feed own basin_rate
    assert not np.allclose(col(a, "basin_rate", 1), col(b, "basin_rate", 1))        # but it does feed its peer's


def test_recall_k_hand_check_and_agreement_with_flood_module():
    df = pd.DataFrame({"storm": [0, 0, 0, 1, 1, 1], "segment": [0, 1, 2] * 2, "incident": [1, 0, 1, 0, 0, 0]})
    r, fa = world.recall_k(df, np.array([0.9, 0.8, 0.1, 0.5, 0.4, 0.3]), k=2)
    assert (r, fa) == (0.5, 1.5)
    d = world.make_world(seed=4); s = np.random.default_rng(0).random(len(d))
    assert world.recall_k(d, s, k=8) == pytest.approx(flood.recall_at_k(d, s, k=8))
    t = world.recall_k(pd.DataFrame({"storm": [0] * 3, "segment": [0, 1, 2], "incident": [0, 1, 0]}), np.ones(3), k=1)
    assert t[0] == 0.0                                                              # ties go to the lower segment id (segment 0), which has no incident


def test_forecasts_are_constant_within_a_storm():
    d = world.make_world(seed=1)
    for c in ("peak_fc", "total_fc", "tide_fc"):
        assert (d.groupby("storm")[c].nunique() == 1).all()


def test_drift_only_changes_things_after_the_change_point():
    a = world.make_world(seed=7, drift_frac=0.0); b = world.make_world(seed=7, drift_frac=0.6, drift_at=0.5)
    cut = int(0.5 * 120)
    assert np.allclose(a[a.storm < cut].p_true, b[b.storm < cut].p_true) and (a[a.storm < cut].flooded.values == b[b.storm < cut].flooded.values).all()
    assert not np.allclose(a[a.storm >= cut].p_true, b[b.storm >= cut].p_true)


def test_reporting_bias_makes_reporting_uneven_across_segments():
    sd = lambda bias: (lambda d: (d.groupby("segment").incident.sum() / d.groupby("segment").flooded.sum().clip(lower=1)).std())(world.make_world(seed=3, n_storms=3000, report_bias=bias))
    assert sd(True) > 1.5 * sd(False)
    d = world.make_world(seed=3, n_storms=1500, report_bias=True)
    assert 0.5 < d.incident.sum() / d.flooded.sum() < 0.9                          # average reporting rate preserved near 0.7


def test_default_world_incident_rate_is_plausible_and_window_is_respected():
    d = world.make_world(seed=1, window=6)
    assert 0.02 < d.incident.mean() < 0.08 and d.prior.max() <= 6


def test_recall_k_works_on_a_held_out_slice_in_any_row_order():
    d = world.make_world(seed=5); te = d[d.storm > 84]
    s = np.random.default_rng(1).random(len(te))
    a = world.recall_k(te, s, k=8)
    perm = np.random.default_rng(2).permutation(len(te))
    assert world.recall_k(te.iloc[perm], s[perm], k=8) == pytest.approx(a)             # row order must not matter
    assert a == pytest.approx(flood.recall_at_k(te, s, k=8))
    ps = world.per_storm(te, s, k=8)
    assert ps.hits.sum() / ps.total.sum() == pytest.approx(a[0])


def test_drift_share_direction_and_timing():
    """Same seed with and without drift: only the drifting share of segments changes, in both directions, only after drift_at."""
    base = W.make_world(n_seg=200, n_storms=60, seed=11, drift_frac=0.0)
    dr = W.make_world(n_seg=200, n_storms=60, seed=11, drift_frac=0.4, drift_at=0.5)
    assert (base.storm.values == dr.storm.values).all() and (base.segment.values == dr.segment.values).all()
    d = dr.p_true.values - base.p_true.values
    after = dr.storm.values >= dr.storm.min() + int(0.5 * (dr.storm.max() - dr.storm.min() + 1))
    assert np.abs(d[~after]).max() < 1e-12                      # nothing changes before the change point
    late = dr[after].assign(d=d[after])
    per_seg = late.groupby("segment").d.mean()
    changed = per_seg.abs() > 1e-9
    assert 0.3 < changed.mean() < 0.5                           # about 40% of segments drift
    assert (per_seg[changed] > 0).any() and (per_seg[changed] < 0).any()   # some get worse, some better
    assert (W.make_world(n_seg=200, n_storms=60, seed=11, drift_frac=0.0).p_true.values == base.p_true.values).all()


def test_tidal_share_and_tide_effect():
    d = W.make_world(n_seg=400, n_storms=30, seed=4)
    share = d.drop_duplicates("segment").tidal.mean()
    assert 0.05 < share < 0.6
    hi = d[d.tide_fc > d.tide_fc.quantile(0.8)]
    assert hi[hi.tidal == 1].p_true.mean() > hi[hi.tidal == 0].p_true.mean()   # tidal roads flood more at high tide
