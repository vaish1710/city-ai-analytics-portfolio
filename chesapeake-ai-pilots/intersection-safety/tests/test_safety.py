import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
import safety as S


def test_eb_hand_examples():
    assert S.eb_estimate(np.array([6.0]), np.array([2.0]), 0.5)[0] == 4.0       # w = 1/(1+0.5*2) = 0.5 -> 0.5*2 + 0.5*6
    assert S.eb_estimate(np.array([9.0]), np.array([0.0]), 0.5)[0] == 0.0       # mu = 0 -> w = 1 -> prediction only
    assert S.eb_estimate(np.array([3.0]), np.array([3.0]), 2.0)[0] == 3.0       # observed equals predicted
    big_k = S.eb_estimate(np.array([10.0]), np.array([2.0]), 1000.0)[0]
    assert 9.9 < big_k < 10.0                                                  # huge k -> trust the observed history


def test_overdispersion_recovers_known_k():
    rng = np.random.default_rng(0)
    mu = rng.uniform(1, 5, 40000)
    k = 0.5
    y = rng.negative_binomial(1 / k, (1 / k) / (1 / k + mu))               # Var = mu + k mu^2
    assert abs(S.overdispersion(y, mu) - k) < 0.05
    assert S.overdispersion(rng.poisson(mu), mu) < 0.05                       # plain Poisson -> near the floor


def test_ranking_ignores_test_years():
    d = S.make_sites(seed=3)
    e = d.copy()
    for y in S.TEST_YEARS:
        e[f"crashes_{y}"] = 0
    a, _ = S.rank_scores(d); b, _ = S.rank_scores(e)
    for name in a:
        assert np.allclose(a[name], b[name])


def test_rate_ranking_favors_low_volume_at_equal_counts():
    d = pd.DataFrame({"aadt": [20000.0, 2000.0], "crashes_2019": [2, 2], "crashes_2020": [2, 2], "crashes_2021": [2, 2]})
    y = S.train_total(d)
    assert (y / d.aadt.values).argmax() == 1


def test_top_idx_and_future_share_hand_check():
    assert list(S.top_idx(np.array([1, 5, 5, 2]), frac=0.5)) == [1, 2]          # ties -> lower id first
    d = pd.DataFrame({"crashes_2022": [3, 0, 1, 0], "crashes_2023": [1, 0, 3, 0]})
    assert S.future_share(d, np.array([9, 1, 8, 0]), frac=0.5) == 1.0           # top 2 sites hold all 8 future crashes
    assert S.future_share(d, np.array([0, 9, 0, 8]), frac=0.5) == 0.0


def test_regression_to_mean_present_and_top_group_selected_on_history():
    pre, post = np.mean([S.regression_to_mean(S.make_sites(seed=s)) for s in range(1, 16)], axis=0)
    assert post < pre


def test_methods_beat_random_and_rate_is_worst_of_history_methods():
    d = S.make_sites(seed=5)
    sc, _ = S.rank_scores(d)
    shares = {n: S.future_share(d, s) for n, s in sc.items()}
    assert shares["empirical Bayes"] > 0.15
    assert shares["crashes per vehicle"] < shares["raw crash count"]
