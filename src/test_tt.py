"""Leakage and correctness checks for tt.py. Run: python -m pytest test_tt.py -q"""
import numpy as np
import pandas as pd

import tt


def toy(m=3000, n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"s": rng.integers(0, n, m) * 7 + 3, "d": rng.integers(0, n, m) * 7 + 3,
                       "r": rng.choice([-1, 1], m, p=[0.2, 0.8]), "t": np.sort(rng.integers(0, 10**6, m))})
    return df


def with_flipped(g, i):
    y = g.y.copy()
    y[i] = 1 - y[i]
    return tt.Graph(g.name, g.s, g.d, y, g.t, g.n)


def test_views_never_read_scored_signs():
    g = tt.load("toy", df=toy())
    for v in (tt.target_view(g), tt.target_fit_view(g), tt.source_view(g, 1.0), tt.source_view(g, 0.4),
              tt.target_fit_view(g, signs=g.win(0, 0.2)[::3]), tt.target_view_strict(g),
              tt.target_fit_view_strict(g), tt.source_view_strict(g, 1.0)):
        assert len(np.intersect1d(v.score, v.signs)) == 0


def test_flipping_a_scored_label_changes_no_feature():
    g = tt.load("toy", df=toy())
    for make in (tt.target_view, tt.target_fit_view, lambda x: tt.source_view(x, 1.0),
                 tt.target_view_strict, tt.target_fit_view_strict):
        v = make(g)
        base = tt.features(v)
        for i in v.score[[0, len(v.score) // 2, -1]]:
            g2 = with_flipped(g, i)
            assert np.array_equal(base, tt.features(make(g2)))


def test_leaked_variant_does_read_scored_labels():
    g = tt.load("toy", df=toy())
    v = tt.target_view(g)
    i = v.score[0]
    a = tt.features(v, leak=True)
    b = tt.features(tt.target_view(with_flipped(g, i)), leak=True)
    assert not np.array_equal(a, b)


def test_features_do_not_depend_on_node_ids():
    df = toy()
    a = tt.features(tt.target_view(tt.load("toy", df=df)))
    b = tt.features(tt.target_view(tt.load("toy", ids="random", seed=5, df=df)))
    assert np.allclose(a, b)


def test_weighted_auc_matches_plain_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.integers(0, 20, 500).astype(float)   # many ties
    assert abs(tt.weighted_auc(s, y, np.ones((1, 500), np.float32))[0] - tt.auc(s, y)) < 1e-6


def test_strict_views_read_signs_only_from_earlier_windows():
    g = tt.load("toy", df=toy())
    for v in (tt.target_view_strict(g), tt.target_fit_view_strict(g), tt.source_view_strict(g, 1.0)):
        assert v.rolling and v.topo.max() < v.score.min() and v.signs.max() < v.score.min()


def test_strict_features_ignore_every_later_edge():
    """Deleting all edges after 30% of the timeline leaves the rolling features of
    edges scored in [20%, 30%) unchanged: nothing at or after a scored edge's chunk is read."""
    g = tt.load("toy", df=toy())
    cut = int(0.3 * g.m)
    g2 = tt.Graph(g.name, g.s[:cut], g.d[:cut], g.y[:cut], g.t[:cut], g.n)
    score, signs = np.arange(int(0.2 * g.m), cut), g.win(0.0, 0.2)
    a = tt.features(tt.View(g, score, signs, signs, True))
    b = tt.features(tt.View(g2, score, signs, signs, True))
    assert np.array_equal(a, b)
    # and the batch view of the same edges does read later edges
    assert not np.array_equal(tt.features(tt.View(g, score, np.arange(int(0.4 * g.m)), signs)),
                              tt.features(tt.View(g2, score, np.arange(cut), signs)))



def test_warm_start_lr_with_zero_prior_is_plain_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    import exp_fewshot
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] + rng.normal(size=400) > 1.2).astype(float)
    w, b = exp_fewshot.map_lr(x, y, np.zeros(5), 0.0)
    ref = LogisticRegression(class_weight="balanced", tol=1e-8, max_iter=5000).fit(x, y)
    assert np.allclose(w, ref.coef_[0], atol=1e-3) and abs(b - ref.intercept_[0]) < 1e-3
    w2, _ = exp_fewshot.map_lr(x[:0], y[:0], np.ones(5), 0.5)   # no labels: the prior itself
    assert np.array_equal(w2, np.ones(5))
