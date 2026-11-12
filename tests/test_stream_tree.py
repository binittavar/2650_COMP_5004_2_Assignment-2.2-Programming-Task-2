"""Tests for numcompute_stream.tree (CART tree with buffered retraining)."""

import numpy as np
import pytest

from numcompute_stream._validation import NotFittedError
from numcompute_stream.tree import DecisionTreeClassifier, best_split, entropy, gini

# The twelve students from the course notes: submitted, attended, study_group -> passed
STUDENTS = np.array([
    [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 0],
    [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
], dtype=float)


def test_impurity_values():
    assert gini([5, 5]) == pytest.approx(0.5)
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert gini([4, 0]) == 0 and entropy([4, 0]) == 0
    assert gini([0, 0]) == 0                       # empty node, no division by zero
    np.testing.assert_allclose(gini(np.array([[1, 1], [2, 0]])), [0.5, 0.0])


def test_course_example_root_split_is_submitted():
    X, y = STUDENTS[:, :3], STUDENTS[:, 3].astype(int)
    for criterion in ("gini", "entropy"):
        split = best_split(X, y, 2, criterion=criterion)
        assert split["feature"] == 0                 # "submitted?" has the biggest gain
    # before 0.486, after 7/12 * 0.408 + 5/12 * 0 = 0.238, gain 0.248 (course tab 03)
    assert best_split(X, y, 2)["gain"] == pytest.approx(0.486 - 0.238, abs=2e-3)


def test_course_example_depth2_tree_misclassifies_only_student5():
    X, y = STUDENTS[:, :3], STUDENTS[:, 3].astype(int)
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    wrong = np.flatnonzero(tree.predict(X) != y)
    assert wrong.tolist() == [4]                     # student 5 (index 4) is the fluke
    assert tree.get_depth() == 2 and tree.get_n_leaves() == 3


def test_vectorised_split_matches_brute_force():
    rng = np.random.default_rng(5)
    X = rng.integers(0, 6, size=(60, 3)).astype(float)
    y = rng.integers(0, 3, size=60)
    best = -1.0
    parent = gini(np.bincount(y, minlength=3))
    for f in range(3):
        for t in np.unique(X[:, f])[:-1]:
            m = X[:, f] <= t
            g = parent - (m.sum() * gini(np.bincount(y[m], minlength=3))
                          + (~m).sum() * gini(np.bincount(y[~m], minlength=3))) / 60
            best = max(best, g)
    assert best_split(X, y, 3)["gain"] == pytest.approx(best)


def test_threshold_lies_between_distinct_values_and_ties_stay_together():
    X = np.array([[1.0], [1.0], [2.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    split = best_split(X, y, 2)
    assert 1.0 <= split["threshold"] < 2.0
    assert split["n_left"] == 2


def test_adjacent_floats_threshold_is_safe():
    a = 1.0
    b = np.nextafter(a, 2.0)
    split = best_split(np.array([[a], [b]]), np.array([0, 1]), 2)
    assert a <= split["threshold"] < b


def test_tie_between_features_goes_to_lowest_index():
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)   # identical columns
    assert best_split(X, np.array([0, 0, 1, 1]), 2)["feature"] == 0


def test_constant_features_give_no_split_and_a_leaf():
    X = np.ones((10, 2))
    y = np.array([0, 1] * 5)
    assert best_split(X, y, 2) is None
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.get_n_leaves() == 1
    assert tree.predict(X).tolist() == [0] * 10      # 5-5 vote tie -> first class


def test_nan_goes_right_and_missingness_can_be_learned():
    X = np.array([[1.0], [2.0], [np.nan], [np.nan]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert tree.score(X, y) == 1.0
    assert tree.predict(np.array([[np.nan], [1.5]])).tolist() == [1, 0]


def test_string_labels_and_predict_proba_rows_sum_to_one():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array(["low", "low", "mid", "mid", "high", "high"])
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert tree.score(X, y) == 1.0
    proba = tree.predict_proba(X)
    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_max_depth_min_samples_and_min_impurity_decrease_limit_growth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.integers(0, 2, size=300)                 # pure noise: a deep tree overfits
    assert DecisionTreeClassifier(max_depth=3).fit(X, y).get_depth() <= 3
    leafy = DecisionTreeClassifier(max_depth=None, min_samples_leaf=30).fit(X, y)
    assert leafy._nodes["n_samples"][leafy._nodes["feature"] == -1].min() >= 30
    assert DecisionTreeClassifier(max_depth=None, min_impurity_decrease=0.2).fit(X, y).get_n_leaves() == 1


def test_max_features_subset_is_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    y = (X[:, 3] > 0).astype(int)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=7).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=7).fit(X, y)
    np.testing.assert_array_equal(a._nodes["feature"], b._nodes["feature"])


def test_feature_importances_point_at_the_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 5))
    y = (X[:, 2] > 0.3).astype(int)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert np.argmax(tree.feature_importances_) == 2
    assert tree.feature_importances_.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- streaming
def _stream(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, (X[:, 0] > 0).astype(int)


def test_partial_fit_retrains_only_when_buffer_is_full():
    X, y = _stream()
    tree = DecisionTreeClassifier(buffer_size=250)
    tree.partial_fit(X[:100], y[:100])
    assert not tree.is_fitted and tree.buffer_fill_ == 100
    tree.partial_fit(X[100:200], y[100:200])
    assert tree.n_retrains_ == 0
    tree.partial_fit(X[200:300], y[200:300])         # 300 >= 250 -> retrain
    assert tree.n_retrains_ == 1 and tree.buffer_fill_ == 0
    assert tree.retrain_log_[0]["n_samples"] == 300


def test_predict_uses_last_tree_between_retrains():
    X, y = _stream()
    tree = DecisionTreeClassifier(buffer_size=200).partial_fit(X[:200], y[:200])
    before = tree.predict(X).copy()
    tree.partial_fit(X[200:350], 1 - y[200:350])     # buffered, not yet used
    np.testing.assert_array_equal(tree.predict(X), before)


def test_number_of_retrains_for_many_chunks():
    X, y = _stream(1000)
    tree = DecisionTreeClassifier(buffer_size=100)
    for chunk in np.array_split(np.arange(1000), 40):  # chunks of 25
        tree.partial_fit(X[chunk], y[chunk])
    assert tree.n_retrains_ == 10 and tree.n_samples_seen_ == 1000
    assert tree.score(X, y) > 0.95


def test_chunk_larger_than_buffer_triggers_one_retrain():
    X, y = _stream(600)
    tree = DecisionTreeClassifier(buffer_size=100).partial_fit(X, y)
    assert tree.n_retrains_ == 1 and tree.retrain_log_[0]["n_samples"] == 600


def test_predict_before_first_retrain_raises_helpful_error():
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(np.ones((10, 2)), np.zeros(10))
    with pytest.raises(NotFittedError, match="500"):
        tree.predict(np.ones((1, 2)))


def test_flush_trains_on_a_partly_filled_buffer():
    X, y = _stream(50)
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(X, y).flush()
    assert tree.is_fitted and tree.buffer_fill_ == 0
    assert DecisionTreeClassifier().flush().n_retrains_ == 0   # empty buffer -> no-op


def test_new_class_after_retrain_gets_zero_probability():
    X = np.arange(4, dtype=float).reshape(-1, 1)
    tree = DecisionTreeClassifier(buffer_size=4).partial_fit(X, np.array([0, 0, 1, 1]))
    tree.partial_fit(np.array([[9.0]]), np.array([2]))          # buffered only
    proba = tree.predict_proba(X)
    assert proba.shape == (4, 3) and np.all(proba[:, 2] == 0)


def test_single_class_buffer_and_empty_chunk():
    tree = DecisionTreeClassifier(buffer_size=3).partial_fit(np.empty((0, 2)), np.empty(0))
    assert tree.n_samples_seen_ == 0
    tree.partial_fit(np.random.default_rng(0).normal(size=(3, 2)), np.array([1, 1, 1]))
    assert tree.get_n_leaves() == 1 and tree.predict(np.zeros((2, 2))).tolist() == [1, 1]


def test_chunk_width_change_raises_with_hint():
    tree = DecisionTreeClassifier(buffer_size=10).partial_fit(np.ones((2, 3)), np.array([0, 1]))
    with pytest.raises(ValueError, match="max_categories"):
        tree.partial_fit(np.ones((2, 4)), np.array([0, 1]))


@pytest.mark.parametrize("kwargs", [{"max_depth": -1}, {"buffer_size": 0}, {"criterion": "mse"},
                                    {"min_samples_split": 1}, {"max_features": 0.0}, {"max_features": "all"}])
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DecisionTreeClassifier(**kwargs)


def test_shape_and_label_errors():
    tree = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="same rows"):
        tree.fit(np.ones((3, 1)), np.array([0, 1]))
    with pytest.raises(ValueError, match="NaN"):
        tree.fit(np.ones((2, 1)), np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="reshape"):
        tree.fit(np.ones(3), np.array([0, 1, 0]))


def test_export_text_lists_the_question():
    X, y = STUDENTS[:, :3], STUDENTS[:, 3].astype(int)
    text = DecisionTreeClassifier(max_depth=2).fit(X, y).export_text(["submitted", "attended", "study_group"])
    assert text.splitlines()[0] == "|--- submitted <= 0.5"
    assert "attended" in text


def test_memory_footprint_counts_buffer():
    tree = DecisionTreeClassifier(buffer_size=1000).partial_fit(np.ones((100, 4)), np.zeros(100))
    assert tree.memory_footprint() == 100 * 4 * 8 + 100 * 8
