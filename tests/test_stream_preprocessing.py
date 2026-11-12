"""Tests for numcompute_stream.preprocessing (incremental transformers)."""

import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler as BatchScaler
from numcompute_stream._validation import NotFittedError
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


@pytest.fixture
def X():
    return np.random.default_rng(2).normal(3.0, 4.0, size=(500, 4))


def test_scaler_partial_fit_matches_group_batch_scaler(X):
    sc = StandardScaler()
    for chunk in np.array_split(X, 11):
        sc.partial_fit(chunk)
    batch = BatchScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, batch.mean_)
    np.testing.assert_allclose(sc.transform(X), batch.transform(X))


def test_scaler_fit_resets_previous_state(X):
    sc = StandardScaler().partial_fit(X + 100)
    sc.fit(X)
    np.testing.assert_allclose(sc.mean_, X.mean(axis=0))
    assert sc.n_samples_seen_ == len(X)


def test_scaler_zero_variance_chunk_maps_to_zero():
    sc = StandardScaler().partial_fit(np.full((4, 2), 7.0))
    out = sc.transform(np.full((2, 2), 7.0))
    assert np.all(out == 0) and np.all(np.isfinite(out))


def test_scaler_inverse_transform_roundtrip(X):
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X)


def test_scaler_nan_passes_through():
    sc = StandardScaler().partial_fit(np.array([[1.0], [np.nan], [3.0]]))
    assert sc.mean_[0] == 2.0
    assert np.isnan(sc.transform(np.array([[np.nan]])))[0, 0]


def test_scaler_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(np.ones((1, 1)))


def test_scaler_empty_chunk_does_not_fit():
    sc = StandardScaler().partial_fit(np.empty((0, 3)))
    with pytest.raises(NotFittedError):
        sc.transform(np.ones((1, 3)))


def test_minmax_running_range_widens():
    mm = MinMaxScaler().partial_fit(np.array([[0.0], [1.0]])).partial_fit(np.array([[4.0]]))
    np.testing.assert_allclose(mm.transform(np.array([[2.0]])), [[0.5]])
    assert MinMaxScaler(clip=True).fit(np.array([[0.0], [1.0]])).transform(np.array([[5.0]]))[0, 0] == 1.0


def test_imputer_mean_updates_on_the_fly():
    imp = Imputer("mean").partial_fit(np.array([[1.0], [np.nan]]))
    assert imp.statistics_[0] == 1.0
    imp.partial_fit(np.array([[5.0]]))
    np.testing.assert_allclose(imp.transform(np.array([[np.nan]])), [[3.0]])


def test_imputer_median_exact_on_short_stream():
    x = np.array([[5.0], [1.0], [np.nan], [3.0], [100.0]])
    imp = Imputer("median")
    for chunk in np.array_split(x, 3):
        imp.partial_fit(chunk)
    assert imp.statistics_[0] == 4.0


def test_imputer_most_frequent_merges_counts_and_breaks_ties_low():
    imp = Imputer("most_frequent")
    imp.partial_fit(np.array([[2.0], [2.0], [9.0]]))
    imp.partial_fit(np.array([[9.0], [np.nan]]))
    assert imp.statistics_[0] == 2.0          # 2 and 9 tie -> smaller value


def test_imputer_column_without_values_falls_back_to_fill_value():
    imp = Imputer("mean", fill_value=-1).partial_fit(np.array([[np.nan, 1.0]]))
    np.testing.assert_array_equal(imp.statistics_, [-1.0, 1.0])
    assert SimpleImputer is Imputer


def test_imputer_rejects_bad_strategy():
    with pytest.raises(ValueError, match="strategy"):
        Imputer("mode")


def test_onehot_expands_without_moving_old_columns():
    enc = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    before = enc.transform(np.array([["a"]]))
    enc.partial_fit(np.array([["c"]]))
    after = enc.transform(np.array([["a"], ["c"]]))
    np.testing.assert_array_equal(after[0, :2], before[0])        # 'a' column unchanged
    assert enc.get_feature_names_out().tolist() == ["x0=a", "x0=b", "x0=c"]
    np.testing.assert_array_equal(after[1], [0, 0, 1])


def test_onehot_unknown_category_ignore_and_error():
    enc = OneHotEncoder().fit(np.array([["a"]]))
    np.testing.assert_array_equal(enc.transform(np.array([["z"]])), [[0.0]])
    strict = OneHotEncoder(handle_unknown="error").fit(np.array([["a"]]))
    with pytest.raises(ValueError, match="unknown category"):
        strict.transform(np.array([["z"]]))


def test_onehot_max_categories_fixes_width_and_uses_other():
    enc = OneHotEncoder(max_categories=3).partial_fit(np.array([["a"]]))
    assert enc.n_features_out_ == 3
    enc.partial_fit(np.array([["b"], ["c"], ["d"]]))
    assert enc.n_features_out_ == 3
    out = enc.transform(np.array([["a"], ["b"], ["d"], ["never-seen"]]))
    np.testing.assert_array_equal(out, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])


def test_onehot_missing_text_is_its_own_category():
    enc = OneHotEncoder().fit(np.array([["yes"], [""], ["no"]]))
    assert "" in enc.categories_[0]
    assert enc.transform(np.array([[""]])).sum() == 1
