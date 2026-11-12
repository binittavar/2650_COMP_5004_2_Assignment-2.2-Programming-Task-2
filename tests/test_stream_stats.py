"""Tests for numcompute_stream.stats (chunk-based statistics)."""

import numpy as np
import pytest

from numcompute_stream.stats import (StreamingHistogram, StreamingQuantile, StreamingStats, chunk_moments,
                                     merge_moments)


@pytest.fixture
def data():
    return np.random.default_rng(1).normal(loc=5.0, scale=2.0, size=(997, 3))


@pytest.mark.parametrize("n_chunks", [1, 2, 7, 50])
def test_streamed_mean_and_variance_equal_batch(data, n_chunks):
    s = StreamingStats()
    for chunk in np.array_split(data, n_chunks):
        s.update_stats(chunk)
    np.testing.assert_allclose(s.mean, data.mean(axis=0))
    np.testing.assert_allclose(s.variance(), data.var(axis=0))
    np.testing.assert_allclose(s.std(ddof=1), data.std(axis=0, ddof=1))
    np.testing.assert_array_equal(s.min, data.min(axis=0))
    np.testing.assert_array_equal(s.max, data.max(axis=0))


def test_merge_moments_is_exact_for_disjoint_parts(data):
    merged = merge_moments(chunk_moments(data[:300]), chunk_moments(data[300:]))
    whole = chunk_moments(data)
    for a, b in zip(merged, whole):
        np.testing.assert_allclose(a, b)


def test_large_offset_is_numerically_stable():
    # naive sum-of-squares would lose all precision at this offset
    x = 1e9 + np.random.default_rng(0).normal(size=(10_000, 1))
    s = StreamingStats()
    for chunk in np.array_split(x, 13):
        s.update_stats(chunk)
    np.testing.assert_allclose(s.variance(), x.var(axis=0), rtol=1e-6)


def test_nan_values_are_skipped_per_column():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    s = StreamingStats().update_stats(X[:2]).update_stats(X[2:])
    np.testing.assert_array_equal(s.count, [2, 2])
    np.testing.assert_allclose(s.mean, [2.0, 6.0])


def test_all_nan_column_gives_nan_not_error():
    s = StreamingStats().update_stats(np.array([[np.nan, 1.0], [np.nan, 2.0]]))
    assert np.isnan(s.mean[0]) and s.mean[1] == 1.5
    assert np.isnan(s.variance()[0])


def test_empty_chunk_is_noop_and_no_data_raises():
    s = StreamingStats().update_stats(np.empty((0, 2)))
    with pytest.raises(ValueError, match="no data"):
        _ = s.mean


def test_width_change_between_chunks_raises():
    s = StreamingStats().update_stats(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        s.update_stats(np.ones((2, 3)))


def test_rejects_1d_and_inf():
    with pytest.raises(ValueError, match="reshape"):
        StreamingStats().update_stats(np.ones(3))
    with pytest.raises(ValueError, match="infinity"):
        StreamingStats().update_stats(np.array([[np.inf]]))


def test_ema_follows_a_shift():
    s = StreamingStats(decay=0.5)
    s.update_stats(np.zeros((10, 1)))
    for _ in range(10):
        s.update_stats(np.full((10, 1), 10.0))
    assert s.mean[0] > 9.9          # exact mean would be ~9.1
    exact = StreamingStats().update_stats(np.zeros((10, 1)))
    for _ in range(10):
        exact.update_stats(np.full((10, 1), 10.0))
    assert exact.mean[0] == pytest.approx(100 / 11)


def test_merge_two_accumulators(data):
    a = StreamingStats().update_stats(data[:400])
    b = StreamingStats().update_stats(data[400:])
    a.merge(b)
    np.testing.assert_allclose(a.mean, data.mean(axis=0))
    assert a.n_samples_seen_ == len(data)


def test_quantile_exact_until_capacity(data):
    q = StreamingQuantile(capacity=2000)
    for chunk in np.array_split(data, 9):
        q.update(chunk)
    assert q.is_exact
    np.testing.assert_allclose(q.quantile(0.5), np.median(data, axis=0))


def test_quantile_reservoir_is_close_after_capacity():
    x = np.random.default_rng(3).uniform(size=(50_000, 1))
    q = StreamingQuantile(capacity=5000, random_state=0)
    for chunk in np.array_split(x, 25):
        q.update(chunk)
    assert not q.is_exact
    assert q.sample_.shape == (5000, 1)
    assert abs(q.quantile(0.9)[0] - 0.9) < 0.02


def test_quantile_bad_q_raises():
    q = StreamingQuantile().update(np.ones((3, 1)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        q.quantile(1.5)


def test_histogram_counts_match_numpy(data):
    h = StreamingHistogram(bins=8, range=(0.0, 10.0))
    for chunk in np.array_split(data, 6):
        h.update(chunk)
    for j in range(data.shape[1]):
        inside = data[(data[:, j] >= 0) & (data[:, j] <= 10), j]
        expected, _ = np.histogram(inside, bins=8, range=(0, 10))
        np.testing.assert_array_equal(h.counts()[j], expected)
    assert (h.counts().sum(axis=1) + h.underflow_ + h.overflow_ == len(data)).all()


def test_histogram_sliding_window_forgets_old_chunks():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0), window=2)
    h.update(np.full((5, 1), 0.5))       # bin 0
    h.update(np.full((3, 1), 1.5))       # bin 1
    h.update(np.full((4, 1), 1.5))       # first chunk falls out
    np.testing.assert_array_equal(h.counts(), [[0, 7]])


def test_histogram_zero_variance_column_gets_valid_edges():
    h = StreamingHistogram(bins=4).update(np.full((5, 1), 3.0))
    assert h.edges_[0, 0] < 3.0 < h.edges_[0, -1]
    assert h.counts().sum() == 5


def test_stats_with_quantile_and_histogram_options(data):
    s = StreamingStats(quantile_capacity=5000, histogram_bins=5)
    s.update_stats(data)
    counts, edges = s.histogram()
    assert counts.shape == (3, 5) and edges.shape == (3, 6)
    assert set(s.summary()) == {"count", "mean", "std", "min", "max", "median"}
