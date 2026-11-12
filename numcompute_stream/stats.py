"""
stats.py -- Chunk-based (streaming) statistics.

Provides:
    chunk_moments(X)
        -> per-column (count, mean, M2) of one chunk, NaN ignored
    merge_moments(a, b)
        -> combine two sets of moments exactly (Chan et al. parallel update)
    StreamingStats(quantile_capacity=None, histogram_bins=None, ...)
        -> running per-column count / mean / variance / min / max, with
           optional quantiles and histograms; fed by update_stats(X_chunk)
    StreamingQuantile(capacity=10_000, random_state=0)
        -> quantiles from a fixed-size reservoir sample (exact while the
           stream is shorter than the reservoir)
    StreamingHistogram(bins=10, range=None, window=None)
        -> per-column histogram with fixed edges, optionally over a sliding
           window of the most recent chunks

Why chunk-based?
----------------
The group package's ``numcompute.stats.StreamingStats`` updates one scalar
at a time (Welford). That is exact but needs a Python loop over every value.
Here a whole chunk is reduced with vectorised NumPy calls and the result is
*merged* into the running state with Chan's formula, which is the same
algebra as Welford's update applied to a block instead of a single value:

    n     = n_a + n_b
    delta = mean_b - mean_a
    mean  = mean_a + delta * n_b / n
    M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

so the cost per chunk is O(chunk size) in compiled code and O(n_features)
in Python. Merging is exact (no approximation) and numerically stable,
because it never forms the catastrophic-cancellation-prone sum of squares
``sum(x**2) - n * mean**2``.

Conventions
-----------
* Input chunks are 2-D ``(n_samples, n_features)``; statistics are always
  per column (``axis=0``), returned as arrays of shape ``(n_features,)``.
* NaN means "missing" and is skipped; each column keeps its own count.
* A statistic of a column that has seen no values is NaN (never an error),
  so a stream that starts with an all-NaN chunk does not crash.
* Empty chunks (0 rows) are accepted and change nothing.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Optional, Tuple

import numpy as np

from ._validation import check_n_features, check_numeric, check_positive_int

__all__ = [
    "chunk_moments",
    "merge_moments",
    "StreamingStats",
    "StreamingQuantile",
    "StreamingHistogram",
]

Moments = Tuple[np.ndarray, np.ndarray, np.ndarray]


# ----------------------------------------------------------------------------
# Moments of one chunk, and the exact merge
# ----------------------------------------------------------------------------
def chunk_moments(X) -> Moments:
    """Per-column count, mean and sum of squared deviations of one chunk.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Numeric chunk; NaN is ignored.

    Returns
    -------
    count : ndarray of float, shape (n_features,)
        Non-NaN values per column.
    mean : ndarray of float, shape (n_features,)
        Column means (0.0 where ``count == 0``).
    M2 : ndarray of float, shape (n_features,)
        Sum of squared deviations from the column mean.

    Complexity
    ----------
    O(n_samples * n_features), fully vectorised.
    """
    X = check_numeric(X, "chunk_moments")
    mask = ~np.isnan(X)
    count = mask.sum(axis=0).astype(np.float64)
    total = np.where(mask, X, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    dev = np.where(mask, X - mean, 0.0)
    M2 = (dev * dev).sum(axis=0)
    return count, mean, M2


def merge_moments(a: Moments, b: Moments) -> Moments:
    """Combine the moments of two disjoint sets of values exactly.

    Parameters
    ----------
    a, b : tuple of (count, mean, M2)
        Each element an array of shape ``(n_features,)`` (as returned by
        :func:`chunk_moments`).

    Returns
    -------
    tuple of (count, mean, M2)
        Moments of the union, identical (up to rounding) to computing them
        on all values at once.

    Notes
    -----
    Columns where both sides are empty stay at ``(0, 0, 0)``; no division
    by zero is performed.
    """
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b
    n = n_a + n_b
    safe_n = np.where(n > 0, n, 1.0)
    delta = mean_b - mean_a
    mean = np.where(n > 0, mean_a + delta * n_b / safe_n, 0.0)
    M2 = np.where(n > 0, m2_a + m2_b + delta * delta * n_a * n_b / safe_n, 0.0)
    return n, mean, M2


def _nan_reduce(func, X):
    """Column-wise nan-reduction that returns NaN for all-NaN columns silently."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(X, axis=0)


# ----------------------------------------------------------------------------
# Quantiles: reservoir sampling
# ----------------------------------------------------------------------------
class StreamingQuantile:
    """Approximate per-column quantiles of a stream with bounded memory.

    Keeps a uniform random sample (a *reservoir*) of at most ``capacity``
    rows using Algorithm R, vectorised per chunk. Quantiles are read from
    the reservoir with ``np.nanquantile``.

    * While the stream holds at most ``capacity`` rows the reservoir is the
      whole stream, so the quantiles are **exact**.
    * After that every row seen so far has the same probability
      ``capacity / n_seen`` of being in the sample, so the estimate is
      unbiased and its error shrinks like ``1 / sqrt(capacity)``.

    Parameters
    ----------
    capacity : int, default 10_000
        Maximum number of rows kept.
    random_state : int or None, default 0
        Seed for the replacement draws (fixed by default, so a stream
        replayed twice gives identical answers).

    Attributes
    ----------
    n_seen_ : int
        Rows offered so far.
    n_features_in_ : int
        Width fixed by the first chunk.
    """

    def __init__(self, capacity: int = 10_000, random_state: Optional[int] = 0):
        self.capacity = check_positive_int(capacity, "capacity", "StreamingQuantile")
        self.random_state = random_state
        self.reset()

    def reset(self) -> "StreamingQuantile":
        """Forget everything seen so far."""
        self.n_seen_ = 0
        self._rng = np.random.default_rng(self.random_state)
        self._res = None
        self._filled = 0
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_
        return self

    def update(self, X_chunk) -> "StreamingQuantile":
        """Offer a chunk of rows to the reservoir.

        Complexity: O(chunk rows * n_features), no Python loop over rows.
        """
        X = check_numeric(X_chunk, "StreamingQuantile.update")
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        if self._res is None:
            self._res = np.empty((self.capacity, X.shape[1]), dtype=np.float64)

        # 1. Fill phase: copy rows straight in until the reservoir is full.
        take = min(self.capacity - self._filled, X.shape[0])
        if take > 0:
            self._res[self._filled:self._filled + take] = X[:take]
            self._filled += take
        rest = X[take:]

        # 2. Replacement phase (Algorithm R): row with global index i
        #    replaces slot j ~ U{0..i} when j < capacity.
        if rest.shape[0]:
            first = self.n_seen_ + take
            idx = np.arange(first, first + rest.shape[0])
            j = self._rng.integers(0, idx + 1)
            keep = j < self.capacity
            # Later rows overwrite earlier ones on repeated slots, matching
            # the sequential algorithm.
            self._res[j[keep]] = rest[keep]
        self.n_seen_ += X.shape[0]
        return self

    @property
    def sample_(self) -> np.ndarray:
        """The current reservoir, shape ``(min(n_seen_, capacity), n_features)``."""
        if self._res is None:
            return np.empty((0, getattr(self, "n_features_in_", 0)))
        return self._res[:self._filled]

    @property
    def is_exact(self) -> bool:
        """True while every row seen is in the reservoir."""
        return self.n_seen_ <= self.capacity

    def quantile(self, q) -> np.ndarray:
        """Per-column quantile(s) of the values seen so far.

        Parameters
        ----------
        q : float or array-like of float in [0, 1]

        Returns
        -------
        ndarray, shape (n_features,) for scalar ``q`` or (len(q), n_features)
            NaN for a column with no non-missing values yet.

        Raises
        ------
        ValueError
            If ``q`` is outside [0, 1] or no rows have been seen.
        """
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 1)) or np.isnan(q_arr).any():
            raise ValueError(f"StreamingQuantile.quantile: q must be in [0, 1], got {q}")
        if self._filled == 0:
            raise ValueError("StreamingQuantile.quantile: no data has been seen yet")
        return _nan_reduce(lambda a, axis: np.nanquantile(a, q_arr, axis=axis), self.sample_)


# ----------------------------------------------------------------------------
# Histograms with fixed edges and an optional sliding window
# ----------------------------------------------------------------------------
class StreamingHistogram:
    """Per-column histogram of a stream, cumulative or over a sliding window.

    Edges are fixed once (from ``range`` or from the first non-empty chunk)
    so counts from different chunks can simply be added. Values below the
    first edge or above the last are counted in ``underflow_`` /
    ``overflow_`` instead of being dropped silently; NaN is counted in
    ``nan_count_``.

    Parameters
    ----------
    bins : int, default 10
        Number of equal-width bins per column.
    range : (float, float) or None
        Lower and upper edge used for every column. If None, each column's
        edges are taken from the min and max of the first chunk.
    window : int or None
        If given, :meth:`counts` reports only the most recent ``window``
        chunks (a sliding window); older chunks fall out automatically.

    Attributes
    ----------
    edges_ : ndarray, shape (n_features, bins + 1)
    """

    def __init__(self, bins: int = 10, range: Optional[Tuple[float, float]] = None,
                 window: Optional[int] = None):
        self.bins = check_positive_int(bins, "bins", "StreamingHistogram")
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"StreamingHistogram: range must satisfy lo < hi, got {range}")
        self.range = range
        self.window = None if window is None else check_positive_int(window, "window", "StreamingHistogram")
        self.reset()

    def reset(self) -> "StreamingHistogram":
        """Clear all counts (the edges are re-learned from the next chunk)."""
        self._total = None
        self._recent = deque()
        self.underflow_ = None
        self.overflow_ = None
        self.nan_count_ = None
        for attr in ("edges_", "n_features_in_"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self

    def _init_edges(self, X: np.ndarray) -> None:
        n_features = X.shape[1]
        if self.range is not None:
            lo = np.full(n_features, float(self.range[0]))
            hi = np.full(n_features, float(self.range[1]))
        else:
            lo = _nan_reduce(np.nanmin, X)
            hi = _nan_reduce(np.nanmax, X)
            lo = np.where(np.isnan(lo), 0.0, lo)
            hi = np.where(np.isnan(hi), 1.0, hi)
            same = hi <= lo                     # zero-variance column
            lo, hi = np.where(same, lo - 0.5, lo), np.where(same, hi + 0.5, hi)
        steps = np.linspace(0.0, 1.0, self.bins + 1)
        self.edges_ = lo[:, None] + (hi - lo)[:, None] * steps[None, :]
        self._total = np.zeros((n_features, self.bins), dtype=np.int64)
        self.underflow_ = np.zeros(n_features, dtype=np.int64)
        self.overflow_ = np.zeros(n_features, dtype=np.int64)
        self.nan_count_ = np.zeros(n_features, dtype=np.int64)

    def _chunk_counts(self, X: np.ndarray) -> np.ndarray:
        """Bin every value of every column with one ``np.bincount`` call."""
        lo, hi = self.edges_[:, 0], self.edges_[:, -1]
        nan = np.isnan(X)
        under = X < lo
        over = X > hi
        inside = ~(nan | under | over)
        width = (hi - lo) / self.bins
        idx = np.floor((np.where(inside, X, lo) - lo) / width).astype(np.int64)
        idx = np.clip(idx, 0, self.bins - 1)    # x == hi goes in the last bin
        col = np.broadcast_to(np.arange(X.shape[1]), X.shape)
        flat = (col * self.bins + idx)[inside]
        counts = np.bincount(flat, minlength=X.shape[1] * self.bins).reshape(X.shape[1], self.bins)
        self.underflow_ += under.sum(axis=0)
        self.overflow_ += over.sum(axis=0)
        self.nan_count_ += nan.sum(axis=0)
        return counts

    def update(self, X_chunk) -> "StreamingHistogram":
        """Add one chunk's values to the histogram."""
        X = check_numeric(X_chunk, "StreamingHistogram.update")
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        if self._total is None:
            self._init_edges(X)
        counts = self._chunk_counts(X)
        self._total += counts
        if self.window is not None:
            self._recent.append(counts)
            if len(self._recent) > self.window:
                self._total -= self._recent.popleft()
        return self

    def counts(self) -> np.ndarray:
        """Counts per bin, shape ``(n_features, bins)`` (window-limited if set)."""
        if self._total is None:
            raise ValueError("StreamingHistogram.counts: no data has been seen yet")
        return self._total.copy()


# ----------------------------------------------------------------------------
# StreamingStats: everything at once
# ----------------------------------------------------------------------------
class StreamingStats:
    """Running per-column statistics of a chunked stream.

    Parameters
    ----------
    decay : float in (0, 1] or None, default None
        None keeps exact cumulative statistics (Chan merge). A float turns
        mean and variance into exponential moving averages over chunks:
        each new chunk gets weight ``decay``, so old data is forgotten and
        the statistics follow a drifting stream.
    quantile_capacity : int or None
        If given, a :class:`StreamingQuantile` with this capacity is kept
        and :meth:`quantile` becomes available.
    histogram_bins : int or None
        If given, a :class:`StreamingHistogram` is kept and
        :meth:`histogram` becomes available.
    histogram_range, histogram_window :
        Passed to :class:`StreamingHistogram`.
    random_state : int or None, default 0
        Seed for the quantile reservoir.

    Examples
    --------
    >>> import numpy as np
    >>> s = StreamingStats()
    >>> for chunk in np.array_split(np.arange(10.0).reshape(-1, 1), 3):
    ...     _ = s.update_stats(chunk)
    >>> float(s.mean[0]), float(s.variance()[0])
    (4.5, 8.25)
    """

    def __init__(self, decay: Optional[float] = None, quantile_capacity: Optional[int] = None,
                 histogram_bins: Optional[int] = None, histogram_range=None,
                 histogram_window: Optional[int] = None, random_state: Optional[int] = 0):
        if decay is not None and not (0.0 < float(decay) <= 1.0):
            raise ValueError(f"StreamingStats: decay must be in (0, 1], got {decay}")
        self.decay = decay
        self.quantile_capacity = quantile_capacity
        self.histogram_bins = histogram_bins
        self.histogram_range = histogram_range
        self.histogram_window = histogram_window
        self.random_state = random_state
        self.reset()

    def reset(self) -> "StreamingStats":
        """Forget everything seen so far."""
        self.n_chunks_ = 0
        self.n_samples_seen_ = 0
        self._count = self._mean = self._m2 = None
        self._var_ema = None
        self._min = self._max = None
        self._quant = (StreamingQuantile(self.quantile_capacity, self.random_state)
                       if self.quantile_capacity else None)
        self._hist = (StreamingHistogram(self.histogram_bins, self.histogram_range, self.histogram_window)
                      if self.histogram_bins else None)
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_
        return self

    # -- the update API -------------------------------------------------------
    def update_stats(self, X_chunk) -> "StreamingStats":
        """Fold one chunk into the running statistics.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
            NaN values are skipped per column. An empty chunk is a no-op.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the chunk is not 2-D, contains infinity, or its width differs
            from earlier chunks.

        Complexity
        ----------
        O(n_samples * n_features) vectorised work per chunk; O(n_features)
        memory for the moments (plus the optional reservoir / histogram).
        """
        X = check_numeric(X_chunk, "StreamingStats.update_stats")
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        cnt, mean, m2 = chunk_moments(X)
        cmin, cmax = _nan_reduce(np.nanmin, X), _nan_reduce(np.nanmax, X)

        if self._count is None:
            self._count, self._mean, self._m2 = cnt, mean, m2
            self._var_ema = np.divide(m2, cnt, out=np.zeros_like(m2), where=cnt > 0)
            self._min, self._max = cmin, cmax
        else:
            if self.decay is None:
                self._count, self._mean, self._m2 = merge_moments((self._count, self._mean, self._m2),
                                                                  (cnt, mean, m2))
            else:
                self._ema_update(cnt, mean, m2)
            self._min = np.fmin(self._min, cmin)    # fmin ignores NaN
            self._max = np.fmax(self._max, cmax)

        if self._quant is not None:
            self._quant.update(X)
        if self._hist is not None:
            self._hist.update(X)
        self.n_chunks_ += 1
        self.n_samples_seen_ += X.shape[0]
        return self

    # ``update`` is an alias so StreamingStats fits the generic update() protocol.
    update = update_stats

    def _ema_update(self, cnt, mean, m2) -> None:
        """Exponentially weighted mean/variance over chunks (columns with data only)."""
        a = float(self.decay)
        has = cnt > 0
        first = has & (self._count == 0)
        var_b = np.divide(m2, cnt, out=np.zeros_like(m2), where=cnt > 0)
        delta = mean - self._mean
        new_mean = self._mean + a * delta
        # variance of the mixture (1-a)*old + a*new
        new_var = (1 - a) * self._var_ema + a * var_b + a * (1 - a) * delta * delta
        self._mean = np.where(first, mean, np.where(has, new_mean, self._mean))
        self._var_ema = np.where(first, var_b, np.where(has, new_var, self._var_ema))
        self._count = self._count + cnt

    def merge(self, other: "StreamingStats") -> "StreamingStats":
        """Absorb another exact accumulator (e.g. from a parallel worker)."""
        if not isinstance(other, StreamingStats):
            raise TypeError(f"merge expects a StreamingStats, got {type(other).__name__}")
        if self.decay is not None or other.decay is not None:
            raise ValueError("merge is only defined for exact statistics (decay=None)")
        if other._count is None:
            return self
        if self._count is None:
            self.n_features_in_ = other.n_features_in_
            self._count, self._mean, self._m2 = other._count.copy(), other._mean.copy(), other._m2.copy()
            self._min, self._max = other._min.copy(), other._max.copy()
        else:
            if other.n_features_in_ != self.n_features_in_:
                raise ValueError("cannot merge statistics with different numbers of features")
            self._count, self._mean, self._m2 = merge_moments((self._count, self._mean, self._m2),
                                                              (other._count, other._mean, other._m2))
            self._min, self._max = np.fmin(self._min, other._min), np.fmax(self._max, other._max)
        self.n_chunks_ += other.n_chunks_
        self.n_samples_seen_ += other.n_samples_seen_
        return self

    # -- results ------------------------------------------------------------
    def _require(self) -> None:
        if self._count is None:
            raise ValueError("StreamingStats: no data has been seen yet; call update_stats first")

    @property
    def count(self) -> np.ndarray:
        """Non-missing values seen per column, shape ``(n_features,)``."""
        self._require()
        return self._count.astype(np.int64)

    @property
    def mean(self) -> np.ndarray:
        """Per-column mean; NaN for a column with no values yet."""
        self._require()
        return np.where(self._count > 0, self._mean, np.nan)

    def variance(self, ddof: int = 0) -> np.ndarray:
        """Per-column variance with ``ddof`` delta degrees of freedom.

        NaN where a column has ``count <= ddof`` values. With ``decay`` set
        the EMA variance is returned and ``ddof`` is ignored.
        """
        self._require()
        if self.decay is not None:
            return np.where(self._count > 0, self._var_ema, np.nan)
        denom = self._count - ddof
        return np.divide(self._m2, denom, out=np.full_like(self._m2, np.nan), where=denom > 0)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Per-column standard deviation (square root of :meth:`variance`)."""
        return np.sqrt(self.variance(ddof))

    @property
    def min(self) -> np.ndarray:
        self._require()
        return self._min.copy()

    @property
    def max(self) -> np.ndarray:
        self._require()
        return self._max.copy()

    def quantile(self, q) -> np.ndarray:
        """Per-column quantile(s); needs ``quantile_capacity`` to be set."""
        if self._quant is None:
            raise ValueError("StreamingStats was created without quantile_capacity; quantiles are unavailable")
        return self._quant.quantile(q)

    def histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(counts, edges)``; needs ``histogram_bins`` to be set."""
        if self._hist is None:
            raise ValueError("StreamingStats was created without histogram_bins; histograms are unavailable")
        return self._hist.counts(), self._hist.edges_.copy()

    def summary(self) -> dict:
        """All available statistics in one dict of per-column arrays."""
        out = {"count": self.count, "mean": self.mean, "std": self.std(),
               "min": self.min, "max": self.max}
        if self._quant is not None:
            out["median"] = self.quantile(0.5)
        return out
