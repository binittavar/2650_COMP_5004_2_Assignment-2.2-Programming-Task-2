"""
preprocessing.py -- Streaming (incremental) transformers.

Provides:
    StandardScaler(with_mean=True, with_std=True, decay=None)
        -> z-scores from a running mean / variance (exact Chan-Welford merge,
           or an exponential moving average when ``decay`` is set)
    MinMaxScaler(feature_range=(0, 1), clip=False)
        -> rescaling from a running per-column min / max
    Imputer(strategy='mean'|'median'|'most_frequent'|'constant', ...)
        -> replaces NaN with fill values that are re-estimated after every
           chunk (``SimpleImputer`` is an alias)
    OneHotEncoder(handle_unknown='ignore', max_categories=None)
        -> one 0/1 column per category; categories are discovered chunk by
           chunk and appended, so existing output columns never move

Streaming protocol (shared by every class here)
-----------------------------------------------
* ``partial_fit(X_chunk, y=None) -> self`` folds one chunk into the
  learned state without forgetting earlier chunks.
* ``fit(X, y=None) -> self`` forgets everything, then calls ``partial_fit``
  once: it is exactly "a stream with one chunk".
* ``transform(X)`` uses the state learned so far, so it can be called
  between chunks (for example to score a chunk before training on it).
* ``fit_transform`` comes from the group package's ``TransformerMixin``.
* Empty chunks are accepted and change nothing; the first non-empty chunk
  fixes ``n_features_in_`` and every later chunk must match it.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from numcompute.preprocessing import TransformerMixin

from ._validation import check_2d, check_fitted, check_n_features, check_numeric, check_positive_int
from .stats import StreamingQuantile, StreamingStats

__all__ = ["StandardScaler", "MinMaxScaler", "Imputer", "SimpleImputer", "OneHotEncoder"]


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    """Replace 0 and NaN scales with 1 so zero-variance columns map to 0, not inf/NaN."""
    return np.where((scale == 0) | np.isnan(scale), 1.0, scale)


class _StreamingTransformer(TransformerMixin):
    """Base class: ``fit`` = ``reset`` + one ``partial_fit``."""

    def _reset(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y=None):
        """Forget all state, then learn from ``X`` as a single chunk."""
        self._reset()
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_
        return self.partial_fit(X, y)


# ----------------------------------------------------------------------------
# StandardScaler
# ----------------------------------------------------------------------------
class StandardScaler(_StreamingTransformer):
    """Standardise features with a running mean and standard deviation.

    Parameters
    ----------
    with_mean : bool, default True
        Subtract the running mean.
    with_std : bool, default True
        Divide by the running (population) standard deviation.
    decay : float in (0, 1] or None, default None
        None: exact statistics of everything seen (Chan-Welford merge,
        identical to fitting on the concatenated chunks). A float: an
        exponential moving average over chunks, where each new chunk has
        weight ``decay``; use it when the stream drifts.

    Attributes
    ----------
    mean_, var_, scale_ : ndarray, shape (n_features,)
        Current statistics. ``scale_`` is 1 for zero-variance columns and for
        columns that have not seen a value yet.
    n_samples_seen_ : int

    Notes
    -----
    NaN in the input is ignored by ``partial_fit`` and passed through by
    ``transform`` (impute first if the model cannot handle NaN).

    Examples
    --------
    >>> import numpy as np
    >>> sc = StandardScaler()
    >>> _ = sc.partial_fit([[0.0], [2.0]]).partial_fit([[4.0]])
    >>> sc.mean_, np.round(sc.var_, 4)
    (array([2.]), array([2.6667]))
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, decay: Optional[float] = None):
        self.with_mean = with_mean
        self.with_std = with_std
        self.decay = decay
        self._reset()

    def _reset(self) -> None:
        self._stats = StreamingStats(decay=self.decay)
        for attr in ("mean_", "var_", "scale_"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.n_samples_seen_ = 0

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Update the running mean and variance with one chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Complexity
        ----------
        O(n_samples * n_features) vectorised; O(n_features) state.
        """
        X = check_numeric(X, "StandardScaler.partial_fit")
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        self._stats.update_stats(X)
        mean = self._stats.mean
        var = self._stats.variance()
        self.mean_ = np.where(np.isnan(mean), 0.0, mean)
        self.var_ = np.where(np.isnan(var), 0.0, var)
        self.scale_ = _safe_scale(np.sqrt(self.var_))
        self.n_samples_seen_ = self._stats.n_samples_seen_
        return self

    def transform(self, X) -> np.ndarray:
        """Return ``(X - mean_) / scale_`` (each part optional)."""
        check_fitted(self, "mean_")
        X = check_numeric(X, "StandardScaler.transform")
        check_n_features(self, X, reset=False)
        out = X - self.mean_ if self.with_mean else X.copy()
        if self.with_std:
            out = out / self.scale_
        return out

    def inverse_transform(self, X) -> np.ndarray:
        """Undo :meth:`transform`."""
        check_fitted(self, "mean_")
        X = check_numeric(X, "StandardScaler.inverse_transform")
        check_n_features(self, X, reset=False)
        out = X * self.scale_ if self.with_std else X.copy()
        return out + self.mean_ if self.with_mean else out


# ----------------------------------------------------------------------------
# MinMaxScaler
# ----------------------------------------------------------------------------
class MinMaxScaler(_StreamingTransformer):
    """Rescale features into ``feature_range`` using a running min and max.

    Because the range can only widen as data arrives, values transformed
    early may later fall outside ``feature_range``; ``clip=True`` caps them.
    Constant columns map to the lower end of the range.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), clip: bool = False):
        lo, hi = feature_range
        if not lo < hi:
            raise ValueError(f"MinMaxScaler: feature_range must satisfy min < max, got {feature_range}")
        self.feature_range = feature_range
        self.clip = clip
        self._reset()

    def _reset(self) -> None:
        for attr in ("data_min_", "data_max_"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.n_samples_seen_ = 0

    def partial_fit(self, X, y=None) -> "MinMaxScaler":
        """Widen the running min/max with one chunk (NaN ignored)."""
        X = check_numeric(X, "MinMaxScaler.partial_fit")
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cmin, cmax = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        if not hasattr(self, "data_min_"):
            self.data_min_, self.data_max_ = cmin, cmax
        else:
            self.data_min_ = np.fmin(self.data_min_, cmin)
            self.data_max_ = np.fmax(self.data_max_, cmax)
        self.n_samples_seen_ += X.shape[0]
        return self

    def transform(self, X) -> np.ndarray:
        check_fitted(self, "data_min_")
        X = check_numeric(X, "MinMaxScaler.transform")
        check_n_features(self, X, reset=False)
        lo, hi = self.feature_range
        dmin = np.where(np.isnan(self.data_min_), 0.0, self.data_min_)
        span = _safe_scale(np.where(np.isnan(self.data_max_), 1.0, self.data_max_) - dmin)
        out = (X - dmin) / span * (hi - lo) + lo
        return np.clip(out, lo, hi) if self.clip else out


# ----------------------------------------------------------------------------
# Imputer
# ----------------------------------------------------------------------------
_STRATEGIES = ("mean", "median", "most_frequent", "constant")


def _merge_value_counts(old_vals, old_counts, new_vals, new_counts):
    """Merge two (sorted unique values, counts) tables into one."""
    vals = np.concatenate([old_vals, new_vals])
    counts = np.concatenate([old_counts, new_counts])
    uniq, inv = np.unique(vals, return_inverse=True)
    return uniq, np.bincount(inv, weights=counts).astype(np.int64)


class Imputer(_StreamingTransformer):
    """Fill NaN with per-column values that are updated on the fly.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default 'mean'
        * ``'mean'``: running mean (exact, Chan-Welford).
        * ``'median'``: median of a :class:`~numcompute_stream.stats.StreamingQuantile`
          reservoir (exact until ``quantile_capacity`` rows, then an
          unbiased estimate).
        * ``'most_frequent'``: running value counts merged chunk by chunk;
          ties go to the smallest value.
        * ``'constant'``: always ``fill_value``.
    fill_value : float, default 0
        Used by ``'constant'`` and as the fallback for a column that has not
        seen a single value yet (common in the first chunks of a stream).
    quantile_capacity : int, default 10_000
        Reservoir size for ``'median'``.

    Attributes
    ----------
    statistics_ : ndarray, shape (n_features,)
        The value each column's NaNs are replaced with right now.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0, quantile_capacity: int = 10_000):
        if strategy not in _STRATEGIES:
            raise ValueError(f"Imputer: strategy must be one of {_STRATEGIES}, got {strategy!r}")
        fill = float(fill_value)
        if not np.isfinite(fill):
            raise ValueError(f"Imputer: fill_value must be finite, got {fill_value!r}")
        self.strategy = strategy
        self.fill_value = fill
        self.quantile_capacity = quantile_capacity
        self._reset()

    def _reset(self) -> None:
        self._stats = StreamingStats() if self.strategy == "mean" else None
        self._quant = StreamingQuantile(self.quantile_capacity) if self.strategy == "median" else None
        self._counts = None                   # list of (values, counts) per column
        if hasattr(self, "statistics_"):
            del self.statistics_
        self.n_samples_seen_ = 0

    def _mode(self) -> np.ndarray:
        out = np.full(len(self._counts), np.nan)
        for j, (vals, cnt) in enumerate(self._counts):   # loop over columns only
            if cnt.size:
                out[j] = vals[np.argmax(cnt)]            # argmax: first = smallest value
        return out

    def partial_fit(self, X, y=None) -> "Imputer":
        """Update the fill values with one chunk."""
        X = check_numeric(X, "Imputer.partial_fit")
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        n_features = X.shape[1]

        if self.strategy == "constant":
            stats = np.full(n_features, self.fill_value)
        elif self.strategy == "mean":
            stats = self._stats.update_stats(X).mean
        elif self.strategy == "median":
            stats = self._quant.update(X).quantile(0.5)
        else:
            if self._counts is None:
                self._counts = [(np.empty(0), np.empty(0, dtype=np.int64))] * n_features
            for j in range(n_features):
                col = X[:, j]
                vals, cnt = np.unique(col[~np.isnan(col)], return_counts=True)
                self._counts[j] = _merge_value_counts(*self._counts[j], vals, cnt)
            stats = self._mode()

        # Columns that have not seen a value yet fall back to fill_value.
        self.statistics_ = np.where(np.isnan(stats), self.fill_value, stats)
        self.n_samples_seen_ += X.shape[0]
        return self

    def transform(self, X) -> np.ndarray:
        """Return a copy of ``X`` with every NaN replaced by ``statistics_``."""
        check_fitted(self, "statistics_")
        X = check_numeric(X, "Imputer.transform")
        check_n_features(self, X, reset=False)
        return np.where(np.isnan(X), self.statistics_, X)


#: Alias matching the group package's name.
SimpleImputer = Imputer


# ----------------------------------------------------------------------------
# OneHotEncoder
# ----------------------------------------------------------------------------
_OTHER = "__other__"


class OneHotEncoder(_StreamingTransformer):
    """One-hot encode categorical columns whose categories arrive over time.

    Values are compared as strings (``str(value)``), so numbers, strings and
    the empty string (the io module's "missing category") all work.

    Layout and why it matters for streams
    -------------------------------------
    * ``max_categories=None`` (expanding): a new category gets a new output
      column **appended at the end** of the output. Existing columns never
      move, so earlier rows keep their meaning, but the output gets wider
      over time.
    * ``max_categories=k`` (fixed width): every input column owns exactly
      ``k`` output columns from the first chunk on. The first ``k - 1``
      categories seen get their own column; any further category shares the
      last column, ``"<feature>=__other__"``. The width never changes, which
      is what a buffered model needs when chunks are concatenated.

    Parameters
    ----------
    handle_unknown : {'ignore', 'error'}, default 'ignore'
        What ``transform`` does with a category not seen by ``partial_fit``:
        an all-zero row block ('ignore') or ``ValueError`` ('error'). With
        ``max_categories`` set, unknown categories always go to the
        ``__other__`` column and this option is not used.
    max_categories : int >= 2 or None
    dtype : numpy dtype, default float64

    Attributes
    ----------
    categories_ : list of list of str
        Categories per input column, in order of discovery.
    n_features_out_ : int
    """

    def __init__(self, handle_unknown: str = "ignore", max_categories: Optional[int] = None,
                 dtype=np.float64):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(f"OneHotEncoder: handle_unknown must be 'ignore' or 'error', got {handle_unknown!r}")
        if max_categories is not None:
            check_positive_int(max_categories, "max_categories", "OneHotEncoder", minimum=2)
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self.dtype = dtype
        self._reset()

    def _reset(self) -> None:
        self.categories_ = None
        self._slots = None          # per column: dict category -> output column
        self._names = []            # output column names, in output order

    @staticmethod
    def _as_text(X) -> np.ndarray:
        X = check_2d(X, "OneHotEncoder")
        return X.astype(str)

    def _init_columns(self, n_features: int) -> None:
        self.categories_ = [[] for _ in range(n_features)]
        self._slots = [{} for _ in range(n_features)]
        if self.max_categories is not None:
            # reserve k slots per feature up front: width is fixed forever
            self._names = []
            for j in range(n_features):
                self._names += [f"x{j}=<slot{s}>" for s in range(self.max_categories - 1)]
                self._names.append(f"x{j}={_OTHER}")

    def partial_fit(self, X, y=None) -> "OneHotEncoder":
        """Discover the categories present in one chunk."""
        X = self._as_text(X)
        if X.shape[0] == 0:
            return self
        check_n_features(self, X, reset=False)
        if self.categories_ is None:
            self._init_columns(X.shape[1])
        k = self.max_categories
        for j in range(X.shape[1]):                     # loop over columns only
            new = [c for c in np.unique(X[:, j]).tolist() if c not in self._slots[j]]
            for cat in new:
                if k is None:
                    self._slots[j][cat] = len(self._names)
                    self._names.append(f"x{j}={cat}")
                    self.categories_[j].append(cat)
                elif len(self.categories_[j]) < k - 1:
                    pos = j * k + len(self.categories_[j])
                    self._slots[j][cat] = pos
                    self._names[pos] = f"x{j}={cat}"
                    self.categories_[j].append(cat)
                # else: overflow -> handled as "other" in transform
        return self

    @property
    def n_features_out_(self) -> int:
        check_fitted(self, "n_features_in_", "reading n_features_out_")
        return len(self._names)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, ``"<feature>=<category>"``."""
        names = np.array(self._names, dtype=object)
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError("input_features must have one name per input column")
            names = np.array([f"{input_features[int(n[1:n.index('=')])]}{n[n.index('='):]}" for n in self._names],
                             dtype=object)
        return names

    def transform(self, X) -> np.ndarray:
        """Encode ``X`` with the categories discovered so far.

        Returns
        -------
        ndarray, shape (n_samples, n_features_out_)
        """
        check_fitted(self, "n_features_in_")
        X = self._as_text(X)
        check_n_features(self, X, reset=False)
        n = X.shape[0]
        out = np.zeros((n, len(self._names)), dtype=self.dtype)
        rows = np.arange(n)
        for j in range(X.shape[1]):
            cats = np.array(list(self._slots[j].keys()), dtype=str)
            slots = np.array(list(self._slots[j].values()), dtype=np.int64)
            if cats.size:
                order = np.argsort(cats)
                cats_sorted, slots_sorted = cats[order], slots[order]
                pos = np.searchsorted(cats_sorted, X[:, j])
                pos_c = np.clip(pos, 0, cats_sorted.size - 1)
                known = cats_sorted[pos_c] == X[:, j]
                out[rows[known], slots_sorted[pos_c[known]]] = 1
            else:
                known = np.zeros(n, dtype=bool)
            if (~known).any():
                if self.max_categories is not None:
                    out[rows[~known], (j + 1) * self.max_categories - 1] = 1
                elif self.handle_unknown == "error":
                    bad = X[np.flatnonzero(~known)[0], j]
                    raise ValueError(f"OneHotEncoder: unknown category {bad!r} in column {j}")
        return out
