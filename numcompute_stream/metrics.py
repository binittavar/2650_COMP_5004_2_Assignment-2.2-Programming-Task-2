"""
metrics.py -- Classification metrics that accumulate over a stream.

Provides:
    StreamingMetric                     base class: update / reset / result
    ConfusionMatrix(labels=None, window=None)
    Accuracy(window=None), ErrorRate(window=None)
    Precision / Recall / F1(average='binary'|'macro'|'micro', pos_label=1, window=None)
    StreamingAUC(n_bins=1000, pos_label=1, window=None)
    MetricCollection({name: metric})    update several metrics in one call

The protocol
------------
Every metric follows the same three-method protocol::

    m = Accuracy()
    for y_true_chunk, y_pred_chunk in stream:
        m.update(y_true_chunk, y_pred_chunk)   # O(chunk) work, O(1) or O(k^2) state
        print(m.result())                      # value over everything so far
    m.reset()                                  # start again

``history_`` keeps the value of ``result()`` after every ``update``, so a
metric can be plotted over time without extra bookkeeping.

Cumulative vs rolling
---------------------
* ``window=None`` (default): the metric covers **every** chunk seen since
  the last ``reset``. Confusion-matrix metrics store only a ``k x k`` count
  matrix (``k`` = number of labels), AUC only two ``n_bins`` histograms.
* ``window=w``: the metric covers the **last w chunks** only, which shows
  how a model is doing *now* rather than on average. ``window=1`` gives the
  per-chunk value.

Edge cases
----------
* Labels may appear for the first time in any chunk; the confusion matrix
  grows and old counts are kept in place.
* Precision / recall / F1 of a class with no predictions or no samples is
  0.0, never NaN (the same rule as the group package's ``metrics``).
* ``result()`` before any data is NaN (AUC: also NaN while only one class
  has been seen), so plots of the first chunks do not crash.
* Empty chunks are accepted and change nothing (they are still recorded in
  ``history_`` so the history lines up with the chunk index).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

from numcompute.metrics import confusion_matrix as _batch_confusion_matrix

from ._validation import check_labels, check_positive_int

__all__ = [
    "StreamingMetric",
    "ConfusionMatrix",
    "Accuracy",
    "ErrorRate",
    "Precision",
    "Recall",
    "F1",
    "StreamingAUC",
    "MetricCollection",
]


def _check_pair(y_true, y_other, owner: str):
    y_true = check_labels(y_true, owner)
    y_other = np.asarray(y_other)
    if y_other.ndim != 1:
        raise ValueError(f"{owner}: predictions must be 1-D, got shape {y_other.shape}")
    if y_true.shape != y_other.shape:
        raise ValueError(f"{owner}: y_true has shape {y_true.shape} but predictions have shape {y_other.shape}")
    return y_true, y_other


class StreamingMetric:
    """Base class for metrics with ``update`` / ``reset`` / ``result``.

    Parameters
    ----------
    window : int or None
        None for a cumulative metric; an int ``w`` for a rolling metric over
        the last ``w`` chunks.
    """

    name = "metric"

    def __init__(self, window: Optional[int] = None):
        self.window = None if window is None else check_positive_int(window, "window", type(self).__name__)
        self.reset()

    def reset(self) -> "StreamingMetric":
        """Forget all accumulated data and the history."""
        self.history_ = []
        self.n_updates_ = 0
        self._reset_state()
        return self

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingMetric":
        """Fold one chunk into the metric and record the new result."""
        self._update_state(y_true_chunk, y_pred_chunk)
        self.n_updates_ += 1
        self.history_.append(self.result())
        return self

    def result(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _reset_state(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _update_state(self, y_true, y_pred) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(window={self.window})"


# ----------------------------------------------------------------------------
# Confusion matrix and everything derived from it
# ----------------------------------------------------------------------------
class ConfusionMatrix(StreamingMetric):
    """Confusion matrix accumulated over a stream.

    Parameters
    ----------
    labels : array-like, optional
        Labels known in advance (more may still appear).
    window : int or None

    ``result()`` returns ``(matrix, labels)`` where ``matrix[i, j]`` counts
    rows with true label ``labels[i]`` predicted as ``labels[j]``.
    """

    name = "confusion_matrix"

    def __init__(self, labels=None, window: Optional[int] = None):
        self._init_labels = None if labels is None else np.unique(np.asarray(labels))
        super().__init__(window)

    def _reset_state(self) -> None:
        self._labels = self._init_labels.copy() if self._init_labels is not None else None
        n = 0 if self._labels is None else len(self._labels)
        self._cm = np.zeros((n, n), dtype=np.int64)
        self._recent = deque()

    def _grow(self, new_values: np.ndarray) -> None:
        """Add unseen labels, moving the old counts to their new positions."""
        if self._labels is None:
            self._labels = np.unique(new_values)
            self._cm = np.zeros((len(self._labels), len(self._labels)), dtype=np.int64)
            return
        merged = np.union1d(self._labels, new_values)
        if merged.size == self._labels.size:
            return
        pos = np.searchsorted(merged, self._labels)
        cm = np.zeros((merged.size, merged.size), dtype=np.int64)
        cm[np.ix_(pos, pos)] = self._cm
        self._labels, self._cm = merged, cm

    def _update_state(self, y_true, y_pred) -> None:
        y_true, y_pred = _check_pair(y_true, y_pred, f"{type(self).__name__}.update")
        if y_true.size == 0:
            if self.window is not None:
                self._recent.append((y_true, y_pred))
                self._trim()
            return
        self._grow(np.concatenate([y_true, y_pred]))
        if self.window is None:
            cm, _ = _batch_confusion_matrix(y_true, y_pred, labels=self._labels)
            self._cm += cm
        else:
            self._recent.append((y_true, y_pred))
            self._trim()

    def _trim(self) -> None:
        while len(self._recent) > self.window:
            self._recent.popleft()

    def matrix(self):
        """``(matrix, labels)`` for the cumulative data or the current window."""
        if self.window is None or self._labels is None:
            labels = self._labels if self._labels is not None else np.array([])
            return self._cm.copy(), labels
        yt = np.concatenate([a for a, _ in self._recent]) if self._recent else np.array([])
        yp = np.concatenate([b for _, b in self._recent]) if self._recent else np.array([])
        if yt.size == 0:
            return np.zeros((len(self._labels),) * 2, dtype=np.int64), self._labels
        cm, _ = _batch_confusion_matrix(yt, yp, labels=self._labels)
        return cm, self._labels

    def result(self):
        return self.matrix()


class Accuracy(ConfusionMatrix):
    """Fraction of correct predictions (trace / total of the confusion matrix)."""

    name = "accuracy"

    def __init__(self, window: Optional[int] = None):
        super().__init__(labels=None, window=window)

    def result(self) -> float:
        cm, _ = self.matrix()
        total = cm.sum()
        return float(np.trace(cm) / total) if total else float("nan")


class ErrorRate(Accuracy):
    """``1 - accuracy``."""

    name = "error_rate"

    def result(self) -> float:
        return 1.0 - super().result()


_AVERAGES = ("binary", "macro", "micro")


class _RatioMetric(ConfusionMatrix):
    """Precision / recall / F1 computed from the accumulated confusion matrix."""

    def __init__(self, average: str = "binary", pos_label=1, window: Optional[int] = None):
        if average not in _AVERAGES:
            raise ValueError(f"{type(self).__name__}: average must be one of {_AVERAGES}, got {average!r}")
        self.average = average
        self.pos_label = pos_label
        super().__init__(labels=None, window=window)

    @staticmethod
    def _ratio(num, den):
        num = np.asarray(num, dtype=np.float64)
        den = np.asarray(den, dtype=np.float64)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def _per_class(self, cm):  # pragma: no cover - overridden
        raise NotImplementedError

    def result(self) -> float:
        cm, labels = self.matrix()
        if cm.sum() == 0:
            return float("nan")
        if self.average == "binary":
            if len(labels) > 2:
                raise ValueError(f"{type(self).__name__}(average='binary') needs at most 2 labels, "
                                 f"got {labels.tolist()}; use average='macro' or 'micro'")
            hit = np.flatnonzero(labels == self.pos_label)
            if hit.size == 0:           # positive class never seen yet
                return 0.0
            return float(self._per_class(cm)[hit[0]])
        if self.average == "macro":
            return float(np.mean(self._per_class(cm)))
        return float(self._micro(cm))

    def _micro(self, cm):
        return self._ratio(np.trace(cm), cm.sum())     # single-label: P = R = F1 = accuracy


class Precision(_RatioMetric):
    """TP / (TP + FP) per class, averaged as requested."""

    name = "precision"

    def _per_class(self, cm):
        return self._ratio(np.diag(cm), cm.sum(axis=0))


class Recall(_RatioMetric):
    """TP / (TP + FN) per class, averaged as requested."""

    name = "recall"

    def _per_class(self, cm):
        return self._ratio(np.diag(cm), cm.sum(axis=1))


class F1(_RatioMetric):
    """Harmonic mean of precision and recall, per class."""

    name = "f1"

    def _per_class(self, cm):
        p = self._ratio(np.diag(cm), cm.sum(axis=0))
        r = self._ratio(np.diag(cm), cm.sum(axis=1))
        return self._ratio(2 * p * r, p + r)


# ----------------------------------------------------------------------------
# AUC from score histograms
# ----------------------------------------------------------------------------
class StreamingAUC(StreamingMetric):
    """Area under the ROC curve, accumulated with fixed-size score histograms.

    Scores in [0, 1] (e.g. ``predict_proba(X)[:, 1]``) are counted into
    ``n_bins`` equal bins, separately for positive and negative rows. The
    AUC is the probability that a random positive outscores a random
    negative, counting rows in the same bin as half a win::

        AUC = sum_b neg_b * (pos_above_b + 0.5 * pos_b) / (P * N)

    Memory is O(n_bins) however long the stream, and the result is exact
    whenever no two different scores share a bin (with tree probabilities,
    which take few distinct values, this is almost always the case).

    Parameters
    ----------
    n_bins : int, default 1000
    pos_label : label, default 1
        The positive class; every other label counts as negative.
    window : int or None
    """

    name = "auc"

    def __init__(self, n_bins: int = 1000, pos_label=1, window: Optional[int] = None):
        self.n_bins = check_positive_int(n_bins, "n_bins", "StreamingAUC", minimum=2)
        self.pos_label = pos_label
        super().__init__(window)

    def _reset_state(self) -> None:
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._recent = deque()

    def _update_state(self, y_true, y_score) -> None:
        y_true, y_score = _check_pair(y_true, y_score, "StreamingAUC.update")
        s = np.asarray(y_score, dtype=np.float64)
        if np.isnan(s).any() or (s < 0).any() or (s > 1).any():
            raise ValueError("StreamingAUC.update: scores must be probabilities in [0, 1] without NaN")
        b = np.minimum((s * self.n_bins).astype(np.int64), self.n_bins - 1)
        is_pos = y_true == self.pos_label
        pos = np.bincount(b[is_pos], minlength=self.n_bins)
        neg = np.bincount(b[~is_pos], minlength=self.n_bins)
        self._pos += pos
        self._neg += neg
        if self.window is not None:
            self._recent.append((pos, neg))
            if len(self._recent) > self.window:
                old_p, old_n = self._recent.popleft()
                self._pos -= old_p
                self._neg -= old_n

    def result(self) -> float:
        P, N = self._pos.sum(), self._neg.sum()
        if P == 0 or N == 0:
            return float("nan")
        pos_above = np.cumsum(self._pos[::-1])[::-1] - self._pos     # positives in higher bins
        return float((self._neg * (pos_above + 0.5 * self._pos)).sum() / (P * N))

    def roc_curve(self):
        """``(fpr, tpr)`` arrays from the accumulated histograms (for plotting)."""
        P, N = self._pos.sum(), self._neg.sum()
        if P == 0 or N == 0:
            raise ValueError("StreamingAUC.roc_curve needs at least one positive and one negative row")
        tp = np.concatenate([[0], np.cumsum(self._pos[::-1])])
        fp = np.concatenate([[0], np.cumsum(self._neg[::-1])])
        return fp / N, tp / P


# ----------------------------------------------------------------------------
# Several metrics at once
# ----------------------------------------------------------------------------
class MetricCollection:
    """Update a named group of metrics with one call.

    Score-based metrics (:class:`StreamingAUC`) receive ``y_score``; label
    metrics receive ``y_pred``. If ``y_score`` is not given, score metrics
    are skipped for that chunk.

    Examples
    --------
    >>> mc = MetricCollection({"acc": Accuracy(), "f1": F1()})
    >>> _ = mc.update([0, 1, 1], [0, 1, 0])
    >>> round(mc.result()["acc"], 3)
    0.667
    """

    def __init__(self, metrics: Dict[str, StreamingMetric]):
        if not isinstance(metrics, dict) or not metrics:
            raise TypeError("MetricCollection expects a non-empty dict {name: metric}")
        for name, m in metrics.items():
            if not isinstance(m, StreamingMetric):
                raise TypeError(f"metric {name!r} must be a StreamingMetric, got {type(m).__name__}")
        self.metrics = dict(metrics)

    def update(self, y_true, y_pred, y_score=None) -> "MetricCollection":
        for m in self.metrics.values():
            if isinstance(m, StreamingAUC):
                if y_score is not None:
                    m.update(y_true, y_score)
            else:
                m.update(y_true, y_pred)
        return self

    def result(self) -> dict:
        return {name: m.result() for name, m in self.metrics.items()}

    def reset(self) -> "MetricCollection":
        for m in self.metrics.values():
            m.reset()
        return self

    def __getitem__(self, name: str) -> StreamingMetric:
        return self.metrics[name]
