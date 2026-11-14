"""
stream.py -- Driving a model or pipeline through a stream, with logging.

Provides:
    StreamTrainer(model, metrics=None, positive_label=1, rolling_window=5, ...)
        -> fit_chunk(X, y): one incremental training step
        -> score_chunk(X, y): predict a chunk and update the metrics
        -> step(X, y): prequential "test-then-train" on one chunk + log record
        -> run(stream): step through an iterable of (X_chunk, y_chunk)
        -> history(key), log_, to_csv(path), summary()

Prequential evaluation
----------------------
Each chunk is first used to **test** the current model (it has never seen
these rows) and only then to **train** it. Every prediction is therefore an
honest out-of-sample prediction, and no data has to be held back for a test
set. Chunks that arrive before the tree's first retrain cannot be scored;
their metric entries are NaN and they show as gaps in plots.

What is logged per chunk (``log_``, one dict per chunk)
-------------------------------------------------------
``chunk, n_rows, samples_seen, scored, chunk_accuracy, cumulative_accuracy,
rolling_accuracy, cumulative_<metric>..., fit_seconds, predict_seconds,
retrained, n_retrains, buffer_fill, memory_bytes`` (plus
``peak_memory_bytes`` when ``track_peak_memory=True``).

``memory_bytes`` is the model's own footprint (tree arrays + buffered rows,
from ``memory_footprint()``) - deterministic and cheap. The optional
``peak_memory_bytes`` uses :mod:`tracemalloc` to measure the peak Python
allocation during the chunk, which is more complete but slows training.
"""

from __future__ import annotations

import csv
import logging
import time
import tracemalloc
from typing import Dict, Iterable, Optional

import numpy as np

from .metrics import Accuracy, MetricCollection, StreamingAUC, StreamingMetric

__all__ = ["StreamTrainer"]

logger = logging.getLogger("numcompute_stream")


def _final_model(model):
    """The object that does the learning (last step of a pipeline, else the model)."""
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


class StreamTrainer:
    """Manage a model or pipeline, its metrics and a per-chunk log.

    Parameters
    ----------
    model : estimator or Pipeline
        Must have ``partial_fit(X, y)`` and ``predict(X)``;
        ``predict_proba`` is used for AUC when available.
    metrics : dict of {name: StreamingMetric}, optional
        Extra cumulative metrics, e.g. ``{"f1": F1(), "auc": StreamingAUC()}``.
        Accuracy (cumulative, rolling and per chunk) is always tracked.
    positive_label : label, default 1
        Class whose probability is passed to AUC metrics.
    rolling_window : int, default 5
        Window, in chunks, of ``rolling_accuracy``.
    log_every : int, default 1
        Emit a ``logging.INFO`` line every ``log_every`` chunks (the
        ``log_`` list always gets every chunk).
    verbose : bool, default False
        Also ``print`` the log line.
    track_peak_memory : bool, default False
        Measure peak allocations with ``tracemalloc``.
    name : str, optional
        Label used in log lines and by :mod:`visualise`.
    """

    def __init__(self, model, metrics: Optional[Dict[str, StreamingMetric]] = None, positive_label=1,
                 rolling_window: int = 5, log_every: int = 1, verbose: bool = False,
                 track_peak_memory: bool = False, name: Optional[str] = None):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"StreamTrainer needs a model with {method}(), got {type(model).__name__}")
        self.model = model
        self.positive_label = positive_label
        self.log_every = log_every
        self.verbose = verbose
        self.track_peak_memory = track_peak_memory
        self.name = name or type(_final_model(model)).__name__
        base = {
            "cumulative_accuracy": Accuracy(),
            "rolling_accuracy": Accuracy(window=rolling_window),
            "chunk_accuracy": Accuracy(window=1),
        }
        extra = {f"cumulative_{k}": m for k, m in (metrics or {}).items()}
        self.metrics = MetricCollection({**base, **extra})
        self.log_ = []
        self.last_chunk_ = None
        self._samples_seen = 0

    # -- readiness -----------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True once the model can make predictions."""
        return bool(getattr(_final_model(self.model), "is_fitted", True))

    def _n_retrains(self) -> int:
        return int(getattr(_final_model(self.model), "n_retrains_", 0))

    def _memory(self) -> int:
        fn = getattr(self.model, "memory_footprint", None) or getattr(_final_model(self.model),
                                                                       "memory_footprint", None)
        return int(fn()) if callable(fn) else 0

    # -- the two primitives ---------------------------------------------------
    def fit_chunk(self, X, y) -> dict:
        """Train on one chunk with ``partial_fit``.

        Returns
        -------
        dict
            ``fit_seconds``, ``retrained`` (bool), ``n_retrains`` and, if
            enabled, ``peak_memory_bytes``.
        """
        before = self._n_retrains()
        if self.track_peak_memory:
            tracemalloc.start()
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        elapsed = time.perf_counter() - start
        info = {"fit_seconds": elapsed}
        if self.track_peak_memory:
            info["peak_memory_bytes"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        after = self._n_retrains()
        info.update(retrained=after > before, n_retrains=after)
        self._samples_seen += len(y)
        return info

    def score_chunk(self, X, y) -> dict:
        """Predict one chunk and fold it into every metric.

        Returns
        -------
        dict
            Metric values after this chunk and ``predict_seconds``; if the
            model cannot predict yet, ``{"scored": False}`` and the metrics
            are left untouched.
        """
        y = np.asarray(y)
        if not self.is_ready or y.size == 0:
            return {"scored": False}
        start = time.perf_counter()
        y_pred = self.model.predict(X)
        y_score = None
        wants_score = any(isinstance(m, StreamingAUC) for m in self.metrics.metrics.values())
        if wants_score and callable(getattr(self.model, "predict_proba", None)):
            proba = self.model.predict_proba(X)
            classes = getattr(_final_model(self.model), "classes_", None)
            if classes is not None and self.positive_label in classes:
                y_score = proba[:, int(np.flatnonzero(classes == self.positive_label)[0])]
            else:
                y_score = np.zeros(y.size)
        elapsed = time.perf_counter() - start
        self.metrics.update(y, y_pred, y_score)
        self.last_chunk_ = {"X": X, "y_true": y, "y_pred": y_pred}
        out = {"scored": True, "predict_seconds": elapsed}
        out.update(self.metrics.result())
        return out

    # -- prequential loop -----------------------------------------------------
    def step(self, X, y) -> dict:
        """Test on the chunk, then train on it; append and return a log record."""
        y = np.asarray(y)
        record = {"chunk": len(self.log_), "n_rows": int(y.size)}
        scored = self.score_chunk(X, y)
        fitted = self.fit_chunk(X, y)
        record["samples_seen"] = self._samples_seen
        record["scored"] = scored.pop("scored")
        for name in self.metrics.metrics:
            record[name] = scored.get(name, np.nan)
        record["predict_seconds"] = scored.get("predict_seconds", np.nan)
        record.update(fitted)
        model = _final_model(self.model)
        record["buffer_fill"] = int(getattr(model, "buffer_fill_", 0))
        record["memory_bytes"] = self._memory()
        self.log_.append(record)
        if record["chunk"] % self.log_every == 0:
            line = (f"[{self.name}] chunk {record['chunk']:>3} | seen {record['samples_seen']:>6} | "
                    f"chunk acc {record['chunk_accuracy']:.3f} | cum acc {record['cumulative_accuracy']:.3f} | "
                    f"fit {record['fit_seconds'] * 1e3:7.2f} ms | retrained {str(record['retrained']):5} | "
                    f"buffer {record['buffer_fill']:>5} | mem {record['memory_bytes'] / 1024:8.1f} KiB")
            logger.info(line)
            if self.verbose:
                print(line)
        return record

    def run(self, stream: Iterable, max_chunks: Optional[int] = None, finalize: bool = False) -> "StreamTrainer":
        """Process every ``(X_chunk, y_chunk)`` of ``stream`` with :meth:`step`.

        Parameters
        ----------
        stream : iterable of (X_chunk, y_chunk)
        max_chunks : int, optional
            Stop after this many chunks.
        finalize : bool, default False
            After the last chunk, call the model's ``flush()`` (if any) so a
            partly filled buffer is used for one last retrain.
        """
        for i, (X, y) in enumerate(stream):
            if max_chunks is not None and i >= max_chunks:
                break
            self.step(X, y)
        if finalize:
            flush = getattr(_final_model(self.model), "flush", None)
            if callable(flush):
                flush()
        return self

    # -- reading the log ------------------------------------------------------
    def history(self, key: str) -> np.ndarray:
        """One logged value per chunk as a float array (NaN where missing)."""
        if self.log_ and key not in self.log_[0] and all(key not in r for r in self.log_):
            raise KeyError(f"{key!r} is not logged; available keys: {sorted(self.log_[0])}")
        return np.array([float(r.get(key, np.nan)) for r in self.log_], dtype=np.float64)

    def to_csv(self, path: str) -> None:
        """Write the per-chunk log to a CSV file (plain ``csv`` module)."""
        if not self.log_:
            raise ValueError("nothing to write: the log is empty")
        keys = list(self.log_[0].keys())
        for r in self.log_:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.log_)

    def summary(self) -> dict:
        """Totals for the whole run."""
        if not self.log_:
            return {}
        fit = self.history("fit_seconds")
        return {
            "name": self.name,
            "chunks": len(self.log_),
            "samples_seen": self._samples_seen,
            "final_cumulative_accuracy": float(self.log_[-1]["cumulative_accuracy"]),
            "n_retrains": int(self.log_[-1]["n_retrains"]),
            "total_fit_seconds": float(np.nansum(fit)),
            "max_memory_bytes": int(np.nanmax(self.history("memory_bytes"))),
        }
