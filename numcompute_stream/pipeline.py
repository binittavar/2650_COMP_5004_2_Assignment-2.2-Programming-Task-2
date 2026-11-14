"""
pipeline.py -- Pipelines that learn incrementally.

Provides:
    Pipeline(steps)
        -> the group package's Pipeline plus partial_fit, predict_proba,
           score and memory_footprint; every step learns chunk by chunk
    ColumnTransformer(transformers, remainder='drop')
        -> applies a different transformer to different columns (e.g.
           impute + scale the numbers, one-hot encode the categories) and
           stacks the results; also supports partial_fit

Incremental flow
----------------
``pipe.partial_fit(X_chunk, y_chunk)`` runs, for each transformer in order::

    step.partial_fit(Xt, y)      # update the step's running state
    Xt = step.transform(Xt)      # transform with the *updated* state

and finally ``model.partial_fit(Xt, y)``. ``predict`` / ``predict_proba``
transform with the state learned so far, so a chunk can be scored before
the pipeline trains on it (prequential, "test-then-train" evaluation).

Note that the tree is retrained on its buffer of *already transformed*
rows: if a scaler keeps moving, rows in one buffer were scaled with
slightly different statistics. Trees only compare a feature with
thresholds, so this matters little; ``StandardScaler(decay=...)`` or no
scaling at all are both reasonable for tree models.
"""

from __future__ import annotations

import numpy as np

from numcompute.pipeline import Pipeline as _BatchPipeline

from ._validation import check_2d

__all__ = ["Pipeline", "ColumnTransformer"]


def _call_partial_fit(step, name: str, X, y):
    method = getattr(step, "partial_fit", None)
    if not callable(method):
        raise TypeError(
            f"Pipeline.partial_fit: step {name!r} ({type(step).__name__}) has no partial_fit, so it "
            "cannot learn from a stream. Use a streaming transformer from numcompute_stream."
        )
    return method(X) if y is None else method(X, y)


class Pipeline(_BatchPipeline):
    """A :class:`numcompute.pipeline.Pipeline` that also learns chunk by chunk.

    Parameters
    ----------
    steps : list of (str, object)
        Transformers followed by an optional final model. For
        :meth:`partial_fit` every step needs ``partial_fit``.

    Examples
    --------
    >>> from numcompute_stream.preprocessing import StandardScaler
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> pipe = Pipeline([("scale", StandardScaler()),
    ...                  ("model", DecisionTreeClassifier(max_depth=3, buffer_size=4))])
    >>> _ = pipe.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    >>> pipe.predict([[2.5]]).tolist()
    [1]
    """

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Pipeline(self.steps[key])
        return super().__getitem__(key)

    def __repr__(self) -> str:
        return "Stream" + super().__repr__()

    def partial_fit(self, X, y=None) -> "Pipeline":
        """Update every step with one chunk, then the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,), optional
            Labels, passed to every step (transformers ignore them).

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If a step has no ``partial_fit``.
        """
        Xt = X
        for name, step in self.steps[:-1]:
            _call_partial_fit(step, name, Xt, y)
            Xt = step.transform(Xt)
        name, last = self.steps[-1]
        _call_partial_fit(last, name, Xt, y)
        return self

    def transform_features(self, X):
        """Apply every step except the last (the model's input for ``X``)."""
        return self._transform_until_last(X)

    def predict_proba(self, X):
        """Class probabilities from the final model."""
        last = self._last("predict_proba", "predict_proba")
        return last.predict_proba(self._transform_until_last(X))

    def score(self, X, y) -> float:
        """Accuracy of :meth:`predict` on ``X, y``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def is_fitted(self) -> bool:
        """True when the final model can predict."""
        return bool(getattr(self.steps[-1][1], "is_fitted", True))

    def memory_footprint(self) -> int:
        """Sum of ``memory_footprint()`` over the steps that report one (bytes)."""
        return int(sum(s.memory_footprint() for _, s in self.steps
                       if callable(getattr(s, "memory_footprint", None))))


class ColumnTransformer:
    """Apply transformers to column subsets and join their outputs side by side.

    Parameters
    ----------
    transformers : list of (name, transformer, columns)
        ``columns`` is a list of column indices or a boolean mask.
    remainder : {'drop', 'passthrough'}, default 'drop'
        What happens to columns no transformer claims. 'passthrough' appends
        them unchanged (they must be numeric).

    Examples
    --------
    >>> from numcompute_stream.preprocessing import Imputer, OneHotEncoder
    >>> X = np.array([[1.0, "a"], [np.nan, "b"]], dtype=object)
    >>> ct = ColumnTransformer([("num", Imputer(), [0]), ("cat", OneHotEncoder(), [1])])
    >>> ct.fit_transform(X).tolist()
    [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    """

    def __init__(self, transformers, remainder: str = "drop"):
        if not isinstance(transformers, (list, tuple)) or not transformers:
            raise TypeError("ColumnTransformer expects a non-empty list of (name, transformer, columns)")
        names = []
        for item in transformers:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                raise TypeError(f"each entry must be (name, transformer, columns), got {item!r}")
            names.append(item[0])
        if len(set(names)) != len(names):
            raise ValueError(f"ColumnTransformer names must be unique, got {names}")
        if remainder not in ("drop", "passthrough"):
            raise ValueError(f"remainder must be 'drop' or 'passthrough', got {remainder!r}")
        self.transformers = list(transformers)
        self.remainder = remainder

    def __repr__(self) -> str:
        inner = ", ".join(f"({n!r}, {t!r}, {c!r})" for n, t, c in self.transformers)
        return f"ColumnTransformer([{inner}], remainder={self.remainder!r})"

    def _indices(self, cols, n_features: int) -> np.ndarray:
        cols = np.asarray(cols)
        if cols.dtype == bool:
            if cols.size != n_features:
                raise ValueError(f"boolean column mask has {cols.size} entries, X has {n_features} columns")
            return np.flatnonzero(cols)
        idx = cols.astype(np.int64)
        if idx.size and (idx.min() < -n_features or idx.max() >= n_features):
            raise ValueError(f"column index out of range for X with {n_features} columns: {cols.tolist()}")
        return idx % n_features

    def _rest(self, n_features: int) -> np.ndarray:
        used = np.zeros(n_features, dtype=bool)
        for _, _, cols in self.transformers:
            used[self._indices(cols, n_features)] = True
        return np.flatnonzero(~used)

    def _check(self, X) -> np.ndarray:
        X = check_2d(X, "ColumnTransformer")
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} columns, ColumnTransformer was fitted with {self.n_features_in_}")
        self.n_features_in_ = X.shape[1]
        return X

    def partial_fit(self, X, y=None) -> "ColumnTransformer":
        """Update every transformer with its columns of the chunk."""
        X = self._check(X)
        for name, tr, cols in self.transformers:
            _call_partial_fit(tr, name, X[:, self._indices(cols, X.shape[1])], None)
        return self

    def fit(self, X, y=None) -> "ColumnTransformer":
        """Fit every transformer from scratch on its columns."""
        X = check_2d(X, "ColumnTransformer")
        self.n_features_in_ = X.shape[1]
        for _, tr, cols in self.transformers:
            tr.fit(X[:, self._indices(cols, X.shape[1])])
        return self

    def transform(self, X) -> np.ndarray:
        """Transform each column group and ``np.hstack`` the results (float64)."""
        X = self._check(X)
        parts = [np.asarray(tr.transform(X[:, self._indices(cols, X.shape[1])]), dtype=np.float64)
                 for _, tr, cols in self.transformers]
        if self.remainder == "passthrough":
            rest = self._rest(X.shape[1])
            if rest.size:
                parts.append(X[:, rest].astype(np.float64))
        return np.hstack(parts)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
