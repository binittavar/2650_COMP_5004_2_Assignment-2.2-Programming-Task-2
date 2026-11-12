"""
_validation.py -- Shared input checks for the streaming package.

Every public class in ``numcompute_stream`` validates its inputs through
these helpers, so the error messages and axis conventions are identical
across modules:

* ``X`` is 2-D, shape ``(n_samples, n_features)``; 1-D input is rejected
  with a ``reshape(-1, 1)`` hint so a single feature is never mistaken for
  a single sample.
* ``y`` is 1-D, shape ``(n_samples,)``, with the same length as ``X``.
* A *chunk* may be empty (``n_samples == 0``). Streaming components treat
  an empty chunk as a no-op instead of an error, because real streams do
  deliver empty batches.
* Once a component has seen data, every later chunk must have the same
  number of features (``n_features_in_``).

The group package's :class:`numcompute.preprocessing.NotFittedError` is
re-used so that ``except NotFittedError`` works across both packages.
"""

from __future__ import annotations

import numpy as np

from numcompute.preprocessing import NotFittedError

__all__ = [
    "NotFittedError",
    "check_2d",
    "check_numeric",
    "check_labels",
    "check_X_y",
    "check_n_features",
    "check_positive_int",
    "check_fitted",
]


def check_2d(X, owner: str) -> np.ndarray:
    """Return ``X`` as an ndarray, raising ``ValueError`` if it is not 2-D."""
    X = np.asarray(X)
    if X.ndim != 2:
        hint = " Use X.reshape(-1, 1) for a single feature." if X.ndim == 1 else ""
        raise ValueError(
            f"{owner} expects a 2-D array of shape (n_samples, n_features), "
            f"got a {X.ndim}-D array with shape {X.shape}.{hint}"
        )
    return X


def check_numeric(X, owner: str, *, allow_nan: bool = True) -> np.ndarray:
    """Validate ``X`` as a 2-D float64 array.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    owner : str
        Name used in error messages, e.g. ``"StandardScaler.partial_fit"``.
    allow_nan : bool, default True
        If False, NaN raises ``ValueError``.

    Returns
    -------
    numpy.ndarray of float64, shape (n_samples, n_features)
        A view when no conversion is needed, otherwise a new array.

    Raises
    ------
    TypeError
        If ``X`` holds non-numeric values (e.g. strings).
    ValueError
        If ``X`` is not 2-D, contains ``+-inf``, or contains NaN when
        ``allow_nan`` is False.
    """
    X = check_2d(X, owner)
    if X.dtype.kind in "biuf":
        X = X.astype(np.float64, copy=False)
    elif X.dtype.kind == "O":
        try:
            X = X.astype(np.float64)
        except (TypeError, ValueError):
            raise TypeError(f"{owner} requires numeric input; X contains non-numeric objects") from None
    else:
        raise TypeError(
            f"{owner} requires numeric input, got dtype {X.dtype}. "
            "Encode categorical columns with OneHotEncoder first."
        )
    if X.size:
        if np.isinf(X).any():
            r, c = np.argwhere(np.isinf(X))[0]
            raise ValueError(f"{owner}: X contains infinity at row {r}, column {c}")
        if not allow_nan and np.isnan(X).any():
            r, c = np.argwhere(np.isnan(X))[0]
            raise ValueError(f"{owner}: X contains NaN at row {r}, column {c}; impute it first")
    return X


def check_labels(y, owner: str) -> np.ndarray:
    """Validate class labels: 1-D and free of NaN / None.

    Raises
    ------
    ValueError
        If ``y`` is not 1-D or contains a missing label.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{owner} expects y as a 1-D array of shape (n_samples,), got shape {y.shape}")
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError(f"{owner}: y contains NaN; class labels must not be missing")
    if y.dtype.kind == "O" and any(v is None or (isinstance(v, float) and v != v) for v in y):
        raise ValueError(f"{owner}: y contains a missing label (None or NaN)")
    return y


def check_X_y(X, y, owner: str, *, allow_nan: bool = True):
    """Validate a numeric ``X`` and labels ``y`` of matching length."""
    X = check_numeric(X, owner, allow_nan=allow_nan)
    y = check_labels(y, owner)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{owner}: X has {X.shape[0]} samples but y has {y.shape[0]}; "
            "they must describe the same rows"
        )
    return X, y


def check_n_features(est, X: np.ndarray, *, reset: bool) -> None:
    """Record (``reset=True``) or verify ``est.n_features_in_``.

    With ``reset=False`` and no recorded value yet, the value is recorded:
    the first chunk of a stream fixes the width for every later chunk.
    """
    n_features = X.shape[1]
    if reset or not hasattr(est, "n_features_in_"):
        if n_features == 0:
            raise ValueError(f"{type(est).__name__} requires at least one feature, got shape {X.shape}")
        est.n_features_in_ = n_features
    elif n_features != est.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {type(est).__name__} was fitted with "
            f"{est.n_features_in_} features"
        )


def check_positive_int(value, name: str, owner: str, *, minimum: int = 1) -> int:
    """Return ``value`` as int, raising if it is not an integer ``>= minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{owner}: {name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{owner}: {name} must be >= {minimum}, got {value}")
    return int(value)


def check_fitted(est, attr: str, action: str = "transform") -> None:
    """Raise :class:`NotFittedError` if ``est`` lacks the attribute ``attr``."""
    if not hasattr(est, attr):
        raise NotFittedError(
            f"This {type(est).__name__} instance has not seen any data yet; "
            f"call fit or partial_fit before {action}."
        )
