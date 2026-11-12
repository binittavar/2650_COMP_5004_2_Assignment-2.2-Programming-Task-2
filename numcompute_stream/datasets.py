"""
datasets.py -- Synthetic classification streams (NumPy only).

Provides:
    make_stream(n_samples=5000, n_features=6, drift_at=None, ...)
        -> X, y for a binary problem whose decision rule can change part-way
           through (concept drift), with optional label noise and missing
           values
    save_csv(path, X, y, feature_names=None, target="label")
        -> write such a stream to CSV so it can be read back with io.py

The concept
-----------
Before ``drift_at`` the label is 1 when ``x0 + x1 > 0`` (plus noise); after
it the rule rotates to ``x0 - x1 > 0``. A tree trained only on old data is
then wrong on about half of the new rows, so the stream shows how quickly a
buffered tree recovers - smaller buffers recover faster, larger buffers
learn a steadier tree from more rows.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._validation import check_positive_int

__all__ = ["make_stream", "save_csv"]


def make_stream(n_samples: int = 5000, n_features: int = 6, drift_at: Optional[int] = None,
                label_noise: float = 0.05, nan_fraction: float = 0.0, random_state: Optional[int] = 0):
    """Generate a binary classification stream with optional concept drift.

    Parameters
    ----------
    n_samples : int
    n_features : int, >= 2
        ``x0`` and ``x1`` carry the signal; the rest are noise.
    drift_at : int or None
        Row index where the rule changes. None: no drift.
    label_noise : float in [0, 0.5)
        Fraction of labels flipped at random.
    nan_fraction : float in [0, 1)
        Fraction of feature cells (outside ``x0``) replaced by NaN.
    random_state : int or None

    Returns
    -------
    X : ndarray of float, shape (n_samples, n_features)
    y : ndarray of int, shape (n_samples,)
    """
    check_positive_int(n_samples, "n_samples", "make_stream")
    check_positive_int(n_features, "n_features", "make_stream", minimum=2)
    if not 0 <= label_noise < 0.5:
        raise ValueError(f"label_noise must be in [0, 0.5), got {label_noise}")
    if not 0 <= nan_fraction < 1:
        raise ValueError(f"nan_fraction must be in [0, 1), got {nan_fraction}")
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    score = X[:, 0] + X[:, 1]
    if drift_at is not None:
        after = np.arange(n_samples) >= drift_at
        score = np.where(after, X[:, 0] - X[:, 1], score)
    y = (score > 0).astype(np.int64)
    flip = rng.random(n_samples) < label_noise
    y[flip] = 1 - y[flip]
    if nan_fraction > 0:
        mask = rng.random((n_samples, n_features)) < nan_fraction
        mask[:, 0] = False
        X[mask] = np.nan
    return X, y


def save_csv(path: str, X, y, feature_names: Optional[Sequence[str]] = None, target: str = "label") -> None:
    """Write ``X`` and ``y`` to a comma-separated file with a header (NaN as blank)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
    with open(path, "w") as fh:
        fh.write(",".join(names + [target]) + "\n")
        for row, label in zip(X, y):
            cells = ["" if np.isnan(v) else repr(float(v)) for v in row]
            fh.write(",".join(cells + [str(label)]) + "\n")
