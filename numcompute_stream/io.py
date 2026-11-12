"""
io.py -- Loading CSV data and turning it into a stream of chunks.

Builds on the group package's :mod:`numcompute.io` (``read_csv`` and
``read_csv_chunks``, which handle missing values, ragged rows and bad cells)
and adds what a streaming experiment needs.

Provides:
    read_csv, read_csv_chunks
        -> re-exported from numcompute.io unchanged
    load_table(path, target, categorical=None, drop=None)
        -> X, y, feature_names from a CSV with mixed numeric / text columns
    iter_chunks(X, y, chunk_size, shuffle=False, random_state=None)
        -> generator of (X_chunk, y_chunk) slices of in-memory arrays
    stream_csv(path, chunk_size, target, categorical=None, drop=None)
        -> generator of (X_chunk, y_chunk) read lazily from disk, so a file
           larger than memory can be streamed

Mixed tables
------------
NumPy arrays are homogeneous, so ``numcompute.io`` reads numeric and text
columns in separate calls. ``load_table`` and ``stream_csv`` do that for
you and join the parts into one ``object`` array (numbers as ``float``,
categories as ``str``), in the file's column order. Numeric transformers
in this package convert their columns to ``float64``; ``OneHotEncoder``
reads its columns as strings. With no categorical columns, ``X`` is a
plain ``float64`` array.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from numcompute.io import read_csv, read_csv_chunks

from ._validation import check_positive_int

__all__ = ["read_csv", "read_csv_chunks", "load_table", "iter_chunks", "stream_csv"]


def _plan_columns(path, target: str, categorical: Optional[Sequence[str]], drop: Optional[Sequence[str]]):
    """Work out which columns are numeric features, text features and the target."""
    with open(path, encoding="utf-8-sig") as fh:
        header = [c.strip() for c in fh.readline().rstrip("\r\n").split(",")]
    if target not in header:
        raise ValueError(f"target column {target!r} not found; columns are {header}")
    categorical = list(categorical or [])
    drop = list(drop or [])
    for name in categorical + drop:
        if name not in header:
            raise ValueError(f"column {name!r} not found; columns are {header}")
    features = [c for c in header if c != target and c not in drop]
    numeric = [c for c in features if c not in categorical]
    text = [c for c in features if c in categorical]
    return features, numeric, text


def _join(features: List[str], numeric: List[str], text: List[str], X_num, X_txt) -> np.ndarray:
    """Interleave numeric and text parts back into the file's column order."""
    if not text:
        return X_num
    n = X_num.shape[0] if X_num is not None else X_txt.shape[0]
    out = np.empty((n, len(features)), dtype=object)
    for j, name in enumerate(features):
        if name in numeric:
            out[:, j] = X_num[:, numeric.index(name)]
        else:
            out[:, j] = X_txt[:, text.index(name)]
    return out


def load_table(path, target: str, categorical: Optional[Sequence[str]] = None,
               drop: Optional[Sequence[str]] = None, target_dtype=float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read a whole CSV into ``X``, ``y`` and the feature names.

    Parameters
    ----------
    path : str
    target : str
        Name of the label column.
    categorical : list of str, optional
        Columns to read as text (missing cells become ``''``).
    drop : list of str, optional
        Columns to ignore.
    target_dtype : float or str, default float
        How to read the target column.

    Returns
    -------
    X : ndarray, shape (n_rows, n_features)
        float64 if every feature is numeric, otherwise object.
    y : ndarray, shape (n_rows,)
    feature_names : list of str

    Raises
    ------
    ValueError
        For unknown column names, plus everything ``numcompute.io.read_csv``
        raises (malformed rows, non-numeric cells in numeric columns, ...).
    """
    features, numeric, text = _plan_columns(path, target, categorical, drop)
    X_num = read_csv(path, usecols=numeric)[0] if numeric else None
    X_txt = read_csv(path, usecols=text, dtype=str)[0] if text else None
    y = read_csv(path, usecols=[target], dtype=target_dtype)[0][:, 0]
    return _join(features, numeric, text, X_num, X_txt), y, features


def iter_chunks(X, y, chunk_size: int, shuffle: bool = False,
                random_state: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(X_chunk, y_chunk)`` of at most ``chunk_size`` rows.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    chunk_size : int
        Rows per chunk; the last chunk may be smaller.
    shuffle : bool, default False
        Visit rows in a random order (fixed by ``random_state``). Useful when
        a file is sorted, e.g. by the target.
    random_state : int, optional

    Raises
    ------
    ValueError
        If X and y differ in length.
    """
    check_positive_int(chunk_size, "chunk_size", "iter_chunks")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"iter_chunks: X has {X.shape[0]} rows but y has {y.shape[0]}")
    order = np.random.default_rng(random_state).permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    for start in range(0, X.shape[0], chunk_size):
        idx = order[start:start + chunk_size]
        yield X[idx], y[idx]


def stream_csv(path, chunk_size: int, target: str, categorical: Optional[Sequence[str]] = None,
               drop: Optional[Sequence[str]] = None,
               target_dtype=float) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Read a CSV lazily and yield ``(X_chunk, y_chunk)`` pairs.

    Only ``chunk_size`` rows are in memory at a time: the numeric columns,
    the text columns and the target are each read with
    ``numcompute.io.read_csv_chunks`` and the three generators advance in
    lock-step.
    """
    check_positive_int(chunk_size, "chunk_size", "stream_csv")
    features, numeric, text = _plan_columns(path, target, categorical, drop)
    num_it = read_csv_chunks(path, chunk_size, usecols=numeric) if numeric else None
    txt_it = read_csv_chunks(path, chunk_size, usecols=text, dtype=str) if text else None
    y_it = read_csv_chunks(path, chunk_size, usecols=[target], dtype=target_dtype)
    for y_chunk, _ in y_it:
        X_num = next(num_it)[0] if num_it is not None else None
        X_txt = next(txt_it)[0] if txt_it is not None else None
        yield _join(features, numeric, text, X_num, X_txt), y_chunk[:, 0]
