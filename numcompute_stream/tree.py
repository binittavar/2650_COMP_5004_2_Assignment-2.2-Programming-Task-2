"""
tree.py -- A depth-limited CART classification tree with buffered streaming.

Provides:
    gini(counts), entropy(counts)
        -> impurity of one or many class-count vectors (vectorised)
    best_split(X, y_enc, n_classes, ...)
        -> the (feature, threshold, gain) that lowers impurity the most,
           searched over every feature and every threshold at once
    DecisionTreeClassifier(max_depth=5, buffer_size=500, ...)
        -> fit(X, y) trains from scratch; partial_fit(X_chunk, y_chunk)
           buffers chunks and retrains once ``buffer_size`` rows are waiting

How the tree is grown (see the course's classification-tree notes)
------------------------------------------------------------------
Every node repeats the same four steps:

1. **try** every question ``x[:, f] <= t`` (all features, all thresholds),
2. **keep** the one with the largest information gain
   ``I(parent) - (n_L/n) I(left) - (n_R/n) I(right)``, where ``I`` is Gini
   (``1 - sum p^2``) or entropy (``-sum p log2 p``),
3. **split** the rows and repeat on each side,
4. **stop** (make a leaf) when ``max_depth`` is reached, the node is pure,
   it has fewer than ``min_samples_split`` rows, or no question improves
   the impurity by more than ``min_impurity_decrease``.

The search in step 1-2 is vectorised: each feature column is sorted once,
the class counts on the left of every possible cut are a cumulative sum of
one-hot labels, and the impurities of *all* cuts of *all* features are
computed in a single array expression. The only Python loop is over nodes.

Numerical rules (stated once, used everywhere)
----------------------------------------------
* **Ties between questions** are broken deterministically: among gains
  within ``1e-12`` of the best, the lowest feature index wins, then the
  lowest threshold. The same data therefore always grows the same tree.
* **Ties in values**: a cut is only placed between two *different* sorted
  values, at their midpoint, so equal values always go to the same side.
  If rounding would push the midpoint onto the upper value, the lower value
  is used instead, which keeps ``x <= t`` correct.
* **NaN in X** means "missing" and always goes to the **right** child
  (``NaN <= t`` is False), at training and prediction time alike. The
  search can also pick the cut "finite values left, missing right" when
  missingness itself is informative.
* **Ties in class votes**: a leaf predicts the class with the most training
  rows; on a tie the class that sorts first wins (``np.argmax``).
* **Zero-variance features** offer no valid cut and are skipped.

Streaming: buffered retraining
------------------------------
``partial_fit`` appends each chunk to an internal buffer. When the buffer
holds at least ``buffer_size`` rows, the tree is rebuilt from scratch on
the buffered rows and the buffer is emptied. Between retrains ``predict``
uses the most recently trained tree. ``flush()`` forces a retrain on a
partly filled buffer (useful at the end of a stream).
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ._validation import NotFittedError, check_n_features, check_numeric, check_positive_int, check_X_y

__all__ = ["gini", "entropy", "best_split", "DecisionTreeClassifier"]

_TIE_TOL = 1e-12          # gains this close are treated as equal
_LEAF = -1                # feature index stored in leaf nodes


# ----------------------------------------------------------------------------
# Impurity measures (work on any leading shape: (..., n_classes))
# ----------------------------------------------------------------------------
def _proportions(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)


def gini(counts) -> np.ndarray:
    """Gini impurity ``1 - sum(p**2)`` of class-count vectors.

    Parameters
    ----------
    counts : array-like, shape (..., n_classes)
        Non-negative class counts. An all-zero vector has impurity 0.

    Returns
    -------
    ndarray, shape (...)
        0 for a pure node; ``1 - 1/k`` for ``k`` equally frequent classes.
    """
    p = _proportions(counts)
    has_rows = p.sum(axis=-1) > 0                 # an empty node is pure, not maximally mixed
    return np.where(has_rows, 1.0 - (p * p).sum(axis=-1), 0.0)


def entropy(counts) -> np.ndarray:
    """Entropy ``-sum(p * log2 p)`` in bits, with ``0 * log 0 = 0``."""
    p = _proportions(counts)
    logp = np.log2(np.where(p > 0, p, 1.0))     # log2(1) = 0 for empty classes
    return -(p * logp).sum(axis=-1)


_CRITERIA = {"gini": gini, "entropy": entropy}


# ----------------------------------------------------------------------------
# Vectorised split search
# ----------------------------------------------------------------------------
def best_split(X, y_enc, n_classes: int, *, criterion: str = "gini", min_samples_leaf: int = 1,
               features: Optional[np.ndarray] = None):
    """Find the best binary question ``X[:, f] <= t`` for one node.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Node data (NaN allowed: it goes right).
    y_enc : ndarray of int, shape (n_samples,)
        Labels encoded as ``0 .. n_classes - 1``.
    n_classes : int
    criterion : {'gini', 'entropy'}
    min_samples_leaf : int
        Cuts leaving fewer rows on either side are not considered.
    features : ndarray of int or None
        Candidate feature indices (sorted). None means all features.

    Returns
    -------
    dict or None
        ``{'feature', 'threshold', 'gain', 'n_left'}`` for the best cut, or
        None if no valid cut exists (e.g. every candidate feature is
        constant).

    Complexity
    ----------
    O(n log n * f) for the sorts plus O(n * f * k) for the counts, where
    ``f`` is the number of candidate features and ``k`` the number of
    classes. No Python loop over samples or thresholds.
    """
    impurity = _CRITERIA[criterion]
    n = X.shape[0]
    if n < 2:
        return None
    feats = np.arange(X.shape[1]) if features is None else np.asarray(features)
    Xf = X[:, feats]                                            # (n, f)
    order = np.argsort(Xf, axis=0, kind="stable")              # NaN sorts last
    xs = np.take_along_axis(Xf, order, axis=0)                 # sorted values
    onehot = np.zeros((n, n_classes), dtype=np.float64)
    onehot[np.arange(n), y_enc] = 1.0
    ys = onehot[order]                                         # (n, f, k)

    left = np.cumsum(ys, axis=0)[:-1]                          # (n-1, f, k)
    total = onehot.sum(axis=0)                                 # (k,)
    right = total - left
    n_left = np.arange(1, n, dtype=np.float64)[:, None]        # (n-1, 1)
    n_right = n - n_left
    parent = impurity(total)
    child = (n_left * impurity(left) + n_right * impurity(right)) / n
    gain = parent - child                                      # (n-1, f)

    lo, hi = xs[:-1], xs[1:]
    valid = (~np.isnan(lo)) & (lo != hi)                       # cut between different values
    valid &= (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    if not valid.any():
        return None
    gain = np.where(valid, gain, -np.inf)

    # deterministic tie-break: lowest feature, then lowest threshold
    g_by_feature = gain.T                                      # (f, n-1)
    best = g_by_feature.max()
    flat = np.flatnonzero(g_by_feature.ravel() >= best - _TIE_TOL)[0]
    fi, pos = divmod(int(flat), n - 1)

    a, b = xs[pos, fi], xs[pos + 1, fi]
    if np.isnan(b):
        thr = a                                                # finite left, missing right
    else:
        thr = a + (b - a) / 2.0
        if thr >= b:                                           # adjacent floats
            thr = a
    return {"feature": int(feats[fi]), "threshold": float(thr), "gain": float(gain[pos, fi]),
            "n_left": int(pos + 1)}


# ----------------------------------------------------------------------------
# The classifier
# ----------------------------------------------------------------------------
class DecisionTreeClassifier:
    """CART classification tree that learns from a stream by buffered retraining.

    Parameters
    ----------
    max_depth : int or None, default 5
        Maximum depth (the root has depth 0). None grows until the other
        stop rules fire.
    min_samples_split : int, default 2
        A node with fewer rows becomes a leaf.
    min_samples_leaf : int, default 1
        Every child must keep at least this many rows.
    max_features : None, int, float, 'sqrt' or 'log2', default None
        Features tried at each node. None tries all of them; otherwise a
        random subset (seeded by ``random_state``) is drawn per node.
    criterion : {'gini', 'entropy'}, default 'gini'
    min_impurity_decrease : float, default 0.0
        A split is kept only if its gain is larger than this.
    buffer_size : int, default 500
        Rows that must be buffered by :meth:`partial_fit` before the tree is
        retrained.
    random_state : int or None
        Seed for ``max_features`` sampling.

    Attributes
    ----------
    classes_ : ndarray
        Every label seen so far (sorted). ``predict_proba`` columns follow it.
    n_features_in_ : int
    n_retrains_ : int
        How many times the tree has been (re)built.
    n_samples_seen_ : int
        Rows received by fit / partial_fit.
    retrain_log_ : list of dict
        One record per (re)build: rows used, seconds, depth, leaves.
    feature_importances_ : ndarray, shape (n_features_in_,)
        Total weighted impurity decrease per feature, normalised to sum to 1.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]]); y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTreeClassifier(max_depth=2, buffer_size=2)
    >>> _ = tree.partial_fit(X[:2], y[:2]).partial_fit(X[2:], y[2:])
    >>> tree.n_retrains_
    2
    """

    def __init__(self, max_depth: Optional[int] = 5, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features=None, criterion: str = "gini", min_impurity_decrease: float = 0.0,
                 buffer_size: int = 500, random_state: Optional[int] = None):
        owner = "DecisionTreeClassifier"
        if max_depth is not None:
            check_positive_int(max_depth, "max_depth", owner, minimum=0)
        check_positive_int(min_samples_split, "min_samples_split", owner, minimum=2)
        check_positive_int(min_samples_leaf, "min_samples_leaf", owner)
        check_positive_int(buffer_size, "buffer_size", owner)
        if criterion not in _CRITERIA:
            raise ValueError(f"{owner}: criterion must be one of {sorted(_CRITERIA)}, got {criterion!r}")
        if min_impurity_decrease < 0:
            raise ValueError(f"{owner}: min_impurity_decrease must be >= 0, got {min_impurity_decrease}")
        self._check_max_features_param(max_features)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.buffer_size = buffer_size
        self.random_state = random_state
        self.reset()

    # -- parameters ----------------------------------------------------------
    @staticmethod
    def _check_max_features_param(mf) -> None:
        if mf is None or mf in ("sqrt", "log2"):
            return
        if isinstance(mf, bool):
            raise TypeError("max_features must be None, an int, a float in (0, 1], 'sqrt' or 'log2'")
        if isinstance(mf, (int, np.integer)):
            if mf < 1:
                raise ValueError(f"max_features must be >= 1, got {mf}")
            return
        if isinstance(mf, float):
            if not 0.0 < mf <= 1.0:
                raise ValueError(f"max_features as a float must be in (0, 1], got {mf}")
            return
        raise TypeError(f"max_features must be None, an int, a float, 'sqrt' or 'log2', got {mf!r}")

    def _n_candidate_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            k = int(np.sqrt(n_features))
        elif mf == "log2":
            k = int(np.log2(n_features)) if n_features > 1 else 1
        elif isinstance(mf, float):
            k = int(mf * n_features)
        else:
            k = int(mf)
        return int(min(max(k, 1), n_features))

    def __repr__(self) -> str:
        return (f"DecisionTreeClassifier(max_depth={self.max_depth}, criterion={self.criterion!r}, "
                f"buffer_size={self.buffer_size}, min_samples_split={self.min_samples_split}, "
                f"min_samples_leaf={self.min_samples_leaf}, max_features={self.max_features!r})")

    # -- state ---------------------------------------------------------------
    def reset(self) -> "DecisionTreeClassifier":
        """Forget the tree, the buffer and every counter."""
        self._rng = np.random.default_rng(self.random_state)
        self._buf_X: List[np.ndarray] = []
        self._buf_y: List[np.ndarray] = []
        self._n_buffered = 0
        self.n_retrains_ = 0
        self.n_samples_seen_ = 0
        self.retrain_log_: List[dict] = []
        self._nodes = None
        for attr in ("classes_", "n_features_in_", "feature_importances_", "_tree_n_classes"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self

    @property
    def is_fitted(self) -> bool:
        """True once a tree has been trained at least once."""
        return self._nodes is not None

    @property
    def buffer_fill_(self) -> int:
        """Rows waiting in the buffer for the next retrain."""
        return self._n_buffered

    # -- training ------------------------------------------------------------
    def _update_classes(self, y: np.ndarray, classes=None) -> None:
        new = np.unique(y) if classes is None else np.unique(np.concatenate([np.asarray(classes), y]))
        if not hasattr(self, "classes_"):
            self.classes_ = new
        else:
            try:
                self.classes_ = np.union1d(self.classes_, new)
            except TypeError:
                raise TypeError("labels in this chunk have a different type from earlier labels") from None

    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Train a tree from scratch on ``X, y`` (the buffer is emptied).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Numeric features; NaN allowed (routed right).
        y : array-like, shape (n_samples,)
            Class labels (ints or strings), no missing values.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D, contains infinity, is empty, or its length
            differs from y.
        """
        X, y = check_X_y(X, y, "DecisionTreeClassifier.fit")
        if X.shape[0] == 0:
            raise ValueError("DecisionTreeClassifier.fit requires at least one sample")
        self.reset()
        check_n_features(self, X, reset=True)
        self._update_classes(y)
        self.n_samples_seen_ = X.shape[0]
        self._train(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None) -> "DecisionTreeClassifier":
        """Buffer one chunk; retrain when ``buffer_size`` rows are waiting.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        classes : array-like, optional
            All labels the stream can contain. Optional: ``classes_`` also
            grows automatically as new labels arrive.

        Returns
        -------
        self

        Notes
        -----
        An empty chunk is a no-op. The retrain happens inside the call that
        makes the buffer reach ``buffer_size``; the buffer is then cleared,
        so each tree is trained on (at least) the latest ``buffer_size``
        rows, and memory stays O(buffer_size * n_features).
        """
        X, y = check_X_y(X_chunk, y_chunk, "DecisionTreeClassifier.partial_fit")
        if X.shape[0] == 0:
            return self
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"chunk has {X.shape[1]} features but earlier chunks had {self.n_features_in_}. "
                "If a OneHotEncoder is upstream, give it max_categories so its width is fixed."
            )
        check_n_features(self, X, reset=False)
        self._update_classes(y, classes)
        self._buf_X.append(X)
        self._buf_y.append(y)
        self._n_buffered += X.shape[0]
        self.n_samples_seen_ += X.shape[0]
        if self._n_buffered >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def flush(self) -> "DecisionTreeClassifier":
        """Retrain now on whatever is buffered (no-op if the buffer is empty)."""
        if self._n_buffered:
            self._retrain_from_buffer()
        return self

    def _retrain_from_buffer(self) -> None:
        X = np.vstack(self._buf_X)
        y = np.concatenate(self._buf_y)
        self._buf_X, self._buf_y, self._n_buffered = [], [], 0
        self._train(X, y)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        start = time.perf_counter()
        y_enc = np.searchsorted(self.classes_, y)
        self._build(X, y_enc, len(self.classes_))
        self.n_retrains_ += 1
        self.retrain_log_.append({
            "retrain": self.n_retrains_,
            "n_samples": int(X.shape[0]),
            "seconds": time.perf_counter() - start,
            "depth": self.get_depth(),
            "n_leaves": self.get_n_leaves(),
            "samples_seen": self.n_samples_seen_,
        })

    def _build(self, X: np.ndarray, y_enc: np.ndarray, n_classes: int) -> None:
        """Grow the tree depth-first with an explicit stack (no recursion limit)."""
        impurity = _CRITERIA[self.criterion]
        n_features = X.shape[1]
        n_cand = self._n_candidate_features(n_features)
        feature, threshold, left, right, value, imp, n_node, depth_of = [], [], [], [], [], [], [], []
        importances = np.zeros(n_features)

        def new_node(idx: np.ndarray, depth: int) -> int:
            counts = np.bincount(y_enc[idx], minlength=n_classes).astype(np.float64)
            feature.append(_LEAF); threshold.append(np.nan); left.append(-1); right.append(-1)
            value.append(counts); imp.append(float(impurity(counts))); n_node.append(idx.size)
            depth_of.append(depth)
            return len(feature) - 1

        root_idx = np.arange(X.shape[0])
        stack = [(new_node(root_idx, 0), root_idx)]
        while stack:
            node, idx = stack.pop()
            d, n = depth_of[node], idx.size
            if ((self.max_depth is not None and d >= self.max_depth) or n < self.min_samples_split
                    or imp[node] <= _TIE_TOL or n < 2 * self.min_samples_leaf):
                continue
            feats = None
            if n_cand < n_features:
                feats = np.sort(self._rng.choice(n_features, n_cand, replace=False))
            split = best_split(X[idx], y_enc[idx], n_classes, criterion=self.criterion,
                               min_samples_leaf=self.min_samples_leaf, features=feats)
            if split is None or split["gain"] <= self.min_impurity_decrease + _TIE_TOL:
                continue
            f, t = split["feature"], split["threshold"]
            go_left = X[idx, f] <= t                       # NaN -> False -> right
            li, ri = idx[go_left], idx[~go_left]
            importances[f] += n * split["gain"]
            feature[node], threshold[node] = f, t
            l_node = new_node(li, d + 1)
            r_node = new_node(ri, d + 1)
            left[node], right[node] = l_node, r_node
            stack.append((r_node, ri))                     # pushed first -> left built first
            stack.append((l_node, li))

        self._nodes = {
            "feature": np.array(feature, dtype=np.int64),
            "threshold": np.array(threshold, dtype=np.float64),
            "left": np.array(left, dtype=np.int64),
            "right": np.array(right, dtype=np.int64),
            "value": np.vstack(value),
            "impurity": np.array(imp),
            "n_samples": np.array(n_node, dtype=np.int64),
            "depth": np.array(depth_of, dtype=np.int64),
        }
        self._tree_n_classes = n_classes
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

    # -- prediction ----------------------------------------------------------
    def _check_predict_input(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise NotFittedError(
                "This DecisionTreeClassifier has not been trained yet: fit it, or keep calling "
                f"partial_fit until {self.buffer_size} rows are buffered (or call flush())."
            )
        X = check_numeric(X, "DecisionTreeClassifier.predict")
        check_n_features(self, X, reset=False)
        return X

    def apply(self, X) -> np.ndarray:
        """Index of the leaf each row lands in, shape ``(n_samples,)``.

        Vectorised by depth: all rows step down one level per iteration, so
        the Python loop runs at most ``depth`` times, never once per row.
        """
        X = self._check_predict_input(X)
        t = self._nodes
        node = np.zeros(X.shape[0], dtype=np.int64)
        active = np.flatnonzero(t["feature"][node] != _LEAF)
        while active.size:
            cur = node[active]
            go_left = X[active, t["feature"][cur]] <= t["threshold"][cur]
            node[active] = np.where(go_left, t["left"][cur], t["right"][cur])
            active = active[t["feature"][node[active]] != _LEAF]
        return node

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, shape ``(n_samples, len(classes_))``.

        The probability of a class is its share of the training rows in the
        leaf. Classes seen in the stream after the last retrain get 0.
        """
        leaves = self.apply(X)
        counts = self._nodes["value"][leaves]
        proba = counts / counts.sum(axis=1, keepdims=True)
        n_now = len(self.classes_)
        if proba.shape[1] < n_now:                  # classes appeared since the last retrain
            proba = np.hstack([proba, np.zeros((proba.shape[0], n_now - proba.shape[1]))])
        return proba

    def predict(self, X) -> np.ndarray:
        """Predicted labels, shape ``(n_samples,)`` (majority class of the leaf)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        """Accuracy of :meth:`predict` on ``X, y``."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y)) if y.size else float("nan")

    # -- inspection ----------------------------------------------------------
    def _require_tree(self) -> dict:
        if not self.is_fitted:
            raise NotFittedError("This DecisionTreeClassifier has not been trained yet")
        return self._nodes

    def get_depth(self) -> int:
        """Depth of the deepest leaf (a single-leaf tree has depth 0)."""
        return int(self._require_tree()["depth"].max())

    def get_n_leaves(self) -> int:
        return int((self._require_tree()["feature"] == _LEAF).sum())

    @property
    def node_count(self) -> int:
        return int(self._require_tree()["feature"].size)

    def memory_footprint(self) -> int:
        """Bytes held by the tree arrays plus the buffered chunks."""
        tree_bytes = sum(a.nbytes for a in self._nodes.values()) if self._nodes else 0
        buf_bytes = sum(a.nbytes for a in self._buf_X) + sum(a.nbytes for a in self._buf_y)
        return int(tree_bytes + buf_bytes)

    def export_text(self, feature_names=None, decimals: int = 3) -> str:
        """Readable if/else listing of the current tree.

        Example output::

            |--- area <= 5000.0
            |   |--- leaf: class=0 counts=[40, 3]
            |--- area >  5000.0  (or missing)
            |   |--- leaf: class=1 counts=[5, 52]
        """
        t = self._require_tree()
        names = (list(feature_names) if feature_names is not None
                 else [f"x{j}" for j in range(self.n_features_in_)])
        if len(names) != self.n_features_in_:
            raise ValueError(f"feature_names must have {self.n_features_in_} entries, got {len(names)}")
        tree_classes = self.classes_[:self._tree_n_classes]
        out: List[str] = []

        def walk(node: int, indent: int) -> None:     # recursion depth <= tree depth
            pad = "|   " * indent
            if t["feature"][node] == _LEAF:
                counts = t["value"][node]
                out.append(f"{pad}|--- leaf: class={tree_classes[int(np.argmax(counts))]} "
                           f"counts={counts.astype(int).tolist()}")
                return
            f, thr = names[t["feature"][node]], round(float(t["threshold"][node]), decimals)
            out.append(f"{pad}|--- {f} <= {thr}")
            walk(int(t["left"][node]), indent + 1)
            out.append(f"{pad}|--- {f} >  {thr}  (or missing)")
            walk(int(t["right"][node]), indent + 1)

        walk(0, 0)
        return "\n".join(out)
