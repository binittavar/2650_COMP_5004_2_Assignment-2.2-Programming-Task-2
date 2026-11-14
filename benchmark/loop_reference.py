"""Plain-Python loop versions of the streaming package's vectorised kernels.

They exist only so the benchmarks can show what vectorisation buys. Each
returns exactly what its vectorised counterpart returns, and
``numcompute.benchmarking.compare`` checks that before reporting a speedup.
"""

import math

import numpy as np


def _gini_counts(counts, n):
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counts)


def best_split_loop(X, y_enc, n_classes):
    """Try every feature and every cut with Python loops; same rules as tree.best_split."""
    n, d = X.shape
    total = [0] * n_classes
    for label in y_enc:
        total[label] += 1
    parent = _gini_counts(total, n)
    best = (-1, 0.0, -math.inf)
    for f in range(d):
        order = sorted(range(n), key=lambda i: X[i, f])
        left = [0] * n_classes
        for pos in range(n - 1):
            left[y_enc[order[pos]]] += 1
            a, b = X[order[pos], f], X[order[pos + 1], f]
            if a == b:
                continue
            n_left = pos + 1
            right = [total[k] - left[k] for k in range(n_classes)]
            gain = parent - (n_left * _gini_counts(left, n_left)
                             + (n - n_left) * _gini_counts(right, n - n_left)) / n
            if gain > best[2] + 1e-12:
                best = (f, a + (b - a) / 2.0, gain)
    return np.array(best)


def best_split_vectorised(X, y_enc, n_classes):
    from numcompute_stream.tree import best_split

    s = best_split(X, y_enc, n_classes)
    return np.array([s["feature"], s["threshold"], s["gain"]])


def predict_loop(tree, X):
    """Walk the tree once per row."""
    t = tree._nodes
    out = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        node = 0
        while t["feature"][node] != -1:
            node = t["left"][node] if X[i, t["feature"][node]] <= t["threshold"][node] else t["right"][node]
        out[i] = node
    return out


def moments_loop(chunks):
    """Welford, one value at a time (the group package's StreamingStats), per column."""
    from numcompute.stats import StreamingStats as ScalarStats

    d = chunks[0].shape[1]
    acc = [ScalarStats() for _ in range(d)]
    for chunk in chunks:
        for row in chunk:
            for j in range(d):
                acc[j].update(float(row[j]))
    return np.array([[a.mean for a in acc], [a.variance for a in acc]])


def moments_vectorised(chunks):
    from numcompute_stream.stats import StreamingStats

    s = StreamingStats()
    for chunk in chunks:
        s.update_stats(chunk)
    return np.array([s.mean, s.variance()])


def confusion_loop(chunks_true, chunks_pred, n_labels):
    cm = [[0] * n_labels for _ in range(n_labels)]
    for yt, yp in zip(chunks_true, chunks_pred):
        for a, b in zip(yt, yp):
            cm[a][b] += 1
    return np.array(cm)


def confusion_vectorised(chunks_true, chunks_pred, n_labels):
    from numcompute_stream.metrics import ConfusionMatrix

    m = ConfusionMatrix(labels=np.arange(n_labels))
    for yt, yp in zip(chunks_true, chunks_pred):
        m.update(yt, yp)
    return m.result()[0]
