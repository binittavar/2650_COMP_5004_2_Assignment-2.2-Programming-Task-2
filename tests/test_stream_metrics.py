"""Tests for numcompute_stream.metrics (update / reset / result protocol)."""

import numpy as np
import pytest

from numcompute import metrics as batch
from numcompute_stream.metrics import (F1, Accuracy, ConfusionMatrix, ErrorRate, MetricCollection, Precision,
                                       Recall, StreamingAUC)


@pytest.fixture
def labels():
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=400)
    y_pred = np.where(rng.random(400) < 0.8, y_true, 1 - y_true)
    return y_true, y_pred


def _feed(metric, y_true, y_pred, n_chunks=9):
    for idx in np.array_split(np.arange(len(y_true)), n_chunks):
        metric.update(y_true[idx], y_pred[idx])
    return metric


def test_streaming_values_equal_batch_values(labels):
    yt, yp = labels
    assert _feed(Accuracy(), yt, yp).result() == pytest.approx(batch.accuracy(yt, yp))
    assert _feed(Precision(), yt, yp).result() == pytest.approx(batch.precision(yt, yp))
    assert _feed(Recall(), yt, yp).result() == pytest.approx(batch.recall(yt, yp))
    assert _feed(F1(), yt, yp).result() == pytest.approx(batch.f1(yt, yp))
    assert _feed(ErrorRate(), yt, yp).result() == pytest.approx(1 - batch.accuracy(yt, yp))


def test_macro_and_micro_on_three_classes():
    rng = np.random.default_rng(0)
    yt, yp = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    assert _feed(F1(average="macro"), yt, yp).result() == pytest.approx(batch.f1(yt, yp, average="macro"))
    assert _feed(Precision(average="micro"), yt, yp).result() == pytest.approx(batch.accuracy(yt, yp))
    with pytest.raises(ValueError, match="at most 2"):
        _feed(F1(), yt, yp).result()


def test_confusion_matrix_grows_when_new_labels_arrive():
    cm = ConfusionMatrix()
    cm.update([0, 0], [0, 0])
    cm.update([2], [1])
    matrix, labels = cm.result()
    assert labels.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(matrix, [[2, 0, 0], [0, 0, 0], [0, 1, 0]])


def test_rolling_window_only_covers_recent_chunks():
    acc = Accuracy(window=2)
    acc.update([1, 1], [0, 0])          # 0.0 - will fall out
    acc.update([1, 1], [1, 1])
    acc.update([1, 1], [1, 0])
    assert acc.result() == pytest.approx(0.75)
    assert acc.history_ == pytest.approx([0.0, 0.5, 0.75])


def test_reset_and_empty_states():
    acc = Accuracy()
    assert np.isnan(acc.result())
    acc.update([1], [1]).reset()
    assert np.isnan(acc.result()) and acc.history_ == []
    acc.update([], [])                  # empty chunk recorded but changes nothing
    assert len(acc.history_) == 1 and np.isnan(acc.result())


def test_positive_class_never_seen_scores_zero():
    assert Precision().update([0, 0], [0, 0]).result() == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        Accuracy().update([0, 1], [0])


def test_streaming_auc_matches_batch_auc():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=600)
    s = np.clip(0.3 * y + rng.random(600) * 0.7, 0, 1).round(3)
    auc = StreamingAUC(n_bins=1000)
    for idx in np.array_split(np.arange(600), 7):
        auc.update(y[idx], s[idx])
    fpr, tpr, _ = batch.roc_curve(y, s)
    assert auc.result() == pytest.approx(batch.auc(fpr, tpr), abs=1e-3)
    f, t = auc.roc_curve()
    assert f[0] == 0 and t[-1] == 1


def test_auc_needs_both_classes_and_valid_scores():
    auc = StreamingAUC().update([1, 1], [0.2, 0.9])
    assert np.isnan(auc.result())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        auc.update([0], [1.5])


def test_auc_window_and_ties():
    auc = StreamingAUC(window=1)
    auc.update([0, 1], [0.9, 0.1])      # perfectly wrong
    auc.update([0, 1], [0.5, 0.5])      # all tied -> 0.5
    assert auc.result() == pytest.approx(0.5)


def test_metric_collection_routes_scores():
    mc = MetricCollection({"acc": Accuracy(), "auc": StreamingAUC()})
    mc.update([0, 1], [0, 1], y_score=[0.1, 0.9])
    assert mc.result() == {"acc": 1.0, "auc": 1.0}
    mc.update([0], [0])                 # no scores -> AUC skipped
    assert len(mc["auc"].history_) == 1
    with pytest.raises(TypeError):
        MetricCollection({"x": object()})
