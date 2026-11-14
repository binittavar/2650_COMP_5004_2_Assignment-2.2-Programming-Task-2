"""Tests for numcompute_stream.pipeline, stream, io, datasets and visualise."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from numcompute_stream.datasets import make_stream, save_csv
from numcompute_stream.io import iter_chunks, load_table, stream_csv
from numcompute_stream.metrics import F1, StreamingAUC
from numcompute_stream.pipeline import ColumnTransformer, Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream import visualise


def _pipe(buffer_size=200):
    return Pipeline([("impute", Imputer()), ("scale", StandardScaler()),
                     ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size))])


def test_pipeline_partial_fit_updates_every_step():
    X, y = make_stream(600, random_state=1)
    pipe = _pipe()
    for Xc, yc in iter_chunks(X, y, 50):
        pipe.partial_fit(Xc, yc)
    np.testing.assert_allclose(pipe["scale"].mean_, X.mean(axis=0))
    assert pipe["tree"].n_retrains_ == 3
    assert pipe.score(X, y) > 0.85
    assert pipe.predict_proba(X[:5]).shape == (5, 2)


def test_pipeline_rejects_step_without_partial_fit():
    class FitOnly:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("f", FitOnly()), ("tree", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="partial_fit"):
        pipe.partial_fit(np.ones((2, 1)), [0, 1])


def test_pipeline_slice_keeps_streaming_class():
    assert isinstance(_pipe()[:2], Pipeline)


def test_column_transformer_mixed_object_array():
    X = np.array([[1.0, "a"], [np.nan, "b"], [3.0, "a"]], dtype=object)
    ct = ColumnTransformer([("num", Imputer(), [0]), ("cat", OneHotEncoder(max_categories=3), [1])])
    ct.partial_fit(X[:2]).partial_fit(X[2:])
    out = ct.transform(X)
    assert out.dtype == np.float64 and out.shape == (3, 4)
    np.testing.assert_allclose(out[1, 0], 2.0)
    with pytest.raises(ValueError, match="columns"):
        ct.transform(X[:, :1])


def test_load_table_and_stream_csv_agree(tmp_path):
    X, y = make_stream(120, n_features=3, nan_fraction=0.1, random_state=3)
    path = tmp_path / "s.csv"
    save_csv(str(path), X, y)
    X2, y2, names = load_table(str(path), "label")
    np.testing.assert_allclose(X2, X)
    assert names == ["x0", "x1", "x2"]
    parts = list(stream_csv(str(path), 50, "label"))
    assert [len(p[1]) for p in parts] == [50, 50, 20]
    np.testing.assert_allclose(np.vstack([p[0] for p in parts]), X)


def test_load_table_unknown_column(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not found"):
        load_table(str(path), "zzz")


def test_iter_chunks_shuffle_is_a_permutation():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    chunks = list(iter_chunks(X, y, 3, shuffle=True, random_state=0))
    assert [len(c[1]) for c in chunks] == [3, 3, 3, 1]
    assert sorted(np.concatenate([c[1] for c in chunks]).tolist()) == list(range(10))


def test_trainer_prequential_log():
    X, y = make_stream(800, random_state=2)
    tr = StreamTrainer(_pipe(200), metrics={"f1": F1(), "auc": StreamingAUC()})
    tr.run(iter_chunks(X, y, 100))
    assert len(tr.log_) == 8
    assert not tr.log_[0]["scored"] and tr.log_[2]["scored"]   # first tree after chunk 1
    assert np.isnan(tr.history("cumulative_accuracy")[0])
    assert tr.history("retrained").sum() == 4
    assert tr.summary()["final_cumulative_accuracy"] > 0.8
    assert 0.5 < tr.log_[-1]["cumulative_auc"] <= 1.0
    assert tr.log_[-1]["memory_bytes"] > 0


def test_trainer_buffer_and_memory_logging_and_csv(tmp_path):
    X, y = make_stream(300, random_state=0)
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=1000), track_peak_memory=True)
    tr.run(iter_chunks(X, y, 100), finalize=True)
    assert tr.history("buffer_fill").tolist() == [100, 200, 300]
    assert tr.model.is_fitted                                   # finalize flushed the buffer
    assert tr.log_[0]["peak_memory_bytes"] > 0
    tr.to_csv(str(tmp_path / "log.csv"))
    assert (tmp_path / "log.csv").read_text().startswith("chunk,")
    with pytest.raises(KeyError):
        tr.history("nope")


def test_trainer_rejects_non_streaming_model():
    with pytest.raises(TypeError):
        StreamTrainer(object())


def test_drift_small_buffer_recovers_faster():
    X, y = make_stream(4000, drift_at=2000, label_noise=0.0, random_state=0)
    final = {}
    for bs in (100, 1000):
        tr = StreamTrainer(DecisionTreeClassifier(max_depth=4, buffer_size=bs), rolling_window=3)
        tr.run(iter_chunks(X, y, 100))
        final[bs] = tr.history("rolling_accuracy")[22:26].mean()   # just after the drift
    assert final[100] > final[1000]


def test_visualise_functions_return_figures(tmp_path):
    f1 = visualise.plot_metric_over_time([np.nan, 0.5, 0.7], "acc", "accuracy", retrain_chunks=[1],
                                         save_path=str(tmp_path / "a.png"))
    f2 = visualise.compare_models([0.5, 0.6], [0.4, 0.7, 0.8], ("a", "b"))
    f3 = visualise.plot_predictions_vs_ground_truth([0, 1, 1], [0, 1, 0])
    assert (tmp_path / "a.png").exists()
    for f in (f1, f2, f3):
        assert f.axes
    with pytest.raises(ValueError):
        visualise.plot_metric_over_time([], "t", "y")
    with pytest.raises(ValueError):
        visualise.plot_predictions_vs_ground_truth([0, 1], [0])
