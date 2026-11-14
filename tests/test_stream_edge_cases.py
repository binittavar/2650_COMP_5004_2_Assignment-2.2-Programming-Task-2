"""Extra edge cases across numcompute_stream (validation, mixed CSVs, plotting)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from numcompute_stream import visualise
from numcompute_stream.datasets import make_stream
from numcompute_stream.io import iter_chunks, load_table, stream_csv
from numcompute_stream.pipeline import ColumnTransformer, Pipeline
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

MIXED_CSV = "price,area,road,furnish\n10,100,yes,full\n20,,no,semi\n30,300,yes,\n40,400,no,full\n"


def test_mixed_csv_is_read_in_file_column_order(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(MIXED_CSV)
    X, y, names = load_table(str(path), "price", categorical=["road", "furnish"])
    assert names == ["area", "road", "furnish"] and X.dtype == object
    assert np.isnan(X[1, 0]) and X[2, 2] == ""            # numeric NaN, empty category
    np.testing.assert_array_equal(y, [10, 20, 30, 40])
    chunks = list(stream_csv(str(path), 3, "price", categorical=["road", "furnish"]))
    assert [c[0].shape for c in chunks] == [(3, 3), (1, 3)]


def test_full_mixed_pipeline_streams_end_to_end(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(MIXED_CSV)
    X, y, _ = load_table(str(path), "price", categorical=["road", "furnish"])
    ct = ColumnTransformer([("num", Pipeline([("imp", Imputer()), ("sc", StandardScaler())]), [0]),
                            ("cat", OneHotEncoder(max_categories=3), [1, 2])])
    pipe = Pipeline([("cols", ct), ("tree", DecisionTreeClassifier(buffer_size=2))])
    for Xc, yc in iter_chunks(X, (y > 20).astype(int), 2):
        pipe.partial_fit(Xc, yc)
    assert pipe.transform_features(X).shape == (4, 1 + 3 + 3)
    assert pipe.is_fitted and pipe.memory_footprint() > 0


def test_column_transformer_bool_mask_passthrough_and_errors():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ct = ColumnTransformer([("mm", MinMaxScaler(), np.array([True, False, False]))], remainder="passthrough")
    np.testing.assert_allclose(ct.fit_transform(X), [[0.0, 2.0, 3.0], [1.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="mask"):
        ColumnTransformer([("mm", MinMaxScaler(), [True])]).fit(X)
    with pytest.raises(ValueError, match="out of range"):
        ColumnTransformer([("mm", MinMaxScaler(), [7])]).fit(X)
    with pytest.raises(ValueError, match="unique"):
        ColumnTransformer([("a", MinMaxScaler(), [0]), ("a", MinMaxScaler(), [1])])
    with pytest.raises(ValueError, match="remainder"):
        ColumnTransformer([("a", MinMaxScaler(), [0])], remainder="keep")


def test_string_input_to_numeric_transformer_has_clear_error():
    with pytest.raises(TypeError, match="OneHotEncoder"):
        StandardScaler().partial_fit(np.array([["a"]]))


def test_merge_rejects_ema_statistics():
    a = StreamingStats(decay=0.5).update_stats(np.ones((2, 1)))
    with pytest.raises(ValueError, match="exact"):
        a.merge(StreamingStats().update_stats(np.ones((2, 1))))


def test_make_stream_options_and_validation():
    X, y = make_stream(500, n_features=4, nan_fraction=0.2, random_state=1)
    assert X.shape == (500, 4) and not np.isnan(X[:, 0]).any() and np.isnan(X).any()
    with pytest.raises(ValueError):
        make_stream(10, label_noise=0.7)


def test_tree_learns_through_nan_features_in_a_stream():
    X, y = make_stream(1500, nan_fraction=0.3, label_noise=0.0, random_state=4)
    tree = DecisionTreeClassifier(max_depth=6, buffer_size=500)
    for Xc, yc in iter_chunks(X, y, 100):
        tree.partial_fit(Xc, yc)
    assert tree.score(X, y) > 0.75


def test_plot_trainer_history(tmp_path):
    X, y = make_stream(400, random_state=0)
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=100)).run(iter_chunks(X, y, 50))
    fig = visualise.plot_trainer_history(tr, save_path=str(tmp_path / "h.png"))
    assert (tmp_path / "h.png").exists() and fig.axes
    with pytest.raises(ValueError, match="at most 3"):
        visualise.plot_trainer_history(tr, metrics=("a", "b", "c", "d"))
