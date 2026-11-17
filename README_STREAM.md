# NumCompute Stream

A **streaming, decision-tree learning framework** written with **plain Python, NumPy and matplotlib only** - no pandas, scikit-learn or other ML libraries.

It extends the group's `numcompute` toolkit (Assignment 2.1) with a new package, `numcompute_stream`, in which every component learns from data that arrives **in chunks**: a CART decision tree that retrains on a buffer, incremental preprocessing, chunk-based statistics, streaming metrics, an incremental pipeline, a trainer that logs everything, and reusable plots.

Built for **COMP 5004 - Assignment 2.2 (Programming Task 2)**, **dev** branch of group Beta's repository. The group code on `main` is unchanged; this branch adds `numcompute_stream/` and reuses `numcompute.io`, `numcompute.preprocessing`, `numcompute.pipeline`, `numcompute.metrics` and `numcompute.benchmarking`.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [How buffered retraining works](#how-buffered-retraining-works)
- [Modules](#modules)
- [Conventions and edge cases](#conventions-and-edge-cases)
- [Demo notebook](#demo-notebook)
- [Tests](#tests)
- [Benchmarks](#benchmarks)
- [Project layout](#project-layout)

## Installation

Python 3.9 or newer (tested on Python 3.13 and 3.14).

```bash
git clone https://github.com/binittavar/2650_COMP_5004_2_Assignment-2.2-Programming-Task-2.git
cd 2650_COMP_5004_2_Assignment-2.2-Programming-Task-2
git checkout dev                 # this assignment's branch

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"   # numpy, matplotlib (+ pytest, pytest-cov for the tests)

python -c "import numcompute_stream; print(numcompute_stream.__version__)"
```

## Quickstart

```python
import numpy as np
from numcompute_stream.datasets import make_stream
from numcompute_stream.io import iter_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.metrics import F1, StreamingAUC
from numcompute_stream.stream import StreamTrainer
from numcompute_stream import visualise as vis

X, y = make_stream(5000, drift_at=2500, random_state=0)       # 5,000 rows, rule changes half-way

pipe = Pipeline([
    ("impute", Imputer()),                                     # running mean, updated per chunk
    ("scale", StandardScaler()),                               # running Welford/Chan mean & variance
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])

trainer = StreamTrainer(pipe, metrics={"f1": F1(), "auc": StreamingAUC()}, verbose=True)
for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100):
    trainer.step(X_chunk, y_chunk)        # predict the chunk first, then pipe.partial_fit(X_chunk, y_chunk)

print(trainer.summary())
vis.plot_metric_over_time(trainer.history("rolling_accuracy"), "Rolling accuracy", "accuracy",
                          retrain_chunks=np.flatnonzero(trainer.history("retrained")), save_path="acc.png")
```

The tree on its own, exactly as in the specification:

```python
tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
for X_chunk, y_chunk in iter_chunks(X, y, 100):
    tree.partial_fit(X_chunk, y_chunk)
tree.n_retrains_, tree.buffer_fill_, tree.predict(X[:5])
```

## How buffered retraining works

```
partial_fit(chunk) ──► append chunk to buffer ──► buffer rows >= buffer_size ?
                                                    │ no  → keep predicting with the last tree
                                                    │ yes → rebuild the tree from scratch on the
                                                    │       buffered rows, then empty the buffer
```

* `fit(X, y)` trains from scratch on the given data (and empties the buffer).
* Between retrains, `predict` / `predict_proba` use the most recently trained tree.
* Before the first retrain there is no tree: `predict` raises `NotFittedError` saying how many rows are still needed. `flush()` retrains early on a partly filled buffer (e.g. at the end of a stream).
* `n_retrains_`, `buffer_fill_`, `retrain_log_` (rows, seconds, depth, leaves per retrain) and `memory_footprint()` expose what happened.

The tree itself follows the course notes: at each node **try** every question `x[:, f] <= t`, **keep** the one with the largest information gain (Gini or entropy), **split**, and **stop** at `max_depth`, a pure node, `min_samples_split` / `min_samples_leaf`, or no useful gain. The search over all features and thresholds is one vectorised computation: sort each column once, cumulative-sum one-hot labels to get the class counts left of every cut, and evaluate every cut's impurity at once.

## Modules

### `tree`
| Name | What it does |
|---|---|
| `DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None, criterion="gini", min_impurity_decrease=0.0, buffer_size=500, random_state=None)` | `fit`, `partial_fit`, `flush`, `predict`, `predict_proba`, `apply`, `score`, `export_text`, `get_depth`, `get_n_leaves`, `feature_importances_`, `memory_footprint` |
| `best_split(X, y_enc, n_classes, criterion, min_samples_leaf, features)` | Vectorised search for the best `(feature, threshold, gain)` |
| `gini(counts)`, `entropy(counts)` | Impurity of one or many class-count vectors |

### `preprocessing` (all have `partial_fit`, `fit`, `transform`, `fit_transform`)
| Class | Streaming behaviour |
|---|---|
| `StandardScaler(with_mean, with_std, decay=None)` | Exact running mean/variance (Chan-Welford merge) or an EMA when `decay` is set |
| `MinMaxScaler(feature_range, clip)` | Running min/max |
| `Imputer(strategy="mean"|"median"|"most_frequent"|"constant")` (alias `SimpleImputer`) | Fill values re-estimated after every chunk; median from a reservoir sample |
| `OneHotEncoder(handle_unknown="ignore", max_categories=None)` | Categories discovered chunk by chunk and **appended**, so old columns never move; `max_categories=k` fixes the width (extra categories go to `__other__`) |

### `stats`
| Name | What it does |
|---|---|
| `StreamingStats(decay, quantile_capacity, histogram_bins, histogram_range, histogram_window)` | `update_stats(X_chunk)`; `mean`, `variance(ddof)`, `std`, `min`, `max`, `count`, `quantile(q)`, `histogram()`, `merge(other)`, `summary()` |
| `StreamingQuantile(capacity, random_state)` | Reservoir-sample quantiles; exact until `capacity` rows |
| `StreamingHistogram(bins, range, window)` | Fixed-edge histogram, optional sliding window of chunks; counts under/overflow and NaN |
| `chunk_moments(X)`, `merge_moments(a, b)` | The exact chunk-merge formula used by everything above |

### `metrics` (all have `update(y_true_chunk, y_pred_chunk)`, `reset()`, `result()`, `history_`)
| Class | Notes |
|---|---|
| `Accuracy`, `ErrorRate` | `window=None` cumulative, `window=w` rolling over the last `w` chunks |
| `Precision`, `Recall`, `F1` | `average="binary"|"macro"|"micro"`, `pos_label` |
| `ConfusionMatrix(labels=None)` | Grows when new labels appear; old counts stay in place |
| `StreamingAUC(n_bins=1000, pos_label=1)` | Accumulates score histograms (O(n_bins) memory); `roc_curve()` |
| `MetricCollection({name: metric})` | Updates several metrics at once (scores go to AUC) |

### `pipeline`
| Class | What it does |
|---|---|
| `Pipeline(steps)` | Group `Pipeline` + `partial_fit`, `predict_proba`, `score`, `memory_footprint` |
| `ColumnTransformer([(name, transformer, columns)], remainder)` | Different transformers for different columns (numbers vs categories), with `partial_fit` |

### `stream`
| Class | What it does |
|---|---|
| `StreamTrainer(model, metrics, positive_label, rolling_window, log_every, verbose, track_peak_memory)` | `fit_chunk`, `score_chunk`, `step` (test-then-train), `run(stream)`, `history(key)`, `to_csv`, `summary`. Logs per chunk: chunk/cumulative/rolling accuracy, extra metrics, fit and predict time, retrain flag, buffer fill, memory footprint |

### `visualise` (matplotlib; every function takes `ax=`, `save_path=`, `show=` and returns the `Figure`)
| Function | Plot |
|---|---|
| `plot_metric_over_time(metric_values, title, ylabel, retrain_chunks=None)` | One metric across chunks, retrains marked |
| `compare_models(metric1, metric2, labels)` | Two models on one axis, e.g. two `buffer_size` values |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | Latest chunk: true vs predicted per row + confusion matrix |
| `plot_trainer_history(trainer, metrics)` | Up to three logged series of a `StreamTrainer` |

### `io` and `datasets`
| Name | What it does |
|---|---|
| `load_table(path, target, categorical, drop)` | Whole CSV with mixed numeric/text columns → `X, y, feature_names` (uses `numcompute.io.read_csv`) |
| `stream_csv(path, chunk_size, target, ...)` | Same, but lazily from disk with `numcompute.io.read_csv_chunks` |
| `iter_chunks(X, y, chunk_size, shuffle, random_state)` | Split in-memory arrays into chunks |
| `make_stream(n_samples, n_features, drift_at, label_noise, nan_fraction)`, `save_csv` | Synthetic stream with optional concept drift |

## Conventions and edge cases

* **Shapes.** `X` is 2-D `(n_samples, n_features)`; `y` 1-D `(n_samples,)`. Statistics are per column (`axis=0`). 1-D `X` is rejected with a `reshape(-1, 1)` hint; mismatched lengths, wrong widths and `±inf` raise `ValueError` naming the problem.
* **Empty chunks** are accepted everywhere and change nothing. The first non-empty chunk fixes `n_features_in_`.
* **NaN in X** means missing. Statistics skip it per column; scalers pass it through; the imputer fills it; the tree sends it to the right child (and can learn "missing vs present" as a split).
* **NaN in y** raises `ValueError`.
* **Ties.** Equal split gains (within 1e-12): lowest feature index, then lowest threshold. Equal values never straddle a threshold. Equal leaf votes: the class that sorts first.
* **Zero variance.** Constant features offer no split; scalers map constant columns to 0, never `inf`/`NaN`.
* **Numerical stability.** Variances use Chan's merge, never `sum(x²) − n·mean²` (tested at an offset of 1e9). Entropy uses `0·log 0 = 0`. Split thresholds are guarded against floating-point midpoints rounding onto the upper value.
* **New classes / categories mid-stream** are absorbed: `classes_`, confusion matrices and one-hot categories grow without moving existing entries.

## Demo notebook

`demo/stream_demo.ipynb` does what the specification asks of the demo: it loads `Dataset/Housing.csv` with the custom I/O pipeline (`numcompute_stream.io`), splits it into 25-row chunks, trains an impute/scale + one-hot + tree pipeline incrementally with `partial_fit`, logs every chunk, and plots cumulative accuracy, error rate per chunk, a comparison of two `buffer_size` values and predictions vs ground truth on the latest chunk.

```bash
python -m pip install notebook
python -m notebook demo/stream_demo.ipynb      # then Run > Run All Cells
```

Or open the notebook in VS Code (Jupyter extension), select your Python kernel and click **Run All**.

```bash
# (optional) the group's original demo
python -m notebook demo/quickstart.ipynb
```

## Tests

```bash
python -m pytest tests/ -q                                  # group tests + 89 new streaming tests
python -m pytest tests/test_stream_*.py -v                  # only the streaming package
python -m pytest tests/test_stream_*.py --cov=numcompute_stream --cov-report=term-missing
```

In Windows PowerShell the `*` wildcard is not expanded, so use:

```powershell
python -m pytest (Get-ChildItem tests\test_stream_*.py) -v
python -m pytest (Get-ChildItem tests\test_stream_*.py) --cov=numcompute_stream --cov-report=term-missing
```

The streaming tests (`tests/test_stream_*.py`, 89 test functions, 97 cases) check every streamed result against a batch computation (scaler vs the group's scaler, metrics vs the group's metrics, vectorised split vs brute force), plus edge cases: empty and single-class chunks, NaN features and labels, zero-variance columns, ties, adjacent floats, values offset by 1e9, new classes and categories mid-stream, width changes, invalid hyperparameters and concept drift.

## Benchmarks

```bash
python -m benchmark.stream_benchmarks      # writes benchmark/stream_results.md and benchmark/figures/
python -m benchmark.run_benchmarks         # the group's Assignment 2.1 benchmarks
```

Part A times loop-based vs vectorised versions of the four hot kernels (split search, prediction, running moments, confusion-matrix update); `numcompute.benchmarking.compare` checks both give the same result before reporting a speedup. Part B sweeps `buffer_size` on a 10,000-row drifting stream and on Housing.csv, recording accuracy, accuracy right after the drift, number of retrains, retraining time and memory. Latest results are in [`benchmark/stream_results.md`](benchmark/stream_results.md); plots in `benchmark/figures/`.

## Project layout

```
.
├── numcompute/               group package (Assignment 2.1, unchanged)
├── numcompute_stream/        this assignment
│   ├── __init__.py
│   ├── _validation.py        shared input checks
│   ├── io.py                 CSV loading + chunk iterators
│   ├── stats.py              chunk-based statistics
│   ├── preprocessing.py      incremental transformers
│   ├── tree.py               DecisionTreeClassifier (buffered retraining)
│   ├── metrics.py            streaming metrics
│   ├── pipeline.py           incremental Pipeline + ColumnTransformer
│   ├── stream.py             StreamTrainer
│   ├── visualise.py          matplotlib plots
│   └── datasets.py           synthetic drifting streams
├── tests/                    test_<module>.py (group) + test_stream_*.py (new)
├── benchmark/
│   ├── run_benchmarks.py     group benchmarks
│   ├── stream_benchmarks.py  loop vs vectorised + buffer_size sweep
│   ├── loop_reference.py     loop versions used by the benchmark
│   ├── stream_results.md
│   └── figures/
├── demo/
│   ├── quickstart.ipynb      group demo
│   └── stream_demo.ipynb     streaming demo
├── report/                   technical report (PDF)
├── docs/
│   └── demo_video_script.md  plan for the demo video
├── Dataset/Housing.csv
└── pyproject.toml
```
