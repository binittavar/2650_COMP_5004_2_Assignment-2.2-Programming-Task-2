"""Benchmarks for numcompute_stream (Assignment 2.2).

Part A - loop vs vectorised, for the four kernels a stream runs most:
    split search, tree prediction, running moments, confusion-matrix update.
Part B - buffer_size sweep: accuracy, number of retrains, retraining time
    and memory as the buffer grows, on (1) a 10,000-row synthetic stream
    with concept drift half-way and (2) the group's Housing.csv.

Run from the repository root:

    python -m benchmark.stream_benchmarks

Writes benchmark/stream_results.md and the figures in benchmark/figures/.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmark import loop_reference as ref
from numcompute.benchmarking import compare, format_results_table
from numcompute_stream.datasets import make_stream
from numcompute_stream.io import iter_chunks, load_table
from numcompute_stream.pipeline import ColumnTransformer, Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import PALETTE, compare_models

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
HOUSING_CAT = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea",
               "furnishingstatus"]


# ----------------------------------------------------------------------------- Part A
def part_a(repeats: int = 3):
    rng = np.random.default_rng(0)
    rows = []

    X = rng.normal(size=(2000, 5)).round(2)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(np.int64)
    r = compare(ref.best_split_vectorised, ref.best_split_loop, X, y, 2, repeats=repeats)
    r["name"] = "best_split (2,000 rows x 5 features)"
    rows.append(r)

    Xb = rng.normal(size=(5000, 6))
    yb = (Xb[:, 0] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=8).fit(Xb, yb)
    Xp = rng.normal(size=(50_000, 6))
    r = compare(tree.apply, lambda A: ref.predict_loop(tree, A), Xp, repeats=repeats)
    r["name"] = "tree predict (50,000 rows, depth 8)"
    rows.append(r)

    chunks = np.array_split(rng.normal(5, 2, size=(100_000, 3)), 50)
    r = compare(ref.moments_vectorised, ref.moments_loop, chunks, repeats=repeats)
    r["name"] = "running mean/var (50 chunks, 100,000 x 3)"
    rows.append(r)

    yt = np.array_split(rng.integers(0, 4, 200_000), 20)
    yp = np.array_split(rng.integers(0, 4, 200_000), 20)
    r = compare(ref.confusion_vectorised, ref.confusion_loop, yt, yp, 4, repeats=repeats)
    r["name"] = "confusion matrix (20 chunks, 200,000 labels)"
    rows.append(r)
    return rows


# ----------------------------------------------------------------------------- Part B
def _sweep(X, y, chunk, buffer_sizes, make_model, drift_chunk=None):
    out = []
    trainers = {}
    for bs in buffer_sizes:
        tr = StreamTrainer(make_model(bs), rolling_window=3, name=f"buffer_size={bs}")
        tr.run(iter_chunks(X, y, chunk))
        tree = tr.model if isinstance(tr.model, DecisionTreeClassifier) else tr.model["tree"]
        secs = np.array([r["seconds"] for r in tree.retrain_log_])
        roll = tr.history("rolling_accuracy")
        rec = {
            "buffer_size": bs,
            "final_cum_acc": tr.summary()["final_cumulative_accuracy"],
            "mean_rolling_acc": float(np.nanmean(roll)),
            "first_scored_chunk": int(np.flatnonzero(tr.history("scored") > 0)[0]),
            "n_retrains": tree.n_retrains_,
            "mean_retrain_ms": 1e3 * float(secs.mean()),
            "total_retrain_ms": 1e3 * float(secs.sum()),
            "max_memory_kib": tr.summary()["max_memory_bytes"] / 1024,
        }
        if drift_chunk is not None:
            after = roll[drift_chunk:drift_chunk + 10]
            rec["post_drift_acc"] = float(np.nanmean(after))
        out.append(rec)
        trainers[bs] = tr
    return out, trainers


def part_b():
    X, y = make_stream(10_000, n_features=6, drift_at=5000, label_noise=0.05, random_state=0)
    synth, synth_tr = _sweep(X, y, 100, [100, 250, 500, 1000, 2000],
                             lambda bs: DecisionTreeClassifier(max_depth=6, buffer_size=bs), drift_chunk=50)

    Xh, price, names = load_table(os.path.join(ROOT, "Dataset", "Housing.csv"), "price", categorical=HOUSING_CAT)
    yh = (price >= np.median(price)).astype(int)
    num = [i for i, n in enumerate(names) if n not in HOUSING_CAT]
    cat = [i for i, n in enumerate(names) if n in HOUSING_CAT]
    order = np.random.default_rng(0).permutation(len(yh))     # file is sorted by price
    Xh, yh = Xh[order], yh[order]

    def housing_model(bs):
        cols = ColumnTransformer([("num", Pipeline([("imp", Imputer()), ("sc", StandardScaler())]), num),
                                  ("cat", OneHotEncoder(max_categories=4), cat)])
        return Pipeline([("cols", cols),
                         ("tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, buffer_size=bs))])

    house, _ = _sweep(Xh, yh, 25, [50, 100, 200, 400], housing_model)
    return synth, synth_tr, house


def _table(rows, keys):
    head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    body = ""
    for r in rows:
        cells = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def _figures(synth, synth_tr):
    os.makedirs(FIG, exist_ok=True)
    bs = [r["buffer_size"] for r in synth]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(bs, [r["mean_rolling_acc"] for r in synth], marker="o", color=PALETTE["series"][0], lw=2,
            label="mean rolling accuracy")
    a1.plot(bs, [r["post_drift_acc"] for r in synth], marker="s", ls="--", color=PALETTE["series"][1], lw=2,
            label="accuracy in 10 chunks after drift")
    a1.set_xscale("log"); a1.set_xlabel("buffer_size"); a1.set_ylabel("accuracy")
    a1.set_title("Accuracy vs buffer size", loc="left"); a1.legend(frameon=False); a1.grid(alpha=.3)
    a2.plot(bs, [r["mean_retrain_ms"] for r in synth], marker="o", color=PALETTE["series"][0], lw=2,
            label="time per retrain")
    a2.plot(bs, [r["total_retrain_ms"] for r in synth], marker="s", ls="--", color=PALETTE["series"][1], lw=2,
            label="total retrain time")
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("buffer_size"); a2.set_ylabel("milliseconds")
    a2.set_title("Retraining cost vs buffer size", loc="left"); a2.legend(frameon=False); a2.grid(alpha=.3)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "buffer_size_tradeoff.png"), dpi=150)
    plt.close(fig)
    f = compare_models(synth_tr[100].history("rolling_accuracy"), synth_tr[2000].history("rolling_accuracy"),
                       ("buffer_size=100", "buffer_size=2000"),
                       title="Rolling accuracy on a drifting stream (drift at chunk 50)", ylabel="rolling accuracy",
                       save_path=os.path.join(FIG, "drift_compare.png"))
    plt.close(f)


def main():
    a = part_a()
    synth, synth_tr, house = part_b()
    _figures(synth, synth_tr)
    keys_s = ["buffer_size", "final_cum_acc", "mean_rolling_acc", "post_drift_acc", "first_scored_chunk",
              "n_retrains", "mean_retrain_ms", "total_retrain_ms", "max_memory_kib"]
    keys_h = ["buffer_size", "final_cum_acc", "mean_rolling_acc", "first_scored_chunk", "n_retrains",
              "mean_retrain_ms", "total_retrain_ms", "max_memory_kib"]
    md = "# numcompute_stream benchmarks\n\n"
    md += "## A. Loop vs vectorised\n\n" + format_results_table(a) + "\n\n"
    md += ("Each pair is checked for identical results by `numcompute.benchmarking.compare` before the "
           "speedup is reported. Mean of 3 timed runs after 1 warm-up.\n\n")
    md += "## B1. buffer_size sweep - synthetic stream (10,000 rows, chunks of 100, drift at row 5,000)\n\n"
    md += _table(synth, keys_s) + "\n"
    md += "## B2. buffer_size sweep - Housing.csv (545 rows, chunks of 25, shuffled with seed 0)\n\n"
    md += _table(house, keys_h) + "\n"
    md += ("**Environment**\n\n"
           f"- Date: {date.today().isoformat()}\n- Python: {sys.version.split()[0]}\n"
           f"- NumPy: {np.__version__}\n- OS: {platform.platform()}\n"
           f"- CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} logical cores)\n")
    with open(os.path.join(HERE, "stream_results.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
