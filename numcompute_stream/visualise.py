"""
visualise.py -- Reusable matplotlib plots for streaming experiments.

Provides:
    plot_metric_over_time(metric_values, title, ylabel, ...)
        -> one metric (e.g. cumulative accuracy) across chunks, with optional
           markers at the chunks where the tree was retrained
    compare_models(metric1, metric2, labels, ...)
        -> two models' streaming metrics on one axis (e.g. two buffer sizes)
    plot_predictions_vs_ground_truth(y_true, y_pred, ...)
        -> latest chunk: true vs predicted label per row, plus its confusion
           matrix
    plot_trainer_history(trainer, metrics=(...), ...)
        -> convenience: several logged metrics of a StreamTrainer

Every function
--------------
* draws on ``ax`` if given (so plots compose into your own figures), else
  creates a new figure;
* saves to ``save_path`` if given (PNG/PDF/SVG, chosen by extension);
* calls ``plt.show()`` only if ``show=True`` (in notebooks the figure
  renders inline anyway);
* returns the matplotlib ``Figure`` so callers can tweak or save it later.

Missing values (NaN) in a metric series - for instance chunks scored before
the first retrain - are left as gaps rather than drawn as zero.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

__all__ = [
    "PALETTE",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_trainer_history",
]

#: Colour-blind-checked categorical order (blue, orange, aqua) plus text inks.
PALETTE = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#d9d8d4",
    "surface": "#fcfcfb",
    "good": "#1baf7a",
    "bad": "#e34948",
}


def _new_axes(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=PALETTE["text"])
    ax.set_xlabel(xlabel, color=PALETTE["muted"])
    ax.set_ylabel(ylabel, color=PALETTE["muted"])
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _finish(fig, save_path: Optional[str], show: bool):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _as_series(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence of numbers (one per chunk), got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty; there is nothing to plot")
    return arr


def plot_metric_over_time(metric_values: Sequence[float], title: str = "Metric over time",
                          ylabel: str = "value", *, xlabel: str = "chunk", retrain_chunks=None,
                          ylim=None, ax=None, figsize=(7, 3.6), save_path: Optional[str] = None,
                          show: bool = False):
    """Plot one metric across chunks.

    Parameters
    ----------
    metric_values : sequence of float, length n_chunks
        Value after each chunk (NaN = not available yet; drawn as a gap).
    title, ylabel : str
    retrain_chunks : sequence of int, optional
        Chunk indices where the model was retrained; drawn as faint vertical
        lines so accuracy jumps can be matched to retrains.
    ylim : (float, float), optional
    ax : matplotlib Axes, optional
    save_path : str, optional
        File to save the figure to.
    show : bool, default False

    Returns
    -------
    matplotlib.figure.Figure
    """
    y = _as_series(metric_values, "metric_values")
    fig, ax = _new_axes(ax, figsize)
    x = np.arange(y.size)
    if retrain_chunks is not None:
        for i, c in enumerate(retrain_chunks):
            ax.axvline(c, color=PALETTE["grid"], linewidth=1, linestyle="--", zorder=0,
                       label="retrain" if i == 0 else None)
    ax.plot(x, y, color=PALETTE["series"][0], linewidth=2, marker="o" if y.size <= 40 else None,
            markersize=4, label=ylabel)
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size:
        last = finite[-1]
        ax.annotate(f"{y[last]:.3f}", (x[last], y[last]), textcoords="offset points", xytext=(6, 0),
                    va="center", color=PALETTE["text"], fontsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style(ax, title, xlabel, ylabel)
    if retrain_chunks is not None and len(retrain_chunks):
        ax.legend(frameon=False, loc="lower right")
    return _finish(fig, save_path, show)


def compare_models(metric1: Sequence[float], metric2: Sequence[float], labels=("model 1", "model 2"), *,
                   title: str = "Model comparison", ylabel: str = "accuracy", xlabel: str = "chunk",
                   ylim=None, ax=None, figsize=(7, 3.6), save_path: Optional[str] = None, show: bool = False):
    """Plot two models' streaming metrics on one shared axis.

    Parameters
    ----------
    metric1, metric2 : sequence of float
        One value per chunk; they may differ in length (shorter series stop
        early).
    labels : (str, str)
        Legend names, e.g. ``("buffer_size=100", "buffer_size=500")``. Each
        line is also labelled directly at its end, so colour is never the
        only cue.

    Returns
    -------
    matplotlib.figure.Figure
    """
    a = _as_series(metric1, "metric1")
    b = _as_series(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError(f"labels must hold exactly two names, got {labels!r}")
    fig, ax = _new_axes(ax, figsize)
    for k, (series, label, style) in enumerate(((a, labels[0], "-"), (b, labels[1], "--"))):
        x = np.arange(series.size)
        ax.plot(x, series, color=PALETTE["series"][k], linewidth=2, linestyle=style, label=label)
        finite = np.flatnonzero(np.isfinite(series))
        if finite.size:
            i = finite[-1]
            ax.annotate(f"{label}: {series[i]:.3f}", (x[i], series[i]), textcoords="offset points",
                        xytext=(6, 8 if k == 0 else -10), va="center", fontsize=9, color=PALETTE["text"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, loc="lower right")
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title: str = "Latest chunk: predictions vs ground truth",
                                     max_points: int = 200, figsize=(10, 3.6), save_path: Optional[str] = None,
                                     show: bool = False):
    """Show the latest chunk's predictions against the true labels.

    Left panel: one row per sample (x axis), the true label as a filled dot
    and the prediction as a ring; wrong predictions are marked with a red
    cross and counted in the title. Right panel: the chunk's confusion
    matrix with counts written in each cell.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    max_points : int
        Only the first ``max_points`` rows are drawn on the left (all rows
        count in the confusion matrix).

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must be 1-D and the same shape, got {y_true.shape} and {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("y_true is empty; there is nothing to plot")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    t_idx = np.searchsorted(labels, y_true)
    p_idx = np.searchsorted(labels, y_pred)
    k = labels.size
    cm = np.bincount(t_idx * k + p_idx, minlength=k * k).reshape(k, k)
    wrong = t_idx != p_idx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1]})
    n = min(max_points, y_true.size)
    x = np.arange(n)
    ax1.scatter(x, t_idx[:n], s=36, color=PALETTE["series"][0], label="true", zorder=2)
    ax1.scatter(x, p_idx[:n], s=90, facecolors="none", edgecolors=PALETTE["series"][1], linewidths=1.5,
                label="predicted", zorder=3)
    ax1.scatter(x[wrong[:n]], p_idx[:n][wrong[:n]], marker="x", s=50, color=PALETTE["bad"],
                label="wrong", zorder=4)
    ax1.set_yticks(range(k), [str(v) for v in labels])
    ax1.set_ylim(-0.6, k - 0.4)
    acc = 1.0 - wrong.mean()
    _style(ax1, f"{title}  ({int(wrong.sum())} wrong of {y_true.size}, accuracy {acc:.3f})", "row in chunk", "class")
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    ax2.imshow(cm, cmap="Blues")
    for i in range(k):
        for j in range(k):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else PALETTE["text"], fontsize=11)
    ax2.set_xticks(range(k), [str(v) for v in labels])
    ax2.set_yticks(range(k), [str(v) for v in labels])
    ax2.set_xlabel("predicted", color=PALETTE["muted"])
    ax2.set_ylabel("true", color=PALETTE["muted"])
    ax2.set_title("confusion matrix", loc="left", fontsize=12, color=PALETTE["text"])
    return _finish(fig, save_path, show)


def plot_trainer_history(trainer, metrics=("cumulative_accuracy", "rolling_accuracy"), *,
                         title: str = "Streaming metrics", ylabel: str = "value", ax=None, figsize=(7, 3.6),
                         save_path: Optional[str] = None, show: bool = False):
    """Plot up to three logged series of a :class:`~numcompute_stream.stream.StreamTrainer`.

    Retrain chunks from the trainer's log are marked with dashed lines.
    """
    if len(metrics) > 3:
        raise ValueError("plot_trainer_history draws at most 3 series; split them into several plots")
    fig, ax = _new_axes(ax, figsize)
    retrains = np.flatnonzero(trainer.history("retrained") > 0)
    for i, c in enumerate(retrains):
        ax.axvline(c, color=PALETTE["grid"], linewidth=1, linestyle="--", zorder=0,
                   label="retrain" if i == 0 else None)
    for k, name in enumerate(metrics):
        y = trainer.history(name)
        ax.plot(np.arange(y.size), y, color=PALETTE["series"][k], linewidth=2,
                linestyle=("-", "--", ":")[k], label=name)
    _style(ax, title, "chunk", ylabel)
    ax.legend(frameon=False, loc="lower right")
    return _finish(fig, save_path, show)
