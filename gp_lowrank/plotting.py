"""Reusable plotting helpers.

Every function here takes data in and returns a matplotlib ``Figure``
instead of calling ``plt.show()`` itself, so callers (scripts, notebooks,
tests) decide whether/how to display or save it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def plot_1d_regression(
    X_train, y_train, X_test, y_true, y_mean, y_std, title: str = ""
) -> Figure:
    fig = Figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    ax.plot(X_test, y_true, "r--", label="True function")
    ax.plot(X_test, y_mean, "b", label="GP mean prediction")
    ax.fill_between(
        np.ravel(X_test),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="b",
        alpha=0.2,
        label="95% predictive interval",
    )
    ax.scatter(X_train, y_train, color="black", marker="x", label="Observations")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_rank_and_runtime_vs_size(result, title: str = "") -> tuple[Figure, Figure]:
    """Two figures: runtime vs. size, and rank vs. size, for the three
    incomplete Cholesky variants in a :class:`~gp_lowrank.benchmarks.RankRuntimeResult`.
    """
    fig_time = Figure(figsize=(9, 5))
    ax = fig_time.add_subplot(111)
    ax.plot(result.sizes, result.times_no_pivot_ms, "s-", label="Incomplete Cholesky (no pivot)")
    ax.plot(result.sizes, result.times_pivot_ms, "^-", label="Incomplete Cholesky (pivot, dense)")
    ax.plot(
        result.sizes,
        result.times_pivot_matrixfree_ms,
        "o-",
        label="Incomplete Cholesky (pivot, matrix-free)",
    )
    ax.set_xlabel("Training set size n")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title(title or "Runtime vs. training set size")
    ax.legend()
    ax.grid(True, alpha=0.4)

    fig_rank = Figure(figsize=(9, 5))
    ax = fig_rank.add_subplot(111)
    ax.plot(result.sizes, result.ranks_no_pivot, "s-", label="No pivot")
    ax.plot(result.sizes, result.ranks_pivot, "^-", label="Pivot, dense")
    ax.plot(result.sizes, result.ranks_pivot_matrixfree, "o-", label="Pivot, matrix-free")
    ax.set_xlabel("Training set size n")
    ax.set_ylabel("Effective rank k")
    ax.set_title("Effective rank vs. training set size")
    ax.legend()
    ax.grid(True, alpha=0.4)

    return fig_time, fig_rank


def plot_accuracy_vs_eta(result, title: str = "") -> tuple[Figure, Figure]:
    """Two figures: MSE-vs-exact and runtime, both against eta, for an
    :class:`~gp_lowrank.benchmarks.AccuracyVsEtaResult`.
    """
    fig_mse = Figure(figsize=(9, 5))
    ax = fig_mse.add_subplot(111)
    ax.loglog(result.etas, result.mse_vs_exact, "go-", label="Low-rank vs. exact GP")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("MSE vs. exact GP prediction")
    ax.set_title(title or r"Prediction accuracy as a function of $\eta$")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    fig_time = Figure(figsize=(9, 5))
    ax = fig_time.add_subplot(111)
    ax.semilogx(result.etas, result.fit_time_s, "o-", label="Low-rank fit time")
    ax.axhline(result.exact_fit_time_s, linestyle="--", color="r", label="Exact fit time")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Time (s)")
    ax.set_title(r"Runtime as a function of $\eta$")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    return fig_mse, fig_time


def plot_eigenvalue_spectrum(matrix: np.ndarray, log_scale: bool = True) -> Figure:
    eigenvalues = np.linalg.eigvalsh(matrix)
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    if log_scale:
        positive = eigenvalues[eigenvalues > 0]
        bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 30)
        ax.hist(positive, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xscale("log")
    else:
        ax.hist(eigenvalues, bins=30, edgecolor="black", alpha=0.7)

    ax.set_title("Kernel matrix eigenvalue spectrum")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return fig
