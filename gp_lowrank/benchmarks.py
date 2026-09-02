"""Synthetic-data benchmarking utilities.

These reproduce, on generated data, the runtime/accuracy/rank studies
used to validate the low-rank emulator (originally run against
project-specific simulator datasets that are not part of this
repository). Consolidated from three near-duplicate exploration scripts
into a single, parametrized, reusable set of functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .cholesky import (
    incomplete_cholesky,
    incomplete_cholesky_pivoted,
    incomplete_cholesky_pivoted_rbf,
)
from .kernels import rbf_covariance
from .regression import GaussianProcessRegressor


def generate_synthetic_points(n: int, d: int = 4, seed: int | None = 0) -> np.ndarray:
    """Uniform random points in [0, 1]^d, standing in for a real design
    of experiments / simulator input space.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def generate_random_spd_matrix(n: int, seed: int | None = 0) -> np.ndarray:
    """Random symmetric positive-definite matrix with eigenvalues in
    [1, 10], for algorithm-level (kernel-agnostic) benchmarks.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.random((n, n)))
    D = np.diag(rng.uniform(1, 10, n))
    return Q.T @ D @ Q


@dataclass
class RankRuntimeResult:
    sizes: list[int]
    times_no_pivot_ms: list[float] = field(default_factory=list)
    ranks_no_pivot: list[int] = field(default_factory=list)
    times_pivot_ms: list[float] = field(default_factory=list)
    ranks_pivot: list[int] = field(default_factory=list)
    times_pivot_matrixfree_ms: list[float] = field(default_factory=list)
    ranks_pivot_matrixfree: list[int] = field(default_factory=list)


def benchmark_rank_and_runtime_vs_size(
    sizes: list[int],
    length_scale: float,
    amplitude: float,
    eta: float,
    d: int = 4,
    repeats: int = 3,
    seed: int = 0,
) -> RankRuntimeResult:
    """Compare runtime and effective rank of the three incomplete
    Cholesky variants as the training set size grows.

    The matrix-free pivoted variant (:func:`incomplete_cholesky_pivoted_rbf`)
    is the one used in production, since it never builds the dense
    kernel matrix; the other two are included as points of comparison.
    """
    result = RankRuntimeResult(sizes=list(sizes))

    for n in sizes:
        no_pivot_times, no_pivot_ranks = [], []
        pivot_times, pivot_ranks = [], []
        matrixfree_times, matrixfree_ranks = [], []

        for rep in range(repeats):
            X = generate_synthetic_points(n, d=d, seed=seed + rep)
            K = rbf_covariance(X, X, length_scale, amplitude)

            start = time.perf_counter()
            _, rank = incomplete_cholesky(K, eta)
            no_pivot_times.append((time.perf_counter() - start) * 1000)
            no_pivot_ranks.append(rank)

            start = time.perf_counter()
            _, rank = incomplete_cholesky_pivoted(K, eta)
            pivot_times.append((time.perf_counter() - start) * 1000)
            pivot_ranks.append(rank)

            start = time.perf_counter()
            _, rank = incomplete_cholesky_pivoted_rbf(X, length_scale, amplitude, eta)
            matrixfree_times.append((time.perf_counter() - start) * 1000)
            matrixfree_ranks.append(rank)

        result.times_no_pivot_ms.append(float(np.mean(no_pivot_times)))
        result.ranks_no_pivot.append(int(np.mean(no_pivot_ranks)))
        result.times_pivot_ms.append(float(np.mean(pivot_times)))
        result.ranks_pivot.append(int(np.mean(pivot_ranks)))
        result.times_pivot_matrixfree_ms.append(float(np.mean(matrixfree_times)))
        result.ranks_pivot_matrixfree.append(int(np.mean(matrixfree_ranks)))

    return result


@dataclass
class AccuracyVsEtaResult:
    etas: np.ndarray
    ranks: list[int] = field(default_factory=list)
    mse_vs_exact: list[float] = field(default_factory=list)
    fit_time_s: list[float] = field(default_factory=list)
    exact_fit_time_s: float = 0.0


def benchmark_accuracy_vs_eta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    length_scale: float,
    amplitude: float,
    noise_std: float,
    etas: np.ndarray,
) -> AccuracyVsEtaResult:
    """For a range of ``eta`` tolerances, fit the low-rank GP and compare
    its predictions against the exact GP on the same data.

    This is the key plot behind the "near-identical accuracy, ~100x
    speedup" claim: MSE vs. the exact predictions should stay flat and
    small while rank (and runtime) drop sharply as eta grows.
    """
    exact_gp = GaussianProcessRegressor(length_scale, amplitude, noise_std, method="exact")
    exact_gp.fit(X_train, y_train)
    exact_mean, _ = exact_gp.predict(X_test)

    result = AccuracyVsEtaResult(etas=np.asarray(etas), exact_fit_time_s=exact_gp.fit_time_s)

    for eta in etas:
        low_rank_gp = GaussianProcessRegressor(
            length_scale, amplitude, noise_std, method="low_rank", eta=eta
        )
        low_rank_gp.fit(X_train, y_train)
        mean, _ = low_rank_gp.predict(X_test)

        result.ranks.append(low_rank_gp.rank_)
        result.mse_vs_exact.append(float(np.mean((mean - exact_mean) ** 2)))
        result.fit_time_s.append(low_rank_gp.fit_time_s)

    return result
