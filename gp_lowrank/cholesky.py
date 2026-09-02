"""Exact and incomplete (low-rank) Cholesky factorizations.

The incomplete factorizations are the core building block behind the
scalable GP emulator: instead of computing the full O(n^3) Cholesky
factor of the kernel matrix, they stop early once a target fraction
``eta`` of the trace has been captured, returning a low-rank factor
``L`` of shape ``(n, k)`` with ``k << n`` such that ``K ~ L @ L.T``.
"""

from __future__ import annotations

import numpy as np

from .kernels import _as_2d, rbf_covariance_column


def cholesky_dense(matrix: np.ndarray) -> np.ndarray:
    """Textbook (from-scratch) exact Cholesky factorization, ``A = L @ L.T``.

    For production use prefer ``numpy.linalg.cholesky``, which is faster
    and more numerically robust; this implementation is kept for
    reference/teaching purposes.
    """
    n = matrix.shape[0]
    L = np.zeros((n, n))
    L[0, 0] = np.sqrt(matrix[0, 0])
    L[1:, 0] = matrix[1:, 0] / L[0, 0]

    for i in range(1, n):
        L[i, i] = np.sqrt(matrix[i, i] - np.dot(L[i, :i], L[i, :i]))
        if i + 1 < n:
            L[i + 1 :, i] = (
                matrix[i + 1 :, i] - L[i + 1 :, :i] @ L[i, :i]
            ) / L[i, i]

    return L


def incomplete_cholesky(matrix: np.ndarray, eta: float) -> tuple[np.ndarray, int]:
    """Incomplete Cholesky factorization without pivoting.

    Stops once the captured trace reaches ``(1 - eta) * trace(matrix)``.

    Returns
    -------
    L : (n, k) array such that ``matrix ~ L @ L.T``
    k : effective rank reached
    """
    n = matrix.shape[0]
    target_trace = (1.0 - eta) * np.trace(matrix)

    L = np.zeros((n, n))
    L[0, 0] = np.sqrt(matrix[0, 0])
    L[1:, 0] = matrix[1:, 0] / L[0, 0]
    captured = np.sum(L[:, 0] ** 2)

    i = 1
    while captured < target_trace and i < n:
        diag_val = matrix[i, i] - np.dot(L[i, :i], L[i, :i])
        L[i, i] = np.sqrt(max(diag_val, 0.0))

        if i + 1 < n:
            column = (matrix[i + 1 :, i] - L[i + 1 :, :i] @ L[i, :i]) / L[i, i]
            L[i + 1 :, i] = column
            captured += L[i, i] ** 2 + np.sum(column**2)
        else:
            captured += L[i, i] ** 2

        i += 1

    return L[:, :i], i


def incomplete_cholesky_pivoted(
    matrix: np.ndarray, eta: float
) -> tuple[np.ndarray, int]:
    """Incomplete Cholesky factorization with diagonal pivoting.

    At each step, the remaining point with the largest residual variance
    is chosen as the next pivot, which typically reaches a target
    accuracy at a lower rank than the unpivoted variant.

    Parameters
    ----------
    matrix : (n, n) symmetric positive semi-definite matrix.
    eta : target relative trace tolerance in (0, 1); the factorization
        stops once ``trace(L @ L.T) >= (1 - eta) * trace(matrix)``.

    Returns
    -------
    L : (n, k) array such that ``matrix ~ L @ L.T``
    k : effective rank reached
    """
    n = matrix.shape[0]
    target_trace = (1.0 - eta) * np.trace(matrix)

    L = np.zeros((n, n))
    residual_diag = np.diag(matrix).copy()
    remaining = np.arange(n)
    captured = 0.0
    rank = 0

    while captured < target_trace and remaining.size > 0:
        pivot = remaining[np.argmax(residual_diag[remaining])]
        if residual_diag[pivot] <= 0:
            break  # numerical loss of positive semi-definiteness

        L[pivot, rank] = np.sqrt(residual_diag[pivot])
        remaining = remaining[remaining != pivot]

        if rank > 0:
            L[remaining, rank] = (
                matrix[remaining, pivot] - L[remaining, :rank] @ L[pivot, :rank]
            ) / L[pivot, rank]
        else:
            L[remaining, rank] = matrix[remaining, pivot] / L[pivot, rank]

        residual_diag[remaining] -= L[remaining, rank] ** 2
        captured += L[pivot, rank] ** 2 + np.sum(L[remaining, rank] ** 2)
        rank += 1

    return L[:, :rank], rank


def incomplete_cholesky_pivoted_rbf(
    x: np.ndarray, length_scale: float, amplitude: float, eta: float
) -> tuple[np.ndarray, int]:
    """Matrix-free pivoted incomplete Cholesky for an RBF kernel.

    Identical algorithm to :func:`incomplete_cholesky_pivoted`, but never
    materializes the full ``(n, n)`` kernel matrix: each kernel column is
    computed on the fly from the raw points ``x``. This is what makes the
    factorization usable for large ``n`` where the dense kernel matrix
    would not fit in memory.

    Parameters
    ----------
    x : (n, d) training points.
    length_scale, amplitude : RBF kernel hyperparameters.
    eta : target relative trace tolerance in (0, 1).

    Returns
    -------
    L : (n, k) array such that ``K(x, x) ~ L @ L.T``
    k : effective rank reached
    """
    x = _as_2d(x)
    n = x.shape[0]

    residual_diag = np.full(n, amplitude**2, dtype=np.float64)
    target_trace = (1.0 - eta) * np.sum(residual_diag)

    L = np.zeros((n, n))
    remaining = np.arange(n)
    captured = 0.0
    rank = 0

    while captured < target_trace and remaining.size > 0:
        pivot = remaining[np.argmax(residual_diag[remaining])]
        if residual_diag[pivot] <= 0:
            break

        L[pivot, rank] = np.sqrt(residual_diag[pivot])
        remaining = remaining[remaining != pivot]

        pivot_column = rbf_covariance_column(x, pivot, length_scale, amplitude)
        if rank > 0:
            L[remaining, rank] = (
                pivot_column[remaining] - L[remaining, :rank] @ L[pivot, :rank]
            ) / L[pivot, rank]
        else:
            L[remaining, rank] = pivot_column[remaining] / L[pivot, rank]

        residual_diag[remaining] -= L[remaining, rank] ** 2
        captured += L[pivot, rank] ** 2 + np.sum(L[remaining, rank] ** 2)
        rank += 1

    return L[:, :rank], rank


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``L @ y = b`` for lower-triangular ``L``."""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``U @ x = y`` for upper-triangular ``U``."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def lower_triangular_inverse(L: np.ndarray) -> np.ndarray:
    """Invert a lower-triangular matrix ``L`` in O(n^3) without a
    general-purpose linear solve, by forward-substituting each column of
    the identity.
    """
    n = L.shape[0]
    inv = np.zeros((n, n))
    for i in range(n):
        inv[i, i] = 1.0 / L[i, i]
        for j in range(i):
            inv[i, j] = -np.dot(L[i, :i], inv[:i, j]) / L[i, i]
    return inv


def inverse_via_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive-definite matrix via its exact Cholesky
    factor: ``inv(A) = inv(L).T @ inv(L)`` where ``A = L @ L.T``.
    """
    L = cholesky_dense(matrix)
    inv_L = lower_triangular_inverse(L)
    return inv_L.T @ inv_L
