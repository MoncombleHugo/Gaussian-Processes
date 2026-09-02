"""Fast linear algebra for low-rank-plus-diagonal matrices via the
Woodbury identity.

Given a low-rank factor ``L`` (n, k) with ``k << n`` approximating a
kernel matrix ``K ~ L @ L.T``, and observation noise ``sigma``, this
module solves systems involving ``(K + sigma^2 I)`` in ``O(n * k^2)``
instead of the ``O(n^3)`` a direct dense solve would cost:

    (L L^T + sigma^2 I)^-1 = (1/sigma^2) * (I - L (I + L^T L / sigma^2)^-1 L^T / sigma^2)
"""

from __future__ import annotations

import numpy as np


def woodbury_solve(L: np.ndarray, sigma: float, rhs: np.ndarray) -> np.ndarray:
    """Apply ``(L @ L.T + sigma^2 * I)^-1`` to ``rhs`` without forming the
    full ``(n, n)`` inverse.

    Parameters
    ----------
    L : (n, k) low-rank factor.
    sigma : observation noise standard deviation.
    rhs : (n,) or (n, m) right-hand side.

    Returns
    -------
    Array of the same shape as ``rhs``.
    """
    inv_sigma_sq = 1.0 / sigma**2
    k = L.shape[1]

    projected_rhs = L.T @ rhs
    system_matrix = np.eye(k) + inv_sigma_sq * (L.T @ L)
    correction = L @ np.linalg.solve(system_matrix, projected_rhs)

    return inv_sigma_sq * rhs - inv_sigma_sq**2 * correction


def woodbury_inverse(L: np.ndarray, sigma: float) -> np.ndarray:
    """Form the full dense inverse ``(L @ L.T + sigma^2 * I)^-1``.

    Prefer :func:`woodbury_solve` whenever only a matrix-vector (or
    matrix-matrix) product is needed -- this function materializes an
    ``(n, n)`` array and should only be used when the full inverse is
    genuinely required.
    """
    n, k = L.shape
    inv_sigma_sq = 1.0 / sigma**2

    system_matrix = np.eye(k) + inv_sigma_sq * (L.T @ L)
    middle_term = L @ np.linalg.solve(system_matrix, L.T)

    return inv_sigma_sq * (np.eye(n) - inv_sigma_sq * middle_term)
