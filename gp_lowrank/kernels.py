"""RBF (squared-exponential) covariance functions."""

from __future__ import annotations

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Treat a 1D array as a column of scalar (1D) points, i.e. shape
    (n,) -> (n, 1), rather than numpy's default atleast_2d behavior of
    (n,) -> (1, n).
    """
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def rbf_covariance(
    x1: np.ndarray, x2: np.ndarray, length_scale: float, amplitude: float
) -> np.ndarray:
    """Dense RBF covariance matrix between two point sets.

    k(x, x') = amplitude^2 * exp(-||x - x'||^2 / (2 * length_scale^2))

    Parameters
    ----------
    x1 : (n1, d) array
    x2 : (n2, d) array
    length_scale : float
    amplitude : float

    Returns
    -------
    (n1, n2) covariance matrix.
    """
    x1 = _as_2d(x1)
    x2 = _as_2d(x2)

    sq_norm1 = np.sum(x1**2, axis=1)[:, None]
    sq_norm2 = np.sum(x2**2, axis=1)[None, :]
    sq_dists = sq_norm1 + sq_norm2 - 2.0 * x1 @ x2.T
    np.clip(sq_dists, 0.0, None, out=sq_dists)  # guard against tiny negatives

    return amplitude**2 * np.exp(-0.5 * sq_dists / length_scale**2)


def rbf_covariance_column(
    x: np.ndarray, index: int, length_scale: float, amplitude: float
) -> np.ndarray:
    """Column ``index`` of ``rbf_covariance(x, x, ...)`` without forming
    the full matrix, i.e. ``K[:, index]``.

    Used by the matrix-free incomplete Cholesky factorization, where only
    a handful of kernel columns are ever needed.
    """
    x = _as_2d(x)
    diff = x - x[index]
    sq_dist = np.einsum("ij,ij->i", diff, diff)
    return amplitude**2 * np.exp(-0.5 * sq_dist / length_scale**2)
