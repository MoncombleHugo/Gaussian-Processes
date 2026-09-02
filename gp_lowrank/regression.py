"""Gaussian Process regression, exact and low-rank (matrix-free)."""

from __future__ import annotations

import time

import numpy as np

from .cholesky import incomplete_cholesky_pivoted_rbf
from .kernels import _as_2d, rbf_covariance
from .woodbury import woodbury_solve


class GaussianProcessRegressor:
    """GP regression with an RBF kernel and Gaussian observation noise.

    Two fitting strategies are available:

    - ``method="exact"``: dense Cholesky-based solve, O(n^3) in the
      number of training points. Used as the accuracy reference.
    - ``method="low_rank"``: pivoted incomplete Cholesky computed
      directly from the raw training points (never materializes the
      full kernel matrix) combined with the Woodbury identity for the
      noisy solve. Cost scales with the effective rank ``k`` rather
      than ``n``, which is what makes large training sets tractable.

    Parameters
    ----------
    length_scale, amplitude : RBF kernel hyperparameters.
    noise_std : observation noise standard deviation (sigma).
    method : {"exact", "low_rank"}.
    eta : required when ``method="low_rank"``. Target relative trace
        tolerance in (0, 1) for the incomplete Cholesky factorization;
        smaller values keep more rank and are more accurate but slower.
    """

    def __init__(
        self,
        length_scale: float,
        amplitude: float,
        noise_std: float,
        method: str = "exact",
        eta: float | None = None,
    ):
        if method not in ("exact", "low_rank"):
            raise ValueError("method must be 'exact' or 'low_rank'")
        if method == "low_rank" and eta is None:
            raise ValueError("eta is required when method='low_rank'")

        self.length_scale = length_scale
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.method = method
        self.eta = eta

        self.X_train_: np.ndarray | None = None
        self.rank_: int | None = None
        self.fit_time_s: float | None = None

        self._L: np.ndarray | None = None  # low-rank factor (low_rank only)
        self._alpha: np.ndarray | None = None  # (K + sigma^2 I)^-1 y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GaussianProcessRegressor":
        X_train = _as_2d(X_train)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        start = time.perf_counter()

        if self.method == "exact":
            K = rbf_covariance(X_train, X_train, self.length_scale, self.amplitude)
            K_noisy = K + self.noise_std**2 * np.eye(len(X_train))
            self._alpha = np.linalg.solve(K_noisy, y_train)
            self.rank_ = len(X_train)
        else:
            self._L, self.rank_ = incomplete_cholesky_pivoted_rbf(
                X_train, self.length_scale, self.amplitude, self.eta
            )
            self._alpha = woodbury_solve(self._L, self.noise_std, y_train)

        self.fit_time_s = time.perf_counter() - start
        self.X_train_ = X_train
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the predictive mean and standard deviation at ``X_test``."""
        if self.X_train_ is None:
            raise RuntimeError("call fit() before predict()")

        X_test = _as_2d(X_test)
        K_test_train = rbf_covariance(
            X_test, self.X_train_, self.length_scale, self.amplitude
        )
        mean = K_test_train @ self._alpha

        K_test_test_diag = np.full(len(X_test), self.amplitude**2)

        if self.method == "exact":
            K_train_test = rbf_covariance(
                self.X_train_, X_test, self.length_scale, self.amplitude
            )
            K = rbf_covariance(
                self.X_train_, self.X_train_, self.length_scale, self.amplitude
            )
            K_noisy = K + self.noise_std**2 * np.eye(len(self.X_train_))
            correction = np.einsum(
                "ij,ji->i", K_test_train, np.linalg.solve(K_noisy, K_train_test)
            )
        else:
            K_train_test = K_test_train.T
            inv_applied = woodbury_solve(self._L, self.noise_std, K_train_test)
            correction = np.einsum("ij,ji->i", K_test_train, inv_applied)

        variance = np.clip(K_test_test_diag - correction, 0.0, None)
        return mean, np.sqrt(variance)
