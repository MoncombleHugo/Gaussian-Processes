"""Marginal-likelihood-based fitting of RBF kernel hyperparameters.

The (negative) log marginal likelihood is evaluated approximately, using
the same matrix-free pivoted incomplete Cholesky + Woodbury machinery as
the regressor, so hyperparameter search scales the same way prediction
does. Hyperparameters are optimized in log-space to keep them positive
without constrained optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .cholesky import incomplete_cholesky_pivoted_rbf


def neg_log_marginal_likelihood(
    log_theta: np.ndarray, X: np.ndarray, y: np.ndarray, eta: float
) -> float:
    """Approximate negative log marginal likelihood.

    ``log_theta = [log_amplitude, log_noise_std, log_length_scale]``.
    """
    log_amplitude, log_noise_std, log_length_scale = log_theta
    amplitude = np.exp(log_amplitude)
    noise_std = np.exp(log_noise_std)
    length_scale = np.exp(log_length_scale)
    n = X.shape[0]

    L, rank = incomplete_cholesky_pivoted_rbf(X, length_scale, amplitude, eta)

    from .woodbury import woodbury_solve

    alpha = woodbury_solve(L, noise_std, y)
    quadratic_term = float(y @ alpha)

    LtL = L.T @ L
    M = np.eye(rank) + LtL / noise_std**2
    try:
        R = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return 1e30  # heavily penalize numerically invalid regions

    log_det_M = 2.0 * np.sum(np.log(np.diag(R)))
    log_det = 2 * n * np.log(noise_std) + log_det_M

    log_marginal_likelihood = (
        -0.5 * quadratic_term - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
    )
    return -log_marginal_likelihood


def _numerical_gradient(
    log_theta: np.ndarray, X: np.ndarray, y: np.ndarray, eta: float, eps: float = 1e-3
) -> np.ndarray:
    grad = np.zeros_like(log_theta)
    f0 = neg_log_marginal_likelihood(log_theta, X, y, eta)
    for i in range(len(log_theta)):
        perturbed = log_theta.copy()
        perturbed[i] += eps
        grad[i] = (neg_log_marginal_likelihood(perturbed, X, y, eta) - f0) / eps
    return grad


@dataclass
class FitResult:
    amplitude: float
    noise_std: float
    length_scale: float
    n_iterations: int
    loss_history: list[float] = field(default_factory=list)


def fit_by_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    eta: float,
    init_log_theta: np.ndarray,
    learning_rate: float,
    max_iter: int = 50,
    grad_norm_tol: float = 1e-3,
    max_grad_norm: float = 1e3,
) -> FitResult:
    """Fit (amplitude, noise_std, length_scale) by numerical-gradient
    descent on the approximate negative log marginal likelihood.
    """
    log_theta = np.array(init_log_theta, dtype=float)
    loss_history = []

    for iteration in range(max_iter):
        loss = neg_log_marginal_likelihood(log_theta, X, y, eta)
        loss_history.append(loss)

        grad = _numerical_gradient(log_theta, X, y, eta)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_grad_norm:
            grad *= max_grad_norm / grad_norm

        log_theta = log_theta - learning_rate * grad
        log_theta = np.clip(log_theta, np.log(1e-3), np.log(1e3))

        if grad_norm < grad_norm_tol:
            break

    amplitude, noise_std, length_scale = np.exp(log_theta)
    return FitResult(amplitude, noise_std, length_scale, iteration + 1, loss_history)


def fit_by_lbfgs(
    X: np.ndarray,
    y: np.ndarray,
    eta: float,
    noise_std: float,
    init_log_amplitude: float = 0.0,
    init_log_length_scale: float = 0.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    max_iter: int = 50,
) -> FitResult:
    """Fit (amplitude, length_scale) with fixed noise via L-BFGS-B.

    Fixing the noise level is a common simplification when it is known
    a priori (e.g. from repeated simulator runs) -- it turns a 3D
    non-convex search into a better-behaved 2D one.
    """
    loss_history: list[float] = []

    def objective(log_theta_2d: np.ndarray) -> float:
        log_theta_3d = np.array(
            [log_theta_2d[0], np.log(noise_std), log_theta_2d[1]]
        )
        value = neg_log_marginal_likelihood(log_theta_3d, X, y, eta)
        loss_history.append(value)
        return value

    if bounds is None:
        bounds = ((np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e2)))

    x0 = np.array([init_log_amplitude, init_log_length_scale])
    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter}
    )

    amplitude, length_scale = np.exp(result.x)
    return FitResult(
        amplitude, noise_std, length_scale, result.nit, loss_history
    )
