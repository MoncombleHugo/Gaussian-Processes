import numpy as np
import pytest

from gp_lowrank.cholesky import (
    cholesky_dense,
    incomplete_cholesky_pivoted,
    incomplete_cholesky_pivoted_rbf,
)
from gp_lowrank.kernels import rbf_covariance
from gp_lowrank.regression import GaussianProcessRegressor
from gp_lowrank.woodbury import woodbury_inverse, woodbury_solve


@pytest.fixture
def spd_matrix():
    rng = np.random.default_rng(0)
    A = rng.random((30, 30))
    return A @ A.T + 30 * np.eye(30)  # well-conditioned SPD


def test_cholesky_dense_reconstructs_matrix(spd_matrix):
    L = cholesky_dense(spd_matrix)
    np.testing.assert_allclose(L @ L.T, spd_matrix, rtol=1e-8, atol=1e-8)


def test_incomplete_cholesky_pivoted_low_eta_is_near_exact(spd_matrix):
    L, rank = incomplete_cholesky_pivoted(spd_matrix, eta=1e-10)
    assert rank <= spd_matrix.shape[0]
    reconstruction_error = np.max(np.abs(spd_matrix - L @ L.T))
    assert reconstruction_error < 1e-6


def test_incomplete_cholesky_matrixfree_matches_dense_pivoted():
    rng = np.random.default_rng(1)
    X = rng.random((50, 3))
    length_scale, amplitude, eta = 0.8, 2.0, 1e-8

    K = rbf_covariance(X, X, length_scale, amplitude)
    L_dense, rank_dense = incomplete_cholesky_pivoted(K, eta)
    L_free, rank_free = incomplete_cholesky_pivoted_rbf(X, length_scale, amplitude, eta)

    assert rank_dense == rank_free
    np.testing.assert_allclose(L_dense @ L_dense.T, L_free @ L_free.T, atol=1e-6)


def test_woodbury_solve_matches_dense_inverse():
    rng = np.random.default_rng(2)
    L = rng.random((20, 5))
    sigma = 0.7
    rhs = rng.random(20)

    dense_inverse = woodbury_inverse(L, sigma)
    expected = dense_inverse @ rhs
    actual = woodbury_solve(L, sigma, rhs)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_low_rank_gp_matches_exact_gp_at_low_eta():
    rng = np.random.default_rng(3)
    X_train = rng.random((200, 4))
    y_train = np.sin(X_train.sum(axis=1)) + 0.05 * rng.standard_normal(200)
    X_test = rng.random((50, 4))

    exact = GaussianProcessRegressor(length_scale=1.0, amplitude=1.0, noise_std=0.2, method="exact")
    exact.fit(X_train, y_train)
    mean_exact, std_exact = exact.predict(X_test)

    low_rank = GaussianProcessRegressor(
        length_scale=1.0, amplitude=1.0, noise_std=0.2, method="low_rank", eta=1e-10
    )
    low_rank.fit(X_train, y_train)
    mean_low_rank, std_low_rank = low_rank.predict(X_test)

    np.testing.assert_allclose(mean_exact, mean_low_rank, atol=1e-4)
    np.testing.assert_allclose(std_exact, std_low_rank, atol=1e-4)
