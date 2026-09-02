"""
gp_lowrank
==========

Scalable Gaussian Process emulation via matrix-free, low-rank kernel
approximations (pivoted incomplete Cholesky + Woodbury identity).

Public API
----------
- kernels: RBF covariance functions
- cholesky: exact and incomplete Cholesky factorizations
- woodbury: fast linear solves against low-rank + diagonal matrices
- regression: Gaussian Process regressor (exact and low-rank variants)
- hyperparameters: marginal-likelihood estimation for kernel hyperparameters
- benchmarks: synthetic-data utilities for runtime/accuracy studies
- plotting: reusable plotting helpers for the above
"""

from .regression import GaussianProcessRegressor

__all__ = ["GaussianProcessRegressor"]
__version__ = "0.2.0"
