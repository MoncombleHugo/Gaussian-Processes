"""Minimal end-to-end example: fit a GP on noisy 1D sine observations
and plot the predictive mean and 95% interval.

Run with:  python examples/demo_1d_regression.py
"""

import numpy as np

from gp_lowrank import GaussianProcessRegressor
from gp_lowrank.plotting import plot_1d_regression


def main():
    rng = np.random.default_rng(42)

    X_train = np.array([-4, -3, -2, 0, 2, 3, 4], dtype=float)
    y_train = np.sin(X_train) + 0.1 * rng.standard_normal(len(X_train))
    X_test = np.linspace(-5, 5, 200)

    gp = GaussianProcessRegressor(length_scale=1.5, amplitude=1.0, noise_std=0.1)
    gp.fit(X_train, y_train)
    y_mean, y_std = gp.predict(X_test)

    fig = plot_1d_regression(
        X_train,
        y_train,
        X_test,
        y_true=np.sin(X_test),
        y_mean=y_mean,
        y_std=y_std,
        title="1D GP regression (exact)",
    )
    fig.savefig("demo_1d_regression.png", dpi=150, bbox_inches="tight")
    print("Saved demo_1d_regression.png")


if __name__ == "__main__":
    main()
