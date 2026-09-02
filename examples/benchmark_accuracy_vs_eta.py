"""Reproduce the "near-identical accuracy vs. exact Cholesky baseline"
comparison on synthetic data, and the runtime saved for each eta.

Run with:  python examples/benchmark_accuracy_vs_eta.py
"""

import numpy as np

from gp_lowrank.benchmarks import benchmark_accuracy_vs_eta, generate_synthetic_points
from gp_lowrank.plotting import plot_accuracy_vs_eta


def main():
    rng = np.random.default_rng(0)

    n_train, n_test, d = 1500, 300, 8
    length_scale, amplitude, noise_std = 2.0, 150.0, 5.0

    X_train = generate_synthetic_points(n_train, d=d, seed=1)
    X_test = generate_synthetic_points(n_test, d=d, seed=2)
    true_weights = rng.standard_normal(d)
    y_train = X_train @ true_weights + noise_std * rng.standard_normal(n_train)

    etas = np.logspace(-6, -1, 12)
    result = benchmark_accuracy_vs_eta(
        X_train, y_train, X_test, length_scale, amplitude, noise_std, etas
    )

    fig_mse, fig_time = plot_accuracy_vs_eta(result)
    fig_mse.savefig("accuracy_vs_eta.png", dpi=150, bbox_inches="tight")
    fig_time.savefig("runtime_vs_eta.png", dpi=150, bbox_inches="tight")

    speedup = result.exact_fit_time_s / min(result.fit_time_s)
    print(f"Exact GP fit time:        {result.exact_fit_time_s:.4f} s")
    print(f"Fastest low-rank fit time: {min(result.fit_time_s):.4f} s")
    print(f"Speedup: {speedup:.1f}x")
    print("Saved accuracy_vs_eta.png, runtime_vs_eta.png")


if __name__ == "__main__":
    main()
