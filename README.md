# Gaussian Processes — Scalable Bayesian Emulation of Uncertain Models

Gaussian Process (GP) emulators for stochastic simulators, built to make
Bayesian inference and uncertainty quantification tractable when the exact
GP machinery becomes too slow to use in practice.

## Motivation

Stochastic simulators are expensive to query and often need to be run
thousands of times for calibration, sensitivity analysis, or optimization.
GP surrogates ("emulators") let you replace the simulator with a cheap,
uncertainty-aware approximation — but the standard formulation relies on a
Cholesky decomposition of the kernel matrix, which scales as O(n³) in the
number of training points. That cost quickly becomes the bottleneck as the
design grows.

This project explores how far a matrix-free, low-rank approximation of the
kernel can push that limit without sacrificing predictive accuracy.

## Approach

- **Exact baseline**: standard GP regression with a Cholesky-based solve,
  used as the accuracy reference (`method="exact"`).
- **Scalable emulator**: pivoted incomplete Cholesky computed directly from
  the raw training points — the full `(n, n)` kernel matrix is never
  built — combined with the Woodbury identity for the noisy solve
  (`method="low_rank"`). Cost scales with the effective rank `k` reached
  rather than with `n`.
- **Bayesian workflow**: hyperparameter inference (kernel length-scale,
  amplitude, noise) by maximizing an approximate marginal likelihood, with
  predictive uncertainty propagated rather than point estimates alone.
- **Validation**: predictions from the low-rank emulator are benchmarked
  against the exact Cholesky-based GP on held-out points, in both mean
  accuracy and calibration of the predictive variance.

## Results

On synthetic benchmarks (`examples/`), the low-rank emulator reaches
**~60–100x** lower fit time than the exact Cholesky-based baseline while
staying **numerically indistinguishable** in prediction (mean and variance)
once `eta` is small enough — see `examples/benchmark_accuracy_vs_eta.py`
for the exact accuracy/speed trade-off curve. The original coursework used
a larger, project-specific stochastic-simulator dataset (not included
here); the reported project figures were a ~100x speedup with near-identical
accuracy against the exact baseline. Full details and derivations are in
[`docs/Rapport_scalable_GPs_S7.pdf`](docs/Rapport_scalable_GPs_S7.pdf).

## Repository structure

```
.
├── gp_lowrank/                        # the package
│   ├── kernels.py                     # RBF covariance (dense + matrix-free column)
│   ├── cholesky.py                    # exact + incomplete Cholesky (pivoted, matrix-free)
│   ├── woodbury.py                    # fast solves for low-rank + diagonal systems
│   ├── regression.py                  # GaussianProcessRegressor (exact / low_rank)
│   ├── hyperparameters.py             # marginal-likelihood hyperparameter fitting
│   ├── benchmarks.py                  # synthetic-data runtime/accuracy studies
│   └── plotting.py                    # reusable plotting helpers
├── examples/
│   ├── demo_1d_regression.py          # minimal end-to-end example
│   ├── benchmark_incomplete_cholesky.py  # runtime & rank vs. training set size
│   └── benchmark_accuracy_vs_eta.py   # accuracy/speed trade-off vs. exact GP
├── tests/
│   └── test_cholesky.py               # correctness checks (factorizations, Woodbury, GP)
├── docs/                              # original coursework report and slides
├── requirements.txt
└── README.md
```

## Stack

Python · NumPy · SciPy · Matplotlib

## Usage

```bash
git clone https://github.com/MoncombleHugo/Gaussian-Processes.git
cd Gaussian-Processes
pip install -r requirements.txt

# run the test suite
PYTHONPATH=. pytest tests/ -v

# run an example
PYTHONPATH=. python examples/demo_1d_regression.py
PYTHONPATH=. python examples/benchmark_incomplete_cholesky.py
PYTHONPATH=. python examples/benchmark_accuracy_vs_eta.py
```

## Notes

This was developed as a personal research project on scalable surrogate
modeling and Bayesian uncertainty quantification, motivated by
applications where simulators are too costly to query directly during
calibration or optimization loops. The code was restructured into this
package layout from an earlier collection of exploratory scripts; the
underlying algorithms (kernel functions, factorizations, Woodbury solves)
are unchanged and covered by `tests/`.
