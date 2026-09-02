"""Reproduce the rank/runtime study behind the "~100x speedup" claim,
on synthetic data.

Consolidates what used to be three near-duplicate scripts
(temps_et_rangs.py, _v2.py, _v3.py) into one parametrized benchmark.

Run with:  python examples/benchmark_incomplete_cholesky.py
"""

from gp_lowrank.benchmarks import benchmark_rank_and_runtime_vs_size
from gp_lowrank.plotting import plot_rank_and_runtime_vs_size


def main():
    sizes = [200, 500, 1000, 2000, 3000, 5000]

    result = benchmark_rank_and_runtime_vs_size(
        sizes=sizes,
        length_scale=2.0,
        amplitude=200.0,
        eta=1e-3,
        d=8,
        repeats=3,
    )

    fig_time, fig_rank = plot_rank_and_runtime_vs_size(result)
    fig_time.savefig("benchmark_runtime.png", dpi=150, bbox_inches="tight")
    fig_rank.savefig("benchmark_rank.png", dpi=150, bbox_inches="tight")

    print(f"{'n':>6} | {'no-pivot (ms)':>14} | {'pivot (ms)':>11} | {'matrix-free (ms)':>17} | rank")
    for i, n in enumerate(sizes):
        print(
            f"{n:6d} | {result.times_no_pivot_ms[i]:14.2f} | "
            f"{result.times_pivot_ms[i]:11.2f} | "
            f"{result.times_pivot_matrixfree_ms[i]:17.2f} | "
            f"{result.ranks_pivot_matrixfree[i]}"
        )
    print("Saved benchmark_runtime.png, benchmark_rank.png")


if __name__ == "__main__":
    main()
