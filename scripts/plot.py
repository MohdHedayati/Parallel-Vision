"""Plots for the report.

  build\\bin\\pv_bench.exe 5 > results\\bench.csv
  build\\bin\\pv_dispatch_report.exe > results\\dispatch.csv
  python scripts/plot.py results/bench.csv results/dispatch.csv

Writes PNGs next to the CSVs.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_bench(csv: Path):
    df = pd.read_csv(csv)
    threads = int(df["threads"].iloc[0])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].loglog(df["pixels"], df["seq_ms"], "o-", label="sequential")
    ax[0].loglog(df["pixels"], df["omp_ms"], "s-", label=f"OpenMP ({threads} threads)")
    ax[0].set(xlabel="pixels", ylabel="Canny time (ms)", title="Execution time")
    ax[0].legend()

    ax[1].semilogx(df["pixels"], df["speedup"], "o-")
    ax[1].axhline(1, color="gray", ls="--", label="break-even")
    ax[1].axhline(threads, color="red", ls=":", label=f"ideal ({threads}x)")
    ax[1].set(xlabel="pixels", ylabel="speedup", title="Speedup = T_seq / T_omp")
    ax[1].legend()

    ax[2].semilogx(df["pixels"], df["efficiency"], "o-")
    ax[2].axhline(1, color="red", ls=":")
    ax[2].set(xlabel="pixels", ylabel="efficiency", title="Efficiency = speedup / threads", ylim=(0, 1.05))

    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = csv.with_name("bench.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_dispatch(csv: Path):
    df = pd.read_csv(csv)
    ops = list(df["op"].unique())
    fig, axes = plt.subplots(1, len(ops), figsize=(4 * len(ops), 3.8), sharey=False)
    for a, op in zip(axes, ops):
        d = df[df["op"] == op].sort_values("pixels")
        a.loglog(d["pixels"], d["seq_ms"], "o-", label="sequential")
        a.loglog(d["pixels"], d["omp_ms"], "s-", label="OpenMP")
        wrong_h = d[d["heuristic"] != d["oracle"]]
        wrong_c = d[d["calibrated"] != d["oracle"]]
        a.scatter(wrong_h["pixels"], wrong_h[["seq_ms", "omp_ms"]].max(axis=1) * 1.4,
                  marker="v", color="orange", label="heuristic wrong", zorder=3)
        a.scatter(wrong_c["pixels"], wrong_c[["seq_ms", "omp_ms"]].max(axis=1) * 2.0,
                  marker="v", color="red", label="calibrated wrong", zorder=3)
        a.set(title=op, xlabel="pixels")
        a.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("time (ms)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = csv.with_name("dispatch.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

    for name in ("heuristic", "calibrated"):
        acc = (df[name] == df["oracle"]).mean() * 100
        print(f"{name:10s} exact-match with oracle: {acc:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    plot_bench(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        plot_dispatch(Path(sys.argv[2]))
