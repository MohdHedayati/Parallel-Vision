"""Graphs for the live demo, made from the CSV files and images produced by scripts/demo.ps1.

    python scripts/make_demo_figures.py results/demo
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/demo")
C = {"seq": "#6b7280", "omp": "#2563eb", "mpi": "#d97706", "ok": "#059669", "bad": "#dc2626"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})


def downscale(im, step, mode="mean"):
    """Shrink by an integer factor with block max/mean pooling (keeps thin lines visible)."""
    if step <= 1:
        return im
    h, w = im.shape[0] // step * step, im.shape[1] // step * step
    im = im[:h, :w]
    shape = (h // step, step, w // step, step) + ((im.shape[2],) if im.ndim == 3 else ())
    blocks = im.reshape(shape)
    out = blocks.max(axis=(1, 3)) if mode == "max" else blocks.mean(axis=(1, 3))
    return out.astype(im.dtype)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# ---- 1. every algorithm on the real picture --------------------------------------------------
imgs = OUT / "images"
order = ["input", "gray", "blur", "median", "sobel", "prewitt", "laplacian", "canny", "threshold", "equalize", "erode", "dilate"]
titles = {"input": "Input", "gray": "Grayscale", "blur": "Gaussian blur", "median": "Median 3x3", "sobel": "Sobel",
          "prewitt": "Prewitt", "laplacian": "Laplacian", "canny": "Canny", "threshold": "Otsu threshold",
          "equalize": "Histogram equalisation", "erode": "Erosion", "dilate": "Dilation"}
if (imgs / "input.jpg").exists():
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.2))
    for ax, name in zip(axes.flat, order):
        f = imgs / ("input.jpg" if name == "input" else f"{name}.png")
        if f.exists():
            im = plt.imread(f)
            step = max(1, im.shape[1] // 520)
            # thin 1-px edges vanish if we just skip pixels, so edge maps are max-pooled instead
            mode = "max" if name in ("sobel", "prewitt", "laplacian", "canny") else "mean"
            ax.imshow(downscale(im, step, mode), cmap=None if name == "input" else "gray")
        ax.set_title(titles[name], fontsize=10)
        ax.axis("off")
        ax.grid(False)
    save(fig, "fig_demo_montage.png")

# ---- 2. speedup of every operation ------------------------------------------------------------
if (OUT / "demo_ops.csv").exists():
    d = pd.read_csv(OUT / "demo_ops.csv").sort_values("speedup")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(d["op"], d["speedup"], color=C["omp"])
    for b, v in zip(bars, d["speedup"]):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}x", va="center", fontsize=9)
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.set(xlabel="speedup of OpenMP over sequential (2048 x 2048 image)", title="Speedup of every operation")
    ax.set_xlim(0, d["speedup"].max() * 1.15)
    save(fig, "fig_demo_speedup.png")

# ---- 3. dispatcher decisions ------------------------------------------------------------------
if (OUT / "demo_dispatch.csv").exists():
    d = pd.read_csv(OUT / "demo_dispatch.csv")
    ops = list(d["op"].unique())
    fig, axes = plt.subplots(1, len(ops), figsize=(3.6 * len(ops), 3.7))
    for ax, op in zip(axes, ops):
        q = d[d["op"] == op].sort_values("pixels")
        ax.loglog(q["pixels"], q["seq_ms"], "o-", color=C["seq"], ms=4, label="sequential")
        ax.loglog(q["pixels"], q["omp_ms"], "s-", color=C["omp"], ms=4, label="OpenMP")
        wh, wc = q[q["heuristic_ok"] == 0], q[q["calibrated_ok"] == 0]
        ax.scatter(wh["pixels"], wh[["seq_ms", "omp_ms"]].max(axis=1) * 1.5, marker="v", color="#f59e0b", zorder=3,
                   label="fixed heuristic chose wrong")
        ax.scatter(wc["pixels"], wc[["seq_ms", "omp_ms"]].max(axis=1) * 2.5, marker="v", color=C["bad"], zorder=3,
                   label="calibrated chose wrong")
        ax.set(title=op, xlabel="pixels")
    axes[0].set_ylabel("time (ms)")
    h, l = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig_demo_dispatch.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "fig_demo_dispatch.png")
    print(f"  heuristic correct: {int(d['heuristic_ok'].sum())}/{len(d)}   calibrated correct: {int(d['calibrated_ok'].sum())}/{len(d)}")

# ---- 4. MPI on 1, 2, 4 processes ---------------------------------------------------------------
if (OUT / "mpi_demo.csv").exists():
    d = pd.read_csv(OUT / "mpi_demo.csv")
    d = d[d["size"] == d["size"].max()].sort_values("ranks")
    base = float(d[d.ranks == 1]["mpi_ms"].iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].bar(d["ranks"].astype(str), d["mpi_ms"], color=C["mpi"])
    for i, v in enumerate(d["mpi_ms"]):
        ax[0].text(i, v + base * 0.01, f"{v:.0f} ms", ha="center", fontsize=9)
    ax[0].set(xlabel="MPI processes", ylabel="Canny time (ms)", title=f"MPI, {int(d['size'].iloc[0])}x{int(d['size'].iloc[0])} image")
    sp = base / d["mpi_ms"]
    ax[1].bar(d["ranks"].astype(str), sp, color=C["ok"])
    for i, v in enumerate(sp):
        ax[1].text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9)
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set(xlabel="MPI processes", ylabel="speedup vs 1 process", title="Speedup (halo-exchange strips)")
    save(fig, "fig_demo_mpi.png")
