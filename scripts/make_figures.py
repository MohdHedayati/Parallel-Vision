"""Build every figure and the summary tables from results/*.csv.

    python scripts/make_figures.py results

Writes: fig_scaling.png, fig_dispatch.png, fig_crossover.png, fig_opencv.png, fig_edges.png
(if sample images exist) and summary.json (numbers quoted in the report).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
C = {"seq": "#6b7280", "omp": "#2563eb", "mpi": "#d97706", "hyb": "#059669", "cv": "#7c3aed", "bad": "#dc2626"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
summary = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RES / name, dpi=200)
    plt.close(fig)
    print("wrote", RES / name)


# ---------------------------------------------------------------- scaling (seq / OpenMP / MPI / hybrid)
bench = pd.read_csv(RES / "bench.csv")
mpi = pd.read_csv(RES / "mpi.csv")
hyb = pd.read_csv(RES / "mpi_hybrid.csv") if (RES / "mpi_hybrid.csv").exists() else None
threads = int(bench["threads"].iloc[0])
summary["threads"] = threads

big = 4096
seq_big = float(bench.loc[bench["size"] == big, "seq_ms"].iloc[0])
rows = [("Sequential", seq_big, C["seq"], 1),
        (f"OpenMP\n{threads} threads", float(bench.loc[bench["size"] == big, "omp_ms"].iloc[0]), C["omp"], threads)]
for p in (2, 4, 8):
    rows.append((f"MPI\n{p} ranks", float(mpi[(mpi.ranks == p) & (mpi["size"] == big)]["mpi_ms"].iloc[0]), C["mpi"], p))
if hyb is not None:
    for (p, t), g in hyb.groupby(["ranks", "threads_per_rank"]):
        rows.append((f"Hybrid\n{p} ranks x\n{t} thr", float(g[g["size"] == big]["mpi_ms"].iloc[0]), C["hyb"], int(p * t)))
summary["scaling_4096"] = [{"config": r[0].replace("\n", " "), "ms": r[1], "speedup": seq_big / r[1],
                            "workers": r[3], "efficiency": seq_big / r[1] / r[3]} for r in rows]
# best speedup at every size, over all parallel configurations (for the headline figure)
best = []
for n in sorted(bench["size"].unique()):
    cands = {"OpenMP": float(bench.loc[bench["size"] == n, "omp_ms"].iloc[0])}
    for p in (2, 4, 8):
        cands[f"MPI {p}"] = float(mpi[(mpi.ranks == p) & (mpi["size"] == n)]["mpi_ms"].iloc[0]) if n in set(mpi["size"]) else 1e30
    s = float(bench.loc[bench["size"] == n, "seq_ms"].iloc[0])
    k = min(cands, key=cands.get)
    best.append({"size": int(n), "seq_ms": s, "best": k, "best_ms": cands[k], "speedup": s / cands[k]})
summary["best_by_size"] = best

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
a = ax[0]
a.loglog(bench["pixels"], bench["seq_ms"], "o-", color=C["seq"], label="Sequential")
a.loglog(bench["pixels"], bench["omp_ms"], "s-", color=C["omp"], label=f"OpenMP ({threads} thr)")
m4 = mpi[mpi.ranks == 4]
a.loglog(m4["pixels"], m4["mpi_ms"], "^-", color=C["mpi"], label="MPI (4 ranks)")
if hyb is not None:
    h = hyb[(hyb.ranks == 4)]
    a.loglog(h["pixels"], h["mpi_ms"], "D-", color=C["hyb"], label="Hybrid (4 ranks x 2 thr)")
a.set(xlabel="image size (pixels)", ylabel="Canny time (ms)", title="Execution time")
a.legend(frameon=False)
b = ax[1]
names = [r[0] for r in rows]
sp = [seq_big / r[1] for r in rows]
bars = b.bar(range(len(rows)), sp, color=[r[2] for r in rows])
for i, v in enumerate(sp):
    b.text(i, v + 0.04, f"{v:.2f}x", ha="center", fontsize=8)
b.axhline(1, color="k", lw=0.8, ls="--")
b.set_xticks(range(len(rows)))
b.set_xticklabels(names, fontsize=7)
b.set(ylabel="speedup vs sequential", title=f"Speedup on a {big}x{big} image")
save(fig, "fig_scaling.png")

# ---------------------------------------------------------------- architecture diagram
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def box(a, x, y, w, h, text, fc, ec="#374151", fs=9, bold=False):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


fig, a = plt.subplots(figsize=(11, 4.6))
a.set_xlim(0, 11); a.set_ylim(0, 4.6); a.axis("off"); a.grid(False)
box(a, 0.2, 3.6, 3.0, 0.7, "Your program / pv CLI", "#f3f4f6", bold=True)
box(a, 0.2, 2.45, 3.0, 0.8, "Public API (pv.hpp)\ncanny · sobel · blur · gray · threshold", "#dbeafe", bold=True)
box(a, 0.2, 0.3, 3.0, 1.8, "Dispatcher\nBackend::Auto →\nchoose_backend(op, W, H)\n• heuristic switch points\n• calibrated on this machine", "#fef3c7", bold=True)
box(a, 4.4, 3.55, 2.9, 0.75, "Row kernels (written once)\nprocess rows [y0, y1)", "#e5e7eb", bold=True)
box(a, 4.4, 2.4, 2.9, 0.8, "Sequential\nplain loop", "#e5e7eb")
box(a, 4.4, 1.3, 2.9, 0.8, "OpenMP\nparallel for over rows", "#dbeafe")
box(a, 4.4, 0.2, 2.9, 0.8, "MPI (+ hybrid OpenMP)\nstrips + halo exchange", "#fde68a")
box(a, 8.2, 1.3, 2.6, 0.8, "CUDA\n(planned, same interface)", "#f3f4f6", ec="#9ca3af")
box(a, 8.2, 2.4, 2.6, 0.8, "identical output\n(tested bit-for-bit)", "#d1fae5", ec="#059669")
for (x0, y0, x1, y1) in [(1.7, 3.6, 1.7, 3.25), (1.7, 2.45, 1.7, 2.1), (3.2, 1.2, 4.4, 2.8), (3.2, 1.2, 4.4, 1.7), (3.2, 1.0, 4.4, 0.6),
                         (5.85, 3.55, 5.85, 3.2), (7.3, 2.8, 8.2, 2.8), (7.3, 0.6, 8.2, 1.5)]:
    a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2))
save(fig, "fig_architecture.png")

# ---------------------------------------------------------------- dispatcher
d = pd.read_csv(RES / "dispatch.csv")
d["best"] = d[["seq_ms", "omp_ms"]].min(axis=1)


def chosen(col):
    return np.where(d[col] == "sequential", d.seq_ms, d.omp_ms)


strategies = {"Always\nsequential": d.seq_ms.values, "Always\nOpenMP": d.omp_ms.values,
              "pv\nheuristic": chosen("heuristic"), "pv\ncalibrated": chosen("calibrated")}
tab = []
for k, v in strategies.items():
    r = v / d.best.values
    tab.append({"strategy": k.replace("\n", " "), "total_vs_oracle_pct": float(v.sum() / d.best.sum() * 100),
                "geomean": float(np.exp(np.log(r).mean())), "worst": float(r.max()),
                "within10": float((r <= 1.10).mean() * 100)})
summary["dispatch"] = tab
summary["dispatch_cases"] = int(len(d))
small = d[d.pixels < 100000]
bigd = d[d.pixels >= 1_000_000]
summary["small_always_omp_slowdown"] = float(small.omp_ms.sum() / small.seq_ms.sum())
summary["large_always_seq_slowdown"] = float(bigd.seq_ms.sum() / bigd.omp_ms.sum())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
cols = [C["seq"], C["bad"], C["omp"], C["hyb"]]
g = [t["geomean"] for t in tab]
ax[0].bar(range(4), g, color=cols)
for i, v in enumerate(g):
    ax[0].text(i, v * 1.03, f"{v:.2f}x", ha="center")
ax[0].axhline(1, color="k", lw=0.8, ls="--")
ax[0].set_xticks(range(4))
ax[0].set_xticklabels(list(strategies), fontsize=9)
ax[0].set(ylabel="time vs. best possible (geometric mean)", title=f"Average slowdown over {len(d)} test cases")
w = [t["worst"] for t in tab]
ax[1].bar(range(4), w, color=cols)
ax[1].set_yscale("log")
ax[1].set_ylim(1, 200)
for i, v in enumerate(w):
    ax[1].text(i, v * 1.1, f"{v:.1f}x", ha="center")
ax[1].set_xticks(range(4))
ax[1].set_xticklabels(list(strategies), fontsize=9)
ax[1].set(ylabel="slowdown (log scale)", title="Worst single case")
save(fig, "fig_dispatch.png")

ops = list(d["op"].unique())
fig, axes = plt.subplots(1, len(ops), figsize=(3.4 * len(ops), 3.5))
for a, op in zip(axes, ops):
    q = d[d["op"] == op].sort_values("pixels")
    a.loglog(q["pixels"], q["seq_ms"], "o-", color=C["seq"], ms=3, label="sequential")
    a.loglog(q["pixels"], q["omp_ms"], "s-", color=C["omp"], ms=3, label="OpenMP")
    a.set(title=op, xlabel="pixels")
axes[0].set_ylabel("time (ms)")
axes[0].legend(frameon=False, fontsize=8)
save(fig, "fig_crossover.png")

# ---------------------------------------------------------------- vs OpenCV
v = pd.read_csv(RES / "vs_opencv.csv")
r = v[v["size"] == big].iloc[0]
summary["opencv_4096"] = {k: float(r[k]) for k in ["pv_seq_ms", "pv_omp_ms", "pv_auto_ms", "cv_1thread_ms", "cv_nthreads_ms"]}
fig, a = plt.subplots(figsize=(7, 4))
a.loglog(v["pixels"], v["pv_auto_ms"], "o-", color=C["omp"], label="pv (auto dispatch)")
a.loglog(v["pixels"], v["cv_1thread_ms"], "s--", color=C["cv"], alpha=0.6, label="OpenCV 1 thread")
a.loglog(v["pixels"], v["cv_nthreads_ms"], "s-", color=C["cv"], label=f"OpenCV {threads} threads")
a.set(xlabel="image size (pixels)", ylabel="Canny time (ms)", title="Parallel-Vision vs OpenCV (same pipeline)")
a.legend(frameon=False)
save(fig, "fig_opencv.png")

# ---------------------------------------------------------------- logs -> summary
log = RES / "vs_opencv.log"
if log.exists():
    for line in log.read_text().splitlines():
        if "IoU" in line:
            summary["iou"] = float(line.split("IoU=")[1].split()[0])
            summary["edge_px_pv"] = int(line.split("pv=")[1].split()[0])
            summary["edge_px_cv"] = int(line.split("opencv=")[1].split()[0])
log = RES / "dispatch.log"
if log.exists():
    sw = {}
    for line in log.read_text().splitlines():
        if "switch point" in line:
            name = line.split()[0]
            sw[name] = int(line.split(":")[1].split()[0])
    summary["switch_points"] = sw

# ---------------------------------------------------------------- edge-map picture
inp, out = RES / "sample_input.jpg", RES / "sample_canny.png"
if inp.exists() and out.exists():
    a_img, b_img = plt.imread(inp), plt.imread(out)
    step = max(1, a_img.shape[1] // 1100)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].imshow(a_img[::step, ::step])
    ax[0].set_title("Input")
    ax[1].imshow(b_img[::step, ::step], cmap="gray")
    ax[1].set_title("Canny edges (pv, auto backend)")
    for x in ax:
        x.axis("off")
        x.grid(False)
    save(fig, "fig_edges.png")

(RES / "summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", RES / "summary.json")
