"""Build every figure and the summary tables for the report from results/*.csv.

    python scripts/make_figures.py results

Inputs : bench.csv ops.csv mpi.csv mpi_hybrid.csv mpi_model.csv dispatch.csv calib_stability.csv
         vs_opencv.csv (+ the .log files, sample_input.jpg, sample_canny.png)
Outputs: fig_architecture fig_scaling fig_ops fig_dispatch fig_crossover fig_calibration fig_mpi_model
         fig_opencv fig_edges (PNG) and summary.json (every number quoted in the report).
Missing inputs are skipped, so the script also works on a partial run.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
C = {"seq": "#6b7280", "omp": "#2563eb", "mpi": "#d97706", "hyb": "#059669", "cv": "#7c3aed", "bad": "#dc2626", "amber": "#f59e0b"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
summary = {}


def have(name):
    return (RES / name).exists()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RES / name, dpi=200)
    plt.close(fig)
    print("wrote", RES / name)


def downscale(im, step, mode="mean"):
    if step <= 1:
        return im
    h, w = im.shape[0] // step * step, im.shape[1] // step * step
    im = im[:h, :w]
    shape = (h // step, step, w // step, step) + ((im.shape[2],) if im.ndim == 3 else ())
    blocks = im.reshape(shape)
    return (blocks.max(axis=(1, 3)) if mode == "max" else blocks.mean(axis=(1, 3))).astype(im.dtype)


# ================================================================ architecture
def box(a, x, y, w, h, text, fc, ec="#374151", fs=9, bold=False):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


fig, a = plt.subplots(figsize=(11, 4.8))
a.set_xlim(0, 11); a.set_ylim(0, 4.8); a.axis("off"); a.grid(False)
box(a, 0.2, 3.8, 3.0, 0.7, "Your program / pv CLI", "#f3f4f6", bold=True)
box(a, 0.2, 2.6, 3.0, 0.8, "Public API (pv.hpp)\n11 operations", "#dbeafe", bold=True)
box(a, 0.2, 0.3, 3.0, 1.9, "Dispatcher\nBackend::Auto →\nchoose_backend(op, W, H)\n• heuristic switch points\n• calibrated on this machine\n• MPI cost model", "#fef3c7", bold=True)
box(a, 4.4, 3.8, 2.9, 0.75, "Row kernels (written once)\nprocess rows [y0, y1)", "#e5e7eb", bold=True)
box(a, 4.4, 2.65, 2.9, 0.8, "Sequential\nplain loop", "#e5e7eb")
box(a, 4.4, 1.5, 2.9, 0.8, "OpenMP\nparallel for over rows", "#dbeafe")
box(a, 4.4, 0.3, 2.9, 0.9, "MPI (+ hybrid OpenMP)\nstrips + halo exchange", "#fde68a")
box(a, 8.2, 1.5, 2.6, 0.8, "CUDA (Canny)\nGPU backend", "#d1fae5", ec="#059669")
box(a, 8.2, 2.65, 2.6, 0.8, "identical output\n(tested bit-for-bit)", "#d1fae5", ec="#059669")
for (x0, y0, x1, y1) in [(1.7, 3.8, 1.7, 3.4), (1.7, 2.6, 1.7, 2.2), (3.2, 1.3, 4.4, 3.05), (3.2, 1.25, 4.4, 1.9), (3.2, 0.9, 4.4, 0.75),
                         (5.85, 3.8, 5.85, 3.45), (7.3, 3.05, 8.2, 3.05), (7.3, 0.9, 8.2, 1.7)]:
    a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2))
save(fig, "fig_architecture.png")

# ================================================================ scaling (canny: seq / OpenMP / MPI / hybrid)
if have("bench.csv") and have("mpi.csv"):
    bench = pd.read_csv(RES / "bench.csv")
    mpi = pd.read_csv(RES / "mpi.csv")
    hyb = pd.read_csv(RES / "mpi_hybrid.csv") if have("mpi_hybrid.csv") else None
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
    best = []
    for n in sorted(bench["size"].unique()):
        s = float(bench.loc[bench["size"] == n, "seq_ms"].iloc[0])
        o = float(bench.loc[bench["size"] == n, "omp_ms"].iloc[0])
        best.append({"size": int(n), "seq_ms": s, "omp_ms": o, "speedup": s / o})
    summary["canny_by_size"] = best

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    a = ax[0]
    a.loglog(bench["pixels"], bench["seq_ms"], "o-", color=C["seq"], label="Sequential")
    a.loglog(bench["pixels"], bench["omp_ms"], "s-", color=C["omp"], label=f"OpenMP ({threads} thr)")
    m4 = mpi[mpi.ranks == 4]
    a.loglog(m4["pixels"], m4["mpi_ms"], "^-", color=C["mpi"], label="MPI (4 ranks)")
    if hyb is not None:
        h = hyb[hyb.ranks == 4]
        a.loglog(h["pixels"], h["mpi_ms"], "D-", color=C["hyb"], label="Hybrid (4 ranks x 2 thr)")
    a.set(xlabel="image size (pixels)", ylabel="Canny time (ms)", title="Execution time")
    a.legend(frameon=False)
    b = ax[1]
    sp = [seq_big / r[1] for r in rows]
    b.bar(range(len(rows)), sp, color=[r[2] for r in rows])
    for i, v in enumerate(sp):
        b.text(i, v + 0.04, f"{v:.2f}x", ha="center", fontsize=8)
    b.axhline(1, color="k", lw=0.8, ls="--")
    b.set_xticks(range(len(rows)))
    b.set_xticklabels([r[0] for r in rows], fontsize=7)
    b.set(ylabel="speedup vs sequential", title=f"Canny speedup on a {big}x{big} image")
    save(fig, "fig_scaling.png")

# ================================================================ all operations
FAMILY = {"grayscale": "#6b7280", "blur": "#2563eb", "median": "#2563eb", "sobel": "#dc2626", "prewitt": "#dc2626",
          "laplacian": "#dc2626", "canny": "#dc2626", "threshold": "#059669", "equalize": "#059669", "erode": "#d97706",
          "dilate": "#d97706"}
if have("ops.csv"):
    o = pd.read_csv(RES / "ops.csv")
    top = int(o["size"].max())
    t = o[o["size"] == top].sort_values("speedup")
    summary["ops_size"] = top
    summary["ops"] = [{"op": r.op, "seq_ms": r.seq_ms, "omp_ms": r.omp_ms, "speedup": r.speedup, "efficiency": r.efficiency}
                      for r in o[o["size"] == top].itertuples()]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    bars = ax[0].barh(t["op"], t["speedup"], color=[FAMILY.get(x, C["omp"]) for x in t["op"]])
    for b, v in zip(bars, t["speedup"]):
        ax[0].text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}x", va="center", fontsize=9)
    ax[0].axvline(1, color="k", lw=0.8, ls="--")
    ax[0].set_xlim(0, t["speedup"].max() * 1.15)
    ax[0].set(xlabel="speedup of OpenMP over sequential", title=f"Speedup of every operation ({top}x{top}, {int(o['threads'].iloc[0])} threads)")
    ends = []
    for op, g in o.groupby("op"):
        ax[1].semilogx(g["pixels"], g["speedup"], "o-", ms=3, lw=1.2, color=FAMILY.get(op, C["omp"]), alpha=0.85)
        last = g.sort_values("pixels").iloc[-1]
        ends.append([op, float(last["pixels"]), float(last["speedup"])])
    # spread the end-of-line labels vertically so they never overlap
    ends.sort(key=lambda e: e[2])
    ylab = [e[2] for e in ends]
    for i in range(1, len(ylab)):
        ylab[i] = max(ylab[i], ylab[i - 1] + 0.16)
    for (op, xp, yv), yl in zip(ends, ylab):
        ax[1].annotate(op, xy=(xp, yv), xytext=(xp * 1.5, yl), fontsize=7, va="center",
                       arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.5))
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_xlim(right=o["pixels"].max() * 6)
    ax[1].set(xlabel="image size (pixels)", ylabel="speedup", title="Speedup versus image size")
    save(fig, "fig_ops.png")

# ================================================================ dispatcher
if have("dispatch.csv"):
    d = pd.read_csv(RES / "dispatch.csv")
    d["best"] = d[["seq_ms", "omp_ms"]].min(axis=1)

    def chosen(col):
        return np.where(d[col] == "sequential", d.seq_ms, d.omp_ms)

    strategies = {"Always\nsequential": d.seq_ms.values, "Always\nOpenMP": d.omp_ms.values, "pv\nheuristic": chosen("heuristic"),
                  "pv calibrated\n(old rule)": chosen("strict") if "strict" in d else None,
                  "pv calibrated\n(robust)": chosen("calibrated")}
    strategies = {k: v for k, v in strategies.items() if v is not None}
    tab = []
    for k, v in strategies.items():
        r = v / d.best.values
        tab.append({"strategy": k.replace("\n", " "), "total_vs_oracle_pct": float(v.sum() / d.best.sum() * 100),
                    "geomean": float(np.exp(np.log(r).mean())), "worst": float(r.max()), "within10": float((r <= 1.10).mean() * 100)})
    summary["dispatch"] = tab
    summary["dispatch_cases"] = int(len(d))
    summary["dispatch_ops"] = int(d["op"].nunique())
    small, bigd = d[d.pixels < 100000], d[d.pixels >= 1_000_000]
    summary["small_always_omp_slowdown"] = float(small.omp_ms.sum() / small.seq_ms.sum())
    summary["large_always_seq_slowdown"] = float(bigd.seq_ms.sum() / bigd.omp_ms.sum())
    # per-operation accuracy of the final dispatcher
    per = []
    for op, g in d.groupby("op", sort=False):
        bt = g[["seq_ms", "omp_ms"]].min(axis=1).values
        ct = np.where(g["calibrated"] == "sequential", g.seq_ms, g.omp_ms)
        ht = np.where(g["heuristic"] == "sequential", g.seq_ms, g.omp_ms)
        per.append({"op": op, "heuristic_within10": float((ht <= 1.1 * bt).mean() * 100), "calibrated_within10": float((ct <= 1.1 * bt).mean() * 100)})
    summary["dispatch_per_op"] = per

    cols = [C["seq"], C["bad"], C["amber"], "#60a5fa", C["hyb"]][: len(tab)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    g = [t["geomean"] for t in tab]
    ax[0].bar(range(len(g)), g, color=cols)
    for i, v in enumerate(g):
        ax[0].text(i, v * 1.02, f"{v:.2f}x", ha="center")
    ax[0].axhline(1, color="k", lw=0.8, ls="--")
    ax[0].set_xticks(range(len(g)))
    ax[0].set_xticklabels(list(strategies), fontsize=8)
    ax[0].set(ylabel="time vs. best possible (geometric mean)", title=f"Average slowdown over {len(d)} test cases")
    w = [t["worst"] for t in tab]
    ax[1].bar(range(len(w)), w, color=cols)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, max(w) * 3)
    for i, v in enumerate(w):
        ax[1].text(i, v * 1.1, f"{v:.1f}x", ha="center")
    ax[1].set_xticks(range(len(w)))
    ax[1].set_xticklabels(list(strategies), fontsize=8)
    ax[1].set(ylabel="slowdown (log scale)", title="Worst single case")
    save(fig, "fig_dispatch.png")

    ops = list(d["op"].unique())
    ncol = 4
    nrow = int(np.ceil(len(ops) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for a, op in zip(axes.flat, ops):
        q = d[d["op"] == op].sort_values("pixels")
        a.loglog(q["pixels"], q["seq_ms"], "o-", color=C["seq"], ms=3, label="sequential")
        a.loglog(q["pixels"], q["omp_ms"], "s-", color=C["omp"], ms=3, label="OpenMP")
        a.set_title(op, fontsize=10)
    for a in list(axes.flat)[len(ops):]:
        a.axis("off")
    axes[0][0].legend(frameon=False, fontsize=8)
    for r in range(nrow):
        axes[r][0].set_ylabel("time (ms)")
    for a in axes[-1]:
        a.set_xlabel("pixels")
    save(fig, "fig_crossover.png")

# ================================================================ calibration stability
if have("calib_stability.csv"):
    s = pd.read_csv(RES / "calib_stability.csv")
    lo, hi = 1e3, 4e6                                   # clip "never" (huge) and "always" (0) for plotting / statistics
    s["v"] = s["switch_px"].clip(lower=lo, upper=hi)
    ops = list(s["op"].unique())
    fig, a = plt.subplots(figsize=(11, 4.4))
    spread = {}
    rng = np.random.default_rng(0)
    for k, (rule, col, dx, lab) in enumerate([("strict", C["amber"], -0.17, "old rule"), ("robust", C["omp"], 0.17, "robust rule")]):
        q = s[s["rule"] == rule]
        for i, op in enumerate(ops):
            v = q[q["op"] == op]["v"].values
            a.scatter(np.full(len(v), i + dx) + rng.uniform(-0.05, 0.05, len(v)), v, color=col, s=26, alpha=0.85, label=lab if i == 0 else None)
        sp = [np.log10(q[q["op"] == op]["v"].max()) - np.log10(q[q["op"] == op]["v"].min()) for op in ops]
        spread[rule] = float(10 ** np.mean(sp))
    a.set_yscale("log")
    a.set_xticks(range(len(ops)))
    a.set_xticklabels(ops, rotation=30, ha="right")
    a.set(ylabel="switch point found (pixels)", title="Calibration repeatability: each dot is one calibration run")
    a.legend(frameon=False)
    save(fig, "fig_calibration.png")
    summary["calibration_runs"] = int(s["run"].nunique())
    summary["calibration_spread"] = spread            # typical max/min ratio between runs of the same op

# ================================================================ MPI cost model
if have("mpi_model.csv"):
    m = pd.read_csv(RES / "mpi_model.csv")
    cfgs = list(m.groupby(["ranks", "local"], sort=True).groups.keys())
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.3 * len(cfgs), 3.9), squeeze=False)
    out = []
    for a, (ranks, local) in zip(axes[0], cfgs):
        g = m[(m.ranks == ranks) & (m.local == local)]
        for series, col, mk in [("sequential", C["seq"], "o"), ("openmp", C["omp"], "s"), ("mpi", C["mpi"], "^")]:
            q = g[g.series == series].sort_values("pixels")
            a.plot(q["pixels"] / 1e6, q["predicted_ms"], "-", color=col, lw=1.5, label=f"{series} (model)")
            a.scatter(q["pixels"] / 1e6, q["measured_ms"], color=col, marker=mk, s=28, zorder=3, label=f"{series} (measured)")
            held = q[q["size"] >= 4096]                     # larger than any size used for fitting
            err = float((abs(held["predicted_ms"] - held["measured_ms"]) / held["measured_ms"]).mean() * 100)
            out.append({"ranks": int(ranks), "local": local, "series": series, "heldout_error_pct": err,
                        "b_ns_per_px": float(q["b_ns_per_px"].iloc[0]), "a_ms": float(q["a_ms"].iloc[0]),
                        "crossover_px": float(q["mpi_crossover_px"].iloc[0])})
        cross = g["mpi_crossover_px"].iloc[0]
        a.set(title=f"{int(ranks)} ranks" + (" x OpenMP" if local == "openmp" else "") +
                    ("  -  MPI never wins here" if cross < 0 else f"  -  MPI wins above {cross / 1e6:.1f} Mpx"),
              xlabel="image size (megapixels)", ylabel="Canny time (ms)")
    axes[0][0].legend(frameon=False, fontsize=7)
    save(fig, "fig_mpi_model.png")
    summary["mpi_model"] = out

# ================================================================ Canny stage breakdown (Amdahl evidence)
STAGE_COL = {"gray": "#9ca3af", "filters": "#2563eb", "hysteresis": "#dc2626"}
if have("stages.csv"):
    st = pd.read_csv(RES / "stages.csv")
    big = int(st["size"].max())
    rows = {b: st[(st["size"] == big) & (st["backend"] == b)].iloc[0] for b in ("sequential", "openmp")}
    summary["stages"] = {b: {"gray_ms": float(r.gray_ms), "filters_ms": float(r.filters_ms), "hysteresis_ms": float(r.hysteresis_ms),
                             "total_ms": float(r.total_ms)} for b, r in rows.items()}
    summary["stages_size"] = big
    summary["hysteresis_share_openmp_pct"] = float(rows["openmp"].hysteresis_ms / rows["openmp"].total_ms * 100)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    left = {"sequential": 0.0, "openmp": 0.0}
    for key, lab in [("gray_ms", "gray"), ("filters_ms", "filters"), ("hysteresis_ms", "hysteresis")]:
        for i, b in enumerate(("sequential", "openmp")):
            v = float(rows[b][key])
            ax[0].barh(i, v, left=left[b], color=STAGE_COL[lab.split("_")[0]], label=lab if i == 0 else None)
            if v > 12:
                ax[0].text(left[b] + v / 2, i, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9)
            left[b] += v
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(["Sequential", "OpenMP (8 thr)"])
    ax[0].set(xlabel="time (ms)", title=f"Where Canny spends its time ({big}x{big})")
    ax[0].legend(frameon=False, title="stage (ms)", loc="lower right")
    o = st[st["backend"] == "openmp"].sort_values("pixels")
    ax[1].semilogx(o["pixels"], o["hysteresis_ms"] / o["total_ms"] * 100, "o-", color=STAGE_COL["hysteresis"])
    ax[1].set(xlabel="image size (pixels)", ylabel="hysteresis share of Canny time (%)", ylim=(0, 60),
              title="The serial stage takes a growing share when parallel")
    save(fig, "fig_stages.png")

# ================================================================ CUDA projection (EXPECTED values, not measurements)
# A memory-bandwidth model of the GPU path in src/cuda_backend.cu: upload BGR input, run five
# bandwidth-bound kernels, download 1-byte labels, then run hysteresis on the CPU (measured, OpenMP).
#   t_cuda(N) = (3N + N) / PCIe  +  BYTES_PER_PX * N / GPU_BW  +  launch  +  t_hysteresis(N)
CUDA_CASES = {   # (PCIe GB/s for pageable cudaMemcpy, effective GPU memory bandwidth GB/s)
    "typical":    (6.0, 150.0),    # mid-range GPU on PCIe 3.0 x16
    "pessimistic": (3.0, 80.0),    # entry-level / laptop GPU
    "optimistic": (11.0, 300.0),   # recent GPU on PCIe 4.0
}
CUDA_BYTES_PER_PX = 40.0           # device-memory traffic of the 5 kernels per pixel, incl. cache inefficiency
CUDA_LAUNCH_MS = 0.5
if have("stages.csv"):
    st = pd.read_csv(RES / "stages.csv")
    proj = []
    for n in sorted(st["size"].unique()):
        if n < 1024:
            continue
        seq = st[(st["size"] == n) & (st["backend"] == "sequential")].iloc[0]
        omp = st[(st["size"] == n) & (st["backend"] == "openmp")].iloc[0]
        npx = float(n) * n
        row = {"size": int(n), "seq_ms": float(seq.total_ms), "omp_ms": float(omp.total_ms), "hysteresis_ms": float(omp.hysteresis_ms)}
        for case, (pcie, bw) in CUDA_CASES.items():
            t_gpu = (4.0 * npx) / (pcie * 1e9) * 1e3 + CUDA_BYTES_PER_PX * npx / (bw * 1e9) * 1e3 + CUDA_LAUNCH_MS
            row[f"gpu_part_ms_{case}"] = t_gpu
            row[f"cuda_ms_{case}"] = t_gpu + float(omp.hysteresis_ms)
        row["speedup_vs_seq"] = row["seq_ms"] / row["cuda_ms_typical"]
        row["speedup_vs_omp"] = row["omp_ms"] / row["cuda_ms_typical"]
        row["speedup_vs_seq_range"] = [row["seq_ms"] / row["cuda_ms_pessimistic"], row["seq_ms"] / row["cuda_ms_optimistic"]]
        proj.append(row)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in proj]).to_csv(RES / "cuda_projection.csv", index=False)
    summary["cuda_projection"] = {"rows": proj, "bytes_per_px": CUDA_BYTES_PER_PX, "launch_ms": CUDA_LAUNCH_MS,
                                  "cases": {k: {"pcie_gbs": v[0], "gpu_bw_gbs": v[1]} for k, v in CUDA_CASES.items()}}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(len(proj))
    w = 0.26
    ax[0].bar(xs - w, [r["seq_ms"] for r in proj], w, color=C["seq"], label="Sequential (measured)")
    ax[0].bar(xs, [r["omp_ms"] for r in proj], w, color=C["omp"], label="OpenMP, 8 threads (measured)")
    ax[0].bar(xs + w, [r["cuda_ms_typical"] for r in proj], w, color="#34d399", edgecolor="#047857", hatch="//", label="CUDA (expected)")
    for x, r in zip(xs, proj):
        ax[0].text(x + w, r["cuda_ms_typical"] + 3, f"{r['speedup_vs_seq']:.1f}x", ha="center", fontsize=8)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"{r['size']}x{r['size']}" for r in proj])
    ax[0].set(ylabel="Canny time (ms)", title="Canny: measured CPU backends and expected CUDA time")
    ax[0].legend(frameon=False, fontsize=8)
    last = proj[-1]
    parts = [("Upload + kernels + download", last["gpu_part_ms_typical"], "#34d399"), ("Hysteresis on CPU", last["hysteresis_ms"], STAGE_COL["hysteresis"])]
    left = 0.0
    for lab, v, col in parts:
        ax[1].barh(0, v, left=left, color=col, edgecolor="white", label=f"{lab} ({v:.0f} ms)")
        left += v
    ax[1].barh(1, last["omp_ms"], color=C["omp"], label=f"OpenMP total ({last['omp_ms']:.0f} ms)")
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["CUDA (expected)", "OpenMP (measured)"])
    ax[1].set(xlabel="time (ms)", title=f"Expected CUDA time is dominated by hysteresis ({last['size']}x{last['size']})")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig_cuda_projection.png")

# ================================================================ vs OpenCV
if have("vs_opencv.csv"):
    v = pd.read_csv(RES / "vs_opencv.csv")
    threads = summary.get("threads", 8)
    r = v[v["size"] == 4096].iloc[0]
    summary["opencv_4096"] = {k: float(r[k]) for k in ["pv_seq_ms", "pv_omp_ms", "pv_auto_ms", "cv_1thread_ms", "cv_nthreads_ms"]}
    fig, a = plt.subplots(figsize=(7, 4))
    a.loglog(v["pixels"], v["pv_auto_ms"], "o-", color=C["omp"], label="pv (auto dispatch)")
    a.loglog(v["pixels"], v["cv_1thread_ms"], "s--", color=C["cv"], alpha=0.6, label="OpenCV 1 thread")
    a.loglog(v["pixels"], v["cv_nthreads_ms"], "s-", color=C["cv"], label=f"OpenCV {threads} threads")
    a.set(xlabel="image size (pixels)", ylabel="Canny time (ms)", title="Parallel-Vision vs OpenCV (same pipeline)")
    a.legend(frameon=False)
    save(fig, "fig_opencv.png")

# ================================================================ logs -> summary
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
        parts = line.split()
        if len(parts) == 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
            sw[parts[0]] = {"robust": int(parts[1]), "strict": int(parts[2])}
    summary["switch_points"] = sw

# ================================================================ edge picture
inp, outp = RES / "sample_input.jpg", RES / "sample_canny.png"
if inp.exists() and outp.exists():
    a_img, b_img = plt.imread(inp), plt.imread(outp)
    step = max(1, a_img.shape[1] // 1100)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].imshow(downscale(a_img, step))
    ax[0].set_title("Input")
    ax[1].imshow(downscale(b_img, step, "max"), cmap="gray")
    ax[1].set_title("Canny edges (pv, automatic backend)")
    for x in ax:
        x.axis("off")
        x.grid(False)
    save(fig, "fig_edges.png")

(RES / "summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", RES / "summary.json")
