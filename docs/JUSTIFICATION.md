# Why Parallel-Vision? (measured, with limits)

All numbers: Windows 11, 8 OpenMP threads, MinGW g++ -O3, OpenCV 4.13 (MSYS2), one run each
(timings vary ~10-20 % between runs on this desktop). Raw data: `results/*.csv`.

## 1. What we do NOT claim

We do **not** beat OpenCV on raw speed. On the same Canny pipeline (gray -> Gaussian 1.4 -> Canny 50/150),
4096x4096:

| | time |
|---|---|
| pv sequential | ~365 ms |
| pv OpenMP, 8 threads | ~185 ms |
| OpenCV, 1 thread | ~166 ms |
| OpenCV, 8 threads | ~50 ms |

OpenCV is SIMD-tuned and fuses stages into cache-sized tiles; ours makes many full-image passes and is
memory-bandwidth bound (speedup plateaus near 2x on 8 threads). Output agrees with OpenCV: edge-pixel
IoU = 0.973 on a 3840x2400 photo. (Before optimisation our sequential code was 4.4x slower than OpenCV
1-thread, so earlier "speedups" were flattering; see git history.)

## 2. What the data does support

**Choosing the backend matters, and a blanket `#pragma omp parallel for` is a trap.**
70 cases (5 operations x 14 sizes, incl. non-square, none used for calibration), time relative to the
per-case best of {sequential, OpenMP}:

| strategy | total time vs oracle | geometric mean per case | worst case |
|---|---|---|---|
| always sequential | 165 % | 1.17x | 2.7x |
| always OpenMP | 103 % | **2.25x** | **113x** |
| heuristic dispatcher | 100.2 % | 1.01x | 1.2x |
| calibrated dispatcher | 100.6 % | 1.03x | 1.6x |

Small images (<100k px): always-OpenMP is ~2x slower in total (thread start-up dominates; up to 113x on
tiny ones). Large images (>=1M px): always-sequential is ~1.85x slower. The crossover differs per
operation (calibrated: ~65k px for blur, ~260k for Sobel/Canny, ~1M for grayscale/threshold on this
machine), so one global constant is a compromise.

**Limits of that claim (be upfront about these in the viva):**
- The calibrated dispatcher is **not shown to beat the simple heuristic** here: it won in one run
  (91 % vs 86 % exact match) and lost in the next (91 % vs 97 %). Both are within ~0.6 % of the oracle.
  The honest conclusion is that *having* a size-aware dispatcher gives the gain; *calibrating* it
  would matter on different hardware (fewer cores, laptops, slower memory), which we have not tested.
- Only sequential-vs-OpenMP is evaluated. MPI and CUDA are not implemented yet.
- Single machine, single image family (synthetic + one photo), no confidence intervals.
- Whether OpenCV itself loses on tiny images with threads enabled was too noisy to conclude.

## 3. Where a real gap exists (to verify before presenting)

To my knowledge, and worth double-checking in OpenCV's docs before you state it:
- OpenCV has no distributed (MPI) image-processing path; its CUDA support is a separate API
  (`cv::cuda::`) the caller picks explicitly; there is no single call that decides between
  sequential / threads / cluster / GPU from the input size.
- A library that exposes **one API, several backends, and an explainable, measured selection rule**
  (with a harness that scores the rule against an oracle, as `pv_dispatch_report` does) is a legitimate
  mini-project contribution even where raw kernels are not faster than a mature library.

## 4. Suggested framing for the viva

"Parallel-Vision is an experiment in *when* to parallelise image operations, not a faster OpenCV.
We implement the Canny pipeline for several backends, show that naive parallelisation is up to two orders
of magnitude slower on small inputs and 1.85x slower than needed when parallelism is skipped on large
ones, and build a dispatcher that stays within ~1-3 % of the best backend. We validate correctness
against OpenCV and analyse why speedup saturates (memory bandwidth, serial hysteresis)."

## 5. What would strengthen it most (in order)

1. MPI strip decomposition with halo exchange + a communication-cost term in the dispatcher: the one
   capability OpenCV lacks. Expect it to be *slower* on one PC (shared memory, message-passing overhead);
   the useful result is the cost model predicting when it would pay off (huge images, multi-node, batches).
2. Calibration robustness (more repetitions, median, require consecutive wins) and testing on a second
   machine, so the calibrated-vs-heuristic question gets a real answer.
3. Tile-fused Canny (blur+gradient+NMS per cache-sized tile) to close the bandwidth gap with OpenCV.
4. CUDA backend (cannot be tested on this PC).
