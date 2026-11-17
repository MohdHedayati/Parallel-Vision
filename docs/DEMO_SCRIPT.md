# Demo script: Parallel-Vision (about 5 minutes)

Run: `powershell -ExecutionPolicy Bypass -File scripts\demo.ps1`  (takes 20–30 s; graphs open at the end)
Numbers on screen will differ a little from the report (timings vary 10–20 % per run). That is normal.

## 0. Opening (30 s, before pressing Enter)

- "Parallel-Vision is a C++ image-processing library with sequential, OpenMP, MPI and CUDA backends behind one API."
- "The question we study is *when* to go parallel. On a small image, threads cost more than they save; on a large image, a sequential loop wastes the machine. Our library measures the break-even point and chooses the backend itself."
- "11 operations: grayscale, blur, median, Sobel, Prewitt, Laplacian, Canny, threshold/Otsu, histogram equalisation, erosion, dilation."
- Related work (one sentence; the PDFs are in `docs\references\`): "Recent work supports the ideas behind this: a 2026 study (Souza, WPerformance) shows the best processor depends on the filter, batch size and resolution; a 2026 ICS paper (Abram et al.) shows the best settings differ between CPUs and GPUs, so tuning has to be done per machine; and a 2025 paper (Collom et al.) studies halo-style communication in MPI. Each supports one design decision of ours; none of them is our result."

## 1. Tests (screen: "all tests passed", "MPI tests passed")

- "First, correctness. The unit tests check that the parallel output equals the sequential output bit for bit, for every operation."
- "The MPI test does the same on 4 processes, on awkward shapes such as 100×9 pixels, where each process gets very few rows. Every line says *identical*."
- Point: "Parallel code is only useful if it gives the same answer."

## 2. Console demo

**[1] Correctness table:** "All 11 operations, sequential versus OpenMP: 11 of 11 identical."

**[2] Calibration:**
- "The library just measured *this* machine in about a second: the image size where OpenMP starts to win, for each operation."
- "Notice the switch points differ: median pays off at about 3 000 pixels, threshold only at about 60 000. A heavy operation justifies threads earlier than a cheap one."
- Link to the papers: "This matches Souza 2026, who found the CPU-versus-GPU winner depends on the operation and image size, and Abram et al. 2026, who found the best settings differ between architectures, which is why we calibrate on the machine itself."
- Read the dispatcher table: "Left, the fixed 100 000-pixel rule; right, calibrated. Every row that says NO is where the dispatcher picked the slower backend. The calibrated one gets almost all of them right (about 27–28 of 28), the fixed rule about 24–26."
- Over the full 154-case test in the report: **fixed rule 86 %, calibrated 98 %.**

**[3] Speedups:** "All 11 operations at 2048×2048 on 8 threads. Everything is faster; the filters reach about 3–4×, Canny about 2×. That is below 8× because the passes are limited by memory bandwidth, not by arithmetic."

## 3. MPI (screen: 1, 2, 4 processes)

- "Rank 0 splits the image into strips, neighbouring ranks exchange halo rows, each computes its part, and the result is gathered."
- Link to the paper: "Halo exchange is the standard stencil pattern; Collom et al. 2025 show that persistent and partitioned MPI can cut its communication cost (up to 37 % and 68 % over baseline MPI in their benchmark). That is a natural next step for us."
- "Time drops as processes are added, but it does not beat OpenMP on one PC, because all processes share one memory system. MPI is for clusters."
- "We built a cost model that predicts this: it fits time = a + b·pixels per backend and says whether MPI would ever win. On this PC it says *never*, matching the measurement."

## 4. Real picture (screen: 11 commands, each with a backend and time)

- "Same library, used from the command line. `backend=openmp` shows the dispatcher chose OpenMP automatically for this 3840×2400 image."

## 5. Graphs (4 windows)

1. **Montage:** "Every algorithm on the same photo. The Canny edges are one pixel wide and continuous. Our edges overlap OpenCV's at 97 % (IoU 0.97)."
2. **Speedup bars:** "Speedup of every operation."
3. **Dispatch plot:** "Blue is OpenMP, grey is sequential. Left of the crossing, sequential wins; right of it, OpenMP wins. The triangles mark where the fixed rule chose wrong; calibration removes them."
4. **MPI bars:** "Time and speedup on 1, 2, 4 processes."

- "All graphs are drawn by Python from the numbers just measured: nothing is typed in by hand. The report is built the same way from `scripts\run_all.ps1`."

## Closing (30 s)

- "Result: a size-aware dispatcher stays about 0.3 % from the best possible choice on average, while always using OpenMP is about 94 % slower on average and up to 112× slower on tiny images."

## If they ask

| Question | Answer |
|---|---|
| Why not just use OpenCV? | It is faster (about 3× on 8 threads). Our contribution is the multi-backend design and the automatic backend choice, not kernel speed. |
| Why only about 2× for Canny? | Hysteresis is serial (about 40 % of the parallel time, Figure 5b) and the other stages are memory-bound: Amdahl's law. |
| Does CUDA work? | The code is written and mirrors the CPU maths; the **3.8× is an expected value from a bandwidth model, not measured**. `tests/test_cuda.cpp` verifies it on a GPU. |
| What does calibration add? | Per-operation switch points: 86 % → 98 % correct choices. The fitting rule matters much less. |
| Why does MPI not help? | One PC shares one memory bandwidth, plus scatter/gather copies. The cost model predicts exactly that. |
| What related work did you use? | Three recent papers, in `docs\references\`: Souza 2026 (CPU vs GPU choice depends on operation, batch and resolution), Abram et al. 2026 (cross-architecture auto-tuning: best settings differ per machine), Collom et al. 2025 (MPI stencil communication). |
| Do those papers confirm your numbers? | No. Each motivates one design decision; our results come from our own measurements. None of them covers the full combination (Canny, OpenMP, MPI, CUDA and automatic backend selection). |
| How do you know the results are right? | Bit-identical to sequential in the tests; Canny agrees with OpenCV at IoU 0.97. |
| Can we run the full evaluation? | `scripts\run_all.ps1`, about 5 minutes; every table and graph in the report comes from it. |

## If something goes wrong

- No graphs open: they are in `results\demo\` (open the PNGs by hand).
- Build missing: `scripts\demo.ps1 -Rebuild`.
- MPI error: say "MPI needs the Microsoft MPI runtime", and continue; the rest of the demo still runs.
