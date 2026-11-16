# Parallel-Vision

A C++17 image-processing library with **sequential, OpenMP, MPI, hybrid (MPI + OpenMP) and CUDA backends** behind one
API, and a **dispatcher that picks the fastest backend** for the operation and the image size.

Parallelism is not free: on a small image a parallel run can be dozens of times *slower* than a plain loop, and on a
large image a sequential run leaves the machine idle. Parallel-Vision measures where the break-even point is — per
operation, on your machine — and uses it.

**Authors:** Mohammed Mehdi Hedayati (241080023) and Nathan Julian Dsouza (241080016) —
Parallel Programming Laboratory, Semester 5 mini-project.

## Run the demo (about 30 seconds)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
```

Runs the tests, a console demo (correctness, dispatcher, speedups), MPI on 1/2/4 processes, every algorithm on a real
picture, then draws graphs with Python and opens them. Output: `results\demo\`.
Use `-Image C:\path\photo.jpg` for your own picture, `-NoOpen` to skip opening the graphs.

## Features

| | |
|---|---|
| **11 operations** | grayscale, Gaussian blur, median filter, Sobel, Prewitt, Laplacian, Canny, threshold (+ Otsu), histogram equalisation, erosion, dilation |
| **Backends** | sequential and OpenMP for every operation; MPI (strips + halo exchange) and hybrid for Canny; CUDA for Canny (expected ≈ 4× over sequential at 4096², see section 6; not verified on hardware) |
| **Same output everywhere** | OpenMP, MPI (1–8 ranks) and hybrid are bit-identical to sequential (automated tests) |
| **Dispatcher** | `Backend::Auto` picks sequential vs OpenMP per operation and size; switch points are a fixed heuristic or **calibrated** on your machine (`pv::calibrate()`) |
| **MPI cost model** | fits time = a + b·pixels per backend and says whether/when MPI would pay off (`pv_mpi_model`) |
| **Reproducible** | `scripts\run_all.ps1` regenerates every number and graph; `docs\build_report.js` rebuilds the report from them |

Final report: [docs/Parallel-Vision-Report.docx](docs/Parallel-Vision-Report.docx).

---

## 1. Requirements (Windows 11)

| Tool | Why | How to get it |
|---|---|---|
| MSYS2 (UCRT64) with g++, CMake, Ninja, OpenCV, MS-MPI | compiler (OpenMP), build, image file I/O, MPI headers/libs | `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-msmpi` |
| Microsoft MPI runtime | provides `mpiexec` | install *MS-MPI* (`msmpisetup.exe`) from Microsoft |
| Python 3 + `matplotlib`, `pandas` | graphs (`pip install matplotlib pandas`) | python.org |
| Node.js | only to regenerate the Word report | nodejs.org |

Use the **MinGW** OpenCV from MSYS2, not the pre-built MSVC binaries (ABI-incompatible with g++).
The library itself does **not** depend on OpenCV; only the `pv` command-line tool and the OpenCV comparison do.

## 2. Build

```powershell
./build.ps1                       # OpenMP build (sequential + OpenMP backends)
./build.ps1 -DPV_WITH_MPI=ON      # also build the MPI backend, MPI tests and MPI benchmarks
```

`build.ps1` puts `C:\msys64\ucrt64\bin` on `PATH`, configures CMake with Ninja in `build/`, and builds. Executables and
`libpv.dll` end up in `build/bin/`. To run them from your own shell, add `C:\msys64\ucrt64\bin` to `PATH` first.

| CMake option | Default | Meaning |
|---|---|---|
| `PV_WITH_MPI` | OFF | build the MPI backend (needs MS-MPI) |
| `PV_WITH_CUDA` | OFF | build the CUDA backend (needs `nvcc` and an NVIDIA GPU; see section 6) |
| `PV_BUILD_CLI` | ON | build `pv.exe` and `pv_vs_opencv.exe` (need OpenCV) |
| `PV_NATIVE` | OFF | `-march=native` (faster, but the binary only runs on this CPU family) |

## 3. Quick start

### Command line

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
build\bin\pv.exe canny     photo.jpg edges.png                    # Backend::Auto (dispatcher decides)
build\bin\pv.exe canny     photo.jpg edges.png --backend seq      # force sequential (omp = OpenMP, cuda = GPU)
build\bin\pv.exe canny     photo.jpg edges.png --low 40 --high 120 --sigma 1.4
build\bin\pv.exe sobel     photo.jpg out.png                      # also: prewitt laplacian
build\bin\pv.exe blur      photo.jpg out.png --sigma 2
build\bin\pv.exe median    photo.jpg out.png --radius 2           # 5x5 median (radius 1..3)
build\bin\pv.exe equalize  photo.jpg out.png
build\bin\pv.exe erode     photo.jpg out.png --radius 2           # also: dilate
build\bin\pv.exe threshold photo.jpg bw.png --t otsu              # or --t 128
build\bin\pv.exe canny     photo.jpg edges.png --calibrate        # measure this machine, save pv_calibration.txt
build\bin\pv.exe canny     photo.jpg edges.png --use-calibration  # use the saved switch points
```

Each run prints the image size, the backend that was actually used, and the time.

### As a C++ library

```cpp
#include "pv/pv.hpp"

pv::Image img(width, height, 3);               // 8-bit BGR; fill img.data yourself (or via OpenCV, see cli/pv.cpp)

pv::Image edges = pv::canny(img);                                      // Auto: pv chooses
pv::Image a = pv::canny(img, 50, 150, 1.4, pv::Backend::Sequential);   // force a backend
pv::Image g = pv::to_gray(img);
pv::Image m = pv::median_filter(g, 1, pv::Backend::OpenMP);            // also prewitt, laplacian, equalize_hist, erode, dilate

pv::Backend which = pv::choose_backend(pv::Op::Median, width, height); // ask what Auto would do

pv::calibrate();                                  // optional: measure sequential vs OpenMP on this machine
pv::save_calibration("pv_calibration.txt");       // reload later with pv::load_calibration(...)
```

Link with CMake: `add_subdirectory(Parallel-Vision)` then `target_link_libraries(myapp PRIVATE pv)`.

### MPI (distributed memory)

`mpi_canny` is **collective**: every rank calls it; only rank 0's input is read and only rank 0 receives the result.

```cpp
pv::mpi_init(&argc, &argv);
pv::Image out = pv::mpi_canny(in, 50, 150, 1.4, pv::Backend::OpenMP);  // OpenMP inside each rank = hybrid
pv::mpi_finalize();
```

Run: `mpiexec -n 4 your_program.exe`. See `bench/mpi_bench.cpp` and `tests/test_mpi.cpp` for complete examples.

## 4. How it works

* **One kernel, several drivers.** Each algorithm is written once as a function over a range of rows. `for_rows()` runs it
  in a plain loop (Sequential) or under `#pragma omp parallel for` (OpenMP); the MPI backend runs the same kernels on
  each rank's strip. So results are identical by construction — and tested to be.
* **MPI strips with halo exchange.** Rank 0 scatters horizontal strips; neighbouring ranks exchange halo rows of depth
  `ceil(3σ)+2` (blur radius + gradient + non-max suppression); each rank computes its rows; compact edge labels are
  gathered on rank 0 for the global hysteresis step.
* **Dispatcher.** For `Backend::Auto`, OpenMP is chosen when `width × height ≥ N*(op)` (and more than one thread
  exists), otherwise Sequential. `N*` is a fixed 100 000 px by default (*heuristic*) or measured per operation on your
  machine by `pv::calibrate()` (median-smoothed timing ratios, interpolated crossing). MPI is never chosen automatically
  because every process must take part — call it explicitly, and use the cost model to decide.
* **MPI cost model.** `pv::fit_cost` / `pv::fit_mpi_cost` fit `time = a + b·pixels` for each backend;
  `pv::mpi_crossover_pixels` gives the size above which MPI would beat both single-process backends (`-1` = never).

## 5. Reproducing the results

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1      # 30 s   live demo  -> results\demo\
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1   # 10-15 min  everything -> results\
cd docs ; npm install ; node build_report.js                   # rebuild docs\Parallel-Vision-Report.docx
```

`run_all.ps1` options: `-Image C:\path\photo.jpg`, `-Reps 7` (more repeats), `-SkipBuild`. For stable numbers close
other programs and plug the laptop in; timings vary by roughly 10–20 % between runs.

### What each program measures

| Program | Run | Output |
|---|---|---|
| `pv_tests` | `build\bin\pv_tests.exe` | sequential == OpenMP for all 11 operations; reference checks; calibration round-trip; dispatcher rules |
| `pv_mpi_tests` | `mpiexec -n 4 build\bin\pv_mpi_tests.exe` (`PV_HYBRID=1` for MPI+OpenMP) | MPI result == sequential on several image shapes |
| `pv_demo` | `build\bin\pv_demo.exe results\demo` | the console demo (correctness, heuristic vs calibrated dispatcher, speedups) + CSVs |
| `pv_bench` | `build\bin\pv_bench.exe 5 > results\bench.csv` | Canny sequential vs OpenMP, speedup, efficiency per image size |
| `pv_ops_bench` | `build\bin\pv_ops_bench.exe 3 4096 > results\ops.csv` | all 11 operations, sequential vs OpenMP, 128..4096 |
| `pv_mpi_bench` | `mpiexec -n 4 build\bin\pv_mpi_bench.exe 5` | MPI Canny time per size (`PV_HYBRID=1`, `OMP_NUM_THREADS=2` for hybrid) |
| `pv_stage_times` | `build\bin\pv_stage_times.exe 5 > results\stages.csv` | Canny time per stage (gray / filters / hysteresis); input to the CUDA estimate |
| `pv_mpi_model` | `mpiexec -n 4 build\bin\pv_mpi_model.exe 3` | fitted cost model, prediction error on unseen sizes, MPI crossover |
| `pv_dispatch_report` | `build\bin\pv_dispatch_report.exe 7 > results\dispatch.csv` | 154 cases: always-seq / always-OpenMP / heuristic / calibrated (old and robust rule) vs the oracle |
| `pv_calib_stability` | `build\bin\pv_calib_stability.exe 5 > results\calib_stability.csv` | calibration repeated: how repeatable is each rule? |
| `pv_vs_opencv` | `build\bin\pv_vs_opencv.exe 7 photo.jpg > results\vs_opencv.csv` | pv vs OpenCV Canny (1 and N threads); edge IoU on `photo.jpg` |
| `scripts\make_figures.py` | `python scripts\make_figures.py results` | all PNG figures and `summary.json` |

Reading the CSVs: `speedup = seq_ms / omp_ms`, `efficiency = speedup / threads`; in `dispatch.csv` the `oracle` column is
the faster backend in that case and `heuristic` / `calibrated` / `strict` are what each dispatcher chose.

## 6. CUDA backend

`src/cuda_backend.cu` implements Canny on the GPU (gray, blur, gradient and non-max suppression as kernels; hysteresis on
the CPU). It mirrors the CPU code operation for operation and is built with `--fmad=false`, so the output is designed to
be bit-identical. Expected performance (memory-bandwidth model, see report section 6.9): about 4× over sequential and
2× over 8-thread OpenMP on a 4096² image, limited by the CPU-side hysteresis step *(expected values; not verified on
GPU hardware — the development PC has no NVIDIA GPU)*. To check it on a CUDA machine (Linux, Windows + MSVC, or Google
Colab; `nvcc` does not work with MinGW):

```bash
cmake -S . -B build -DPV_WITH_CUDA=ON -DPV_BUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/pv_cuda_tests          # checks GPU == sequential CPU, prints timings
```

Without `PV_WITH_CUDA`, `Backend::CUDA` falls back to OpenMP.

## 7. Repository layout

```
include/pv/pv.hpp        public API (Image, Backend, Op, 11 algorithms, dispatcher, calibration, MPI, cost model, CUDA)
src/                     kernels (color, filters, stencil, histogram, morphology, edges, threshold), dispatcher,
                         calibration, cost model, MPI backend, CUDA backend (+ stub)
cli/pv.cpp               command-line tool (OpenCV only for file I/O)
tests/                   test_main.cpp, test_mpi.cpp (mpiexec), test_cuda.cpp (needs a GPU)
bench/                   demo, bench, ops_bench, mpi_bench, mpi_model, dispatch_report, calib_stability, vs_opencv
scripts/                 demo.ps1, run_all.ps1, make_figures.py, make_demo_figures.py
docs/                    build_report.js -> Parallel-Vision-Report.docx, JUSTIFICATION.md (earlier design notes)
results/                 generated CSVs, figures, summary.json; results\demo\ = live-demo output
```

## 8. Limitations (short)

* **CUDA numbers are expected values** from a bandwidth model, not measurements on a GPU.
* **Not faster than OpenCV.** On a 4096² Canny, OpenCV needs about as long on *one* thread as pv does on eight. The
  contribution is the multi-backend design and the backend selection, not kernel speed.
* **Speedups are modest**: about 1.6–1.8× for Canny and 1.9–3.3× for the other operations on 8 threads (memory
  bandwidth limits all of them; Canny's hysteresis step is also partly serial).
* **MPI on one PC does not beat OpenMP**; not tested on a cluster; only Canny has an MPI path; messages are limited to 2 GB.
* One machine, one image family; run-to-run variation 10–20 %.
