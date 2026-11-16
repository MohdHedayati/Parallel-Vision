// Parallel-Vision public API.
#pragma once
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#  ifdef PV_BUILD_DLL
#    define PV_API __declspec(dllexport)
#  else
#    define PV_API __declspec(dllimport)
#  endif
#else
#  define PV_API
#endif

namespace pv {

// 8-bit interleaved image, 1 (gray) or 3 (BGR) channels. Independent of OpenCV.
struct Image {
    int width = 0, height = 0, channels = 1;
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, int c = 1) : width(w), height(h), channels(c),
        data(static_cast<size_t>(w) * h * c, 0) {}
    uint8_t*       row(int y)       { return data.data() + static_cast<size_t>(y) * width * channels; }
    const uint8_t* row(int y) const { return data.data() + static_cast<size_t>(y) * width * channels; }
    bool empty() const { return data.empty(); }
};

enum class Backend { Auto, Sequential, OpenMP, MPI, CUDA };
enum class Op { Grayscale, GaussianBlur, Sobel, Threshold, Canny,
                Median, Prewitt, Laplacian, HistEqualize, Erode, Dilate };

PV_API const char* to_string(Backend b);
PV_API const char* to_string(Op op);
PV_API bool        backend_available(Backend b);

// Per-operation switch point (pixels per image): below it the thread-fork overhead
// beats the gain, so Auto stays sequential. Defaults are a fixed heuristic;
// calibrate() / load_calibration() replace them with values measured on this machine.
struct DispatchConfig {
    static constexpr int kOps = 11;                // number of Op values
    long omp_min_pixels[kOps] = {100000, 100000, 100000, 100000, 100000, 100000,
                                 100000, 100000, 100000, 100000, 100000};
    // CUDA (Canny only): Auto uses the GPU at or above this size. LONG_MAX = never, unless
    // calibrate() measured that the GPU wins on this machine.
    long cuda_canny_min_pixels = 2147483647L;
};
PV_API DispatchConfig& dispatch_config();
PV_API void reset_dispatch_defaults();             // back to the static heuristic

// Calibration: micro-benchmark sequential vs OpenMP (and CUDA if present) on synthetic
// images and fit the per-op switch points. Takes a few seconds.
//   robust = true : median-smoothed timing ratios, switch point interpolated in log-size
//   robust = false: original rule (first size from which OpenMP wins by >= 10 % at every
//                   larger size); kept so the two can be compared.
PV_API void calibrate(int reps = 5, bool robust = true);
PV_API bool save_calibration(const char* path);
PV_API bool load_calibration(const char* path);    // false if file missing/invalid

// Resolve Auto (or an unavailable backend) to a concrete one.
PV_API Backend choose_backend(Op op, int width, int height, Backend requested = Backend::Auto);

// ---- algorithms: every function takes a Backend; Auto lets the dispatcher decide ----
// filters
PV_API Image to_gray(const Image& in, Backend b = Backend::Auto);
PV_API Image gaussian_blur(const Image& gray, double sigma = 1.4, Backend b = Backend::Auto);
PV_API Image median_filter(const Image& gray, int radius = 1, Backend b = Backend::Auto);   // (2r+1)^2 window
// edge detection
PV_API Image sobel(const Image& gray, Backend b = Backend::Auto);          // magnitude, clipped to 255
PV_API Image prewitt(const Image& gray, Backend b = Backend::Auto);        // magnitude, clipped to 255
PV_API Image laplacian(const Image& gray, Backend b = Backend::Auto);      // |4-neighbour Laplacian|, clipped
PV_API Image canny(const Image& in, double low = 50, double high = 150,
                   double sigma = 1.4, Backend b = Backend::Auto);
// histogram and morphology
PV_API Image threshold(const Image& gray, int t, Backend b = Backend::Auto);
PV_API int   otsu_threshold(const Image& gray);
PV_API Image equalize_hist(const Image& gray, Backend b = Backend::Auto);
PV_API Image erode(const Image& gray, int radius = 1, Backend b = Backend::Auto);          // (2r+1)^2 minimum
PV_API Image dilate(const Image& gray, int radius = 1, Backend b = Backend::Auto);         // (2r+1)^2 maximum

// Run one operation with default parameters (used by calibration and benchmarks).
// `rgb` is a 3-channel image (Grayscale, Canny), `gray` its 1-channel version (all others).
PV_API void run_op(Op op, const Image& rgb, const Image& gray, Backend b);

// ---- MPI backend (distributed memory) ----
// Row-block decomposition: rank 0 scatters horizontal strips, neighbouring ranks exchange
// halo rows (depth = blur radius + 2), every rank runs blur/gradient/NMS on its strip, and
// the compact edge labels are gathered on rank 0 for the (global) hysteresis step.
// COLLECTIVE: every rank must call mpi_canny. Only rank 0's `in` is read; only rank 0
// receives the result (other ranks get an empty Image). Results are bit-identical to canny().
// `local` is the backend each rank uses on its own strip (Sequential or OpenMP => hybrid).
// Without PV_WITH_MPI these throw / return false.
PV_API bool  mpi_init(int* argc, char*** argv);   // MPI_Init unless already initialised
PV_API void  mpi_finalize();
PV_API int   mpi_rank();
PV_API int   mpi_size();
PV_API void  mpi_barrier();
PV_API void  mpi_sleep_barrier();                 // barrier that sleeps instead of spinning (for timing a single rank)
PV_API Image mpi_canny(const Image& in, double low = 50, double high = 150,
                       double sigma = 1.4, Backend local = Backend::Sequential);

// ---- cost model: when does each backend pay off? ----
// time(N) ~ a + b * N  for an image of N pixels, fitted from measurements.
struct LinearCost {
    double a_ms = 0, b_ns_per_px = 0;
    double at(double pixels) const { return a_ms + b_ns_per_px * pixels * 1e-6; }
};
// Fit the Canny cost of a single-process backend (Sequential or OpenMP) on squares 1024..3072.
PV_API LinearCost fit_cost(Backend b, int reps = 3);
// COLLECTIVE. Fit the cost of mpi_canny on the current number of ranks (valid on every rank).
PV_API LinearCost fit_mpi_cost(Backend local = Backend::Sequential, int reps = 3);
// Smallest image size (pixels) at which `mpi` beats both single-process backends; -1 if never.
PV_API double mpi_crossover_pixels(const LinearCost& seq, const LinearCost& omp, const LinearCost& mpi);

// ---- CUDA backend (compiled only with -DPV_WITH_CUDA=ON; needs an NVIDIA GPU) ----
PV_API bool  cuda_available();                    // built with CUDA and a device is present
PV_API Image cuda_canny(const Image& in, double low = 50, double high = 150, double sigma = 1.4);

} // namespace pv
