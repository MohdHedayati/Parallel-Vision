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
enum class Op { Grayscale, GaussianBlur, Sobel, Threshold, Canny };

PV_API const char* to_string(Backend b);
PV_API bool        backend_available(Backend b);

// Per-operation switch point (pixels per image): below it the thread-fork overhead
// beats the gain, so Auto stays sequential. Defaults are a fixed heuristic;
// calibrate() / load_calibration() replace them with values measured on this machine.
struct DispatchConfig {
    static constexpr int kOps = 5;                 // number of Op values
    long omp_min_pixels[kOps] = {100000, 100000, 100000, 100000, 100000};
};
PV_API DispatchConfig& dispatch_config();
PV_API void reset_dispatch_defaults();             // back to the static heuristic

// Calibration: micro-benchmark sequential vs OpenMP on synthetic images and fit
// the per-op switch points. Takes roughly a few seconds.
PV_API void calibrate(int reps = 5);
PV_API bool save_calibration(const char* path);
PV_API bool load_calibration(const char* path);    // false if file missing/invalid

// Resolve Auto (or an unavailable backend) to a concrete one.
PV_API Backend choose_backend(Op op, int width, int height, Backend requested = Backend::Auto);

// ---- algorithms: every function takes a Backend; Auto lets the dispatcher decide ----
PV_API Image to_gray(const Image& in, Backend b = Backend::Auto);
PV_API Image gaussian_blur(const Image& gray, double sigma = 1.4, Backend b = Backend::Auto);
PV_API Image sobel(const Image& gray, Backend b = Backend::Auto);          // magnitude, clipped to 255
PV_API Image threshold(const Image& gray, int t, Backend b = Backend::Auto);
PV_API int   otsu_threshold(const Image& gray);
PV_API Image canny(const Image& in, double low = 50, double high = 150,
                   double sigma = 1.4, Backend b = Backend::Auto);

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
PV_API Image mpi_canny(const Image& in, double low = 50, double high = 150,
                       double sigma = 1.4, Backend local = Backend::Sequential);

} // namespace pv
