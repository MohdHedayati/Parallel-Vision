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

// Tunable thresholds used by the heuristic dispatcher (pixels per image).
struct DispatchConfig {
    long omp_min_pixels = 100000;   // below this, thread overhead beats the gain
};
PV_API DispatchConfig& dispatch_config();

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

} // namespace pv
