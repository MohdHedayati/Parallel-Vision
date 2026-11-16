// Calibration: measure where OpenMP (and CUDA) start to beat sequential on THIS machine.
#include "pv/pv.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <vector>

namespace pv {
namespace {

Image synth(int w, int h, int c) {
    Image im(w, h, c);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * c; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / (16 * c) + y / 16) % 2) * 200 + (x * y) % 50);
    return im;
}

template <class F>
double best_ms(F&& f, int reps) {
    f();  // warm-up
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

// Original rule: smallest size from which the parallel backend wins by >= 10 % at EVERY larger size.
long strict_switch(const std::vector<int>& sides, const std::vector<double>& ratio) {
    const int n = static_cast<int>(sides.size());
    int first = n;
    while (first > 0 && ratio[first - 1] < 0.9) --first;
    return first == n ? LONG_MAX : first == 0 ? 0 : static_cast<long>(sides[first]) * sides[first];
}

// Robust rule: smooth log(parallel/sequential) with a 3-point median (one noisy measurement
// cannot move the answer), find the point from which it stays below zero, and interpolate
// the zero crossing in log-size instead of snapping to the measurement grid.
long robust_switch(const std::vector<int>& sides, const std::vector<double>& ratio) {
    const int n = static_cast<int>(sides.size());
    std::vector<double> lr(n), s(n);
    for (int i = 0; i < n; ++i) lr[i] = std::log(ratio[i]);
    for (int i = 0; i < n; ++i) {
        if (i == 0 || i == n - 1) { s[i] = lr[i]; continue; }
        double a = lr[i - 1], b = lr[i], c = lr[i + 1];
        s[i] = std::max(std::min(a, b), std::min(std::max(a, b), c));   // median of three
    }
    int first = n;
    while (first > 0 && s[first - 1] < 0) --first;
    if (first == n) return LONG_MAX;
    if (first == 0) return 0;
    const double t = s[first - 1] / (s[first - 1] - s[first]);          // s[first-1] >= 0 > s[first]
    const double n0 = static_cast<double>(sides[first - 1]) * sides[first - 1];
    const double n1 = static_cast<double>(sides[first]) * sides[first];
    return static_cast<long>(std::llround(std::exp(std::log(n0) + t * (std::log(n1) - std::log(n0)))));
}

} // namespace

void calibrate(int reps, bool robust) {
    static const int kSides[] = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
    const std::vector<int> sides(std::begin(kSides), std::end(kSides));
    const int n = static_cast<int>(sides.size());
    const bool gpu = cuda_available();
    std::vector<double> gpu_ratio(n, 1.0);

    for (int o = 0; o < DispatchConfig::kOps; ++o) {
        const Op op = static_cast<Op>(o);
        std::vector<double> ratio(n);
        for (int i = 0; i < n; ++i) {
            Image rgb = synth(sides[i], sides[i], 3), gray = to_gray(rgb, Backend::Sequential);
            const int r = sides[i] <= 128 ? reps * 4 : reps;    // small images are cheap: repeat more
            double s = best_ms([&] { run_op(op, rgb, gray, Backend::Sequential); }, r);
            double p = best_ms([&] { run_op(op, rgb, gray, Backend::OpenMP); }, r);
            ratio[i] = p / s;
            if (gpu && op == Op::Canny) {
                double g = best_ms([&] { cuda_canny(rgb, 50, 150, 1.4); }, r);
                gpu_ratio[i] = g / std::min(s, p);              // GPU vs the best CPU backend
            }
        }
        dispatch_config().omp_min_pixels[o] = robust ? robust_switch(sides, ratio) : strict_switch(sides, ratio);
        if (gpu && op == Op::Canny)
            dispatch_config().cuda_canny_min_pixels = robust ? robust_switch(sides, gpu_ratio) : strict_switch(sides, gpu_ratio);
    }
}

bool save_calibration(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;
    for (int o = 0; o < DispatchConfig::kOps; ++o)
        std::fprintf(f, "%d %ld\n", o, dispatch_config().omp_min_pixels[o]);
    std::fprintf(f, "cuda %ld\n", dispatch_config().cuda_canny_min_pixels);
    std::fclose(f);
    return true;
}

bool load_calibration(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    long v[DispatchConfig::kOps], cuda = dispatch_config().cuda_canny_min_pixels;
    bool ok = true;
    for (int o = 0; o < DispatchConfig::kOps && ok; ++o) {
        int idx;
        ok = std::fscanf(f, "%d %ld", &idx, &v[o]) == 2 && idx == o;
    }
    if (ok) std::fscanf(f, " cuda %ld", &cuda);                 // optional line
    std::fclose(f);
    if (!ok) return false;
    for (int o = 0; o < DispatchConfig::kOps; ++o) dispatch_config().omp_min_pixels[o] = v[o];
    dispatch_config().cuda_canny_min_pixels = cuda;
    return true;
}

} // namespace pv
