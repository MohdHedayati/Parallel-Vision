#include "pv/pv.hpp"
#include <chrono>
#include <climits>
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

void run(Op op, const Image& rgb, const Image& gray, Backend b) {
    switch (op) {
        case Op::Grayscale:    to_gray(rgb, b); break;
        case Op::GaussianBlur: gaussian_blur(gray, 1.4, b); break;
        case Op::Sobel:        sobel(gray, b); break;
        case Op::Threshold:    threshold(gray, 128, b); break;
        case Op::Canny:        canny(rgb, 50, 150, 1.4, b); break;
    }
}

double best_ms(Op op, const Image& rgb, const Image& gray, Backend b, int reps) {
    run(op, rgb, gray, b);  // warm-up
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        run(op, rgb, gray, b);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

} // namespace

void calibrate(int reps) {
    static const int sides[] = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
    constexpr int n = sizeof(sides) / sizeof(sides[0]);
    for (int o = 0; o < DispatchConfig::kOps; ++o) {
        Op op = static_cast<Op>(o);
        bool win[n];
        for (int i = 0; i < n; ++i) {
            Image rgb = synth(sides[i], sides[i], 3), gray = to_gray(rgb, Backend::Sequential);
            double s = best_ms(op, rgb, gray, Backend::Sequential, reps);
            double p = best_ms(op, rgb, gray, Backend::OpenMP, reps);
            win[i] = p < 0.9 * s;   // require a clear win, not noise
        }
        // switch point = smallest size from which OpenMP wins at every larger size
        int first = n;
        while (first > 0 && win[first - 1]) --first;
        dispatch_config().omp_min_pixels[o] =
            first == n ? LONG_MAX : first == 0 ? 0 : static_cast<long>(sides[first]) * sides[first];
    }
}

bool save_calibration(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;
    for (int o = 0; o < DispatchConfig::kOps; ++o)
        std::fprintf(f, "%d %ld\n", o, dispatch_config().omp_min_pixels[o]);
    std::fclose(f);
    return true;
}

bool load_calibration(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    long v[DispatchConfig::kOps];
    bool ok = true;
    for (int o = 0; o < DispatchConfig::kOps && ok; ++o) {
        int idx;
        ok = std::fscanf(f, "%d %ld", &idx, &v[o]) == 2 && idx == o;
    }
    std::fclose(f);
    if (!ok) return false;
    for (int o = 0; o < DispatchConfig::kOps; ++o) dispatch_config().omp_min_pixels[o] = v[o];
    return true;
}

} // namespace pv
