#include "parallel.hpp"
#include <cmath>

namespace pv {

// Separable Gaussian. Both passes are arranged so the innermost loop runs along x
// over contiguous memory (tap-outer / pixel-inner), which the compiler vectorises;
// borders are handled once per row via a clamped padded copy, not per pixel.
Image gaussian_blur(const Image& gray, double sigma, Backend req) {
    Backend b = choose_backend(Op::GaussianBlur, gray.width, gray.height, req);
    const int w = gray.width, h = gray.height;

    int r = static_cast<int>(std::ceil(3 * sigma));
    if (r < 1) r = 1;
    std::vector<float> k(2 * r + 1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = std::exp(-(i * i) / (2 * sigma * sigma));
    for (auto& v : k) v /= sum;

    Buf<float> tmp(static_cast<size_t>(w) * h);   // fully written by the first pass
    // horizontal pass
    for_rows(b, 0, h, [&](int y) {
        thread_local std::vector<float> pad;
        pad.resize(static_cast<size_t>(w) + 2 * r);
        const uint8_t* s = gray.row(y);
        for (int x = 0; x < w; ++x) pad[x + r] = s[x];
        for (int i = 0; i < r; ++i) { pad[i] = s[clampi(i - r, 0, w - 1)]; pad[w + r + i] = s[clampi(w + i, 0, w - 1)]; }
        float* t = &tmp[static_cast<size_t>(y) * w];
        for (int x = 0; x < w; ++x) t[x] = 0.f;
        for (int i = 0; i <= 2 * r; ++i) {
            const float kv = k[i];
            const float* p = pad.data() + i;
            for (int x = 0; x < w; ++x) t[x] += kv * p[x];
        }
    });
    // vertical pass (the barrier at the end of the loop above guarantees tmp is complete)
    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        thread_local std::vector<float> acc;
        acc.assign(static_cast<size_t>(w), 0.f);
        for (int i = 0; i <= 2 * r; ++i) {
            const float kv = k[i];
            const float* tr = &tmp[static_cast<size_t>(clampi(y + i - r, 0, h - 1)) * w];
            for (int x = 0; x < w; ++x) acc[x] += kv * tr[x];
        }
        uint8_t* d = out.row(y);
        for (int x = 0; x < w; ++x) d[x] = clamp_u8(acc[x]);
    });
    return out;
}

} // namespace pv
