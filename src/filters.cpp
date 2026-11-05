#include "parallel.hpp"
#include <cmath>

namespace pv {

Image gaussian_blur(const Image& gray, double sigma, Backend req) {
    Backend b = choose_backend(Op::GaussianBlur, gray.width, gray.height, req);
    const int w = gray.width, h = gray.height;

    int r = static_cast<int>(std::ceil(3 * sigma));
    if (r < 1) r = 1;
    std::vector<float> k(2 * r + 1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = std::exp(-(i * i) / (2 * sigma * sigma));
    for (auto& v : k) v /= sum;

    std::vector<float> tmp(static_cast<size_t>(w) * h);
    // horizontal pass
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* s = gray.row(y);
        float* t = &tmp[static_cast<size_t>(y) * w];
        for (int x = 0; x < w; ++x) {
            float acc = 0;
            for (int i = -r; i <= r; ++i) acc += k[i + r] * s[clampi(x + i, 0, w - 1)];
            t[x] = acc;
        }
    });
    // vertical pass (needs the whole horizontal pass finished: implicit barrier above)
    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        uint8_t* d = out.row(y);
        for (int x = 0; x < w; ++x) {
            float acc = 0;
            for (int i = -r; i <= r; ++i)
                acc += k[i + r] * tmp[static_cast<size_t>(clampi(y + i, 0, h - 1)) * w + x];
            d[x] = clamp_u8(acc);
        }
    });
    return out;
}

} // namespace pv
