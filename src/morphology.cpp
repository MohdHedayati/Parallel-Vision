// Grayscale morphology with a (2r+1) x (2r+1) square structuring element.
// Min/max over a square is separable: a horizontal pass then a vertical pass.
// Both passes run tap-outer / pixel-inner so the inner loop is contiguous and vectorises.
#include "parallel.hpp"
#include <vector>

namespace pv {
namespace {

template <bool IsMax>
inline uint8_t pick(uint8_t a, uint8_t b) { return IsMax ? (a > b ? a : b) : (a < b ? a : b); }

template <bool IsMax>
Image morph(const Image& g, int radius, Backend b) {
    const int w = g.width, h = g.height;
    radius = clampi(radius, 1, 15);
    Buf<uint8_t> tmp(static_cast<size_t>(w) * h);                     // fully written by pass 1
    for_rows(b, 0, h, [&](int y) {
        thread_local std::vector<uint8_t> pad;
        pad.resize(static_cast<size_t>(w) + 2 * radius);
        const uint8_t* s = g.row(y);
        for (int x = 0; x < w; ++x) pad[x + radius] = s[x];
        for (int i = 0; i < radius; ++i) { pad[i] = s[0]; pad[w + radius + i] = s[w - 1]; }   // replicate border
        uint8_t* t = &tmp[static_cast<size_t>(y) * w];
        for (int x = 0; x < w; ++x) t[x] = pad[x];
        for (int i = 1; i <= 2 * radius; ++i) {
            const uint8_t* p = pad.data() + i;
            for (int x = 0; x < w; ++x) t[x] = pick<IsMax>(t[x], p[x]);
        }
    });
    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        uint8_t* d = out.row(y);
        const uint8_t* first = &tmp[static_cast<size_t>(clampi(y - radius, 0, h - 1)) * w];
        for (int x = 0; x < w; ++x) d[x] = first[x];
        for (int i = 1; i <= 2 * radius; ++i) {
            const uint8_t* tr = &tmp[static_cast<size_t>(clampi(y + i - radius, 0, h - 1)) * w];
            for (int x = 0; x < w; ++x) d[x] = pick<IsMax>(d[x], tr[x]);
        }
    });
    return out;
}

} // namespace

Image erode(const Image& g, int radius, Backend req) {
    return morph<false>(g, radius, choose_backend(Op::Erode, g.width, g.height, req));
}
Image dilate(const Image& g, int radius, Backend req) {
    return morph<true>(g, radius, choose_backend(Op::Dilate, g.width, g.height, req));
}

} // namespace pv
