// Neighbourhood operations: median filter, Prewitt, Laplacian.
// Every function is written per row (see parallel.hpp) so Sequential and OpenMP share the code.
#include "parallel.hpp"
#include <algorithm>
#include <cmath>

namespace pv {
namespace {

inline void srt(uint8_t& a, uint8_t& b) {
    uint8_t lo = a < b ? a : b, hi = a < b ? b : a;
    a = lo; b = hi;
}
// Median of 9 values with a fixed sorting network (19 compare-exchanges, no branches on data).
inline uint8_t median9(uint8_t* p) {
    srt(p[1], p[2]); srt(p[4], p[5]); srt(p[7], p[8]);
    srt(p[0], p[1]); srt(p[3], p[4]); srt(p[6], p[7]);
    srt(p[1], p[2]); srt(p[4], p[5]); srt(p[7], p[8]);
    srt(p[0], p[3]); srt(p[5], p[8]); srt(p[4], p[7]);
    srt(p[3], p[6]); srt(p[1], p[4]); srt(p[2], p[5]);
    srt(p[4], p[7]); srt(p[4], p[2]); srt(p[6], p[4]);
    srt(p[4], p[2]);
    return p[4];
}

} // namespace

Image median_filter(const Image& g, int radius, Backend req) {
    Backend b = choose_backend(Op::Median, g.width, g.height, req);
    const int w = g.width, h = g.height;
    radius = clampi(radius, 1, 3);
    const int n = (2 * radius + 1) * (2 * radius + 1);
    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* rows[7];
        for (int i = 0; i < 2 * radius + 1; ++i) rows[i] = g.row(clampi(y + i - radius, 0, h - 1));
        uint8_t* d = out.row(y);
        uint8_t win[49];
        for (int x = 0; x < w; ++x) {
            int k = 0;
            for (int i = 0; i < 2 * radius + 1; ++i)
                for (int dx = -radius; dx <= radius; ++dx) win[k++] = rows[i][clampi(x + dx, 0, w - 1)];
            if (radius == 1) {
                d[x] = median9(win);
            } else {
                std::nth_element(win, win + n / 2, win + n);
                d[x] = win[n / 2];
            }
        }
    });
    return out;
}

Image prewitt(const Image& g, Backend req) {
    Backend b = choose_backend(Op::Prewitt, g.width, g.height, req);
    const int w = g.width, h = g.height;
    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* r0 = g.row(clampi(y - 1, 0, h - 1));
        const uint8_t* r1 = g.row(y);
        const uint8_t* r2 = g.row(clampi(y + 1, 0, h - 1));
        uint8_t* d = out.row(y);
        for (int x = 0; x < w; ++x) {
            int xl = clampi(x - 1, 0, w - 1), xr = clampi(x + 1, 0, w - 1);
            float gx = static_cast<float>(-r0[xl] + r0[xr] - r1[xl] + r1[xr] - r2[xl] + r2[xr]);
            float gy = static_cast<float>(-r0[xl] - r0[x] - r0[xr] + r2[xl] + r2[x] + r2[xr]);
            d[x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    });
    return out;
}

Image laplacian(const Image& g, Backend req) {
    Backend b = choose_backend(Op::Laplacian, g.width, g.height, req);
    const int w = g.width, h = g.height;
    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* r0 = g.row(clampi(y - 1, 0, h - 1));
        const uint8_t* r1 = g.row(y);
        const uint8_t* r2 = g.row(clampi(y + 1, 0, h - 1));
        uint8_t* d = out.row(y);
        for (int x = 0; x < w; ++x) {
            int xl = clampi(x - 1, 0, w - 1), xr = clampi(x + 1, 0, w - 1);
            int v = r0[x] + r2[x] + r1[xl] + r1[xr] - 4 * r1[x];
            d[x] = static_cast<uint8_t>(v < 0 ? (-v > 255 ? 255 : -v) : (v > 255 ? 255 : v));
        }
    });
    return out;
}

} // namespace pv
