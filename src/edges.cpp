#include "parallel.hpp"
#include <cmath>
#include <vector>

namespace pv {
namespace {

// Gradient pass: float magnitude + quantised direction (0:horiz 1:\ 2:vert 3:/).
struct Gradient {
    int w, h;
    std::vector<float>   mag;
    std::vector<uint8_t> dir;
};

Gradient compute_gradient(const Image& g, Backend b) {
    const int w = g.width, h = g.height;
    Gradient out{w, h, std::vector<float>(static_cast<size_t>(w) * h, 0.f),
                       std::vector<uint8_t>(static_cast<size_t>(w) * h, 0)};
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* r0 = g.row(clampi(y - 1, 0, h - 1));
        const uint8_t* r1 = g.row(y);
        const uint8_t* r2 = g.row(clampi(y + 1, 0, h - 1));
        for (int x = 0; x < w; ++x) {
            int xl = clampi(x - 1, 0, w - 1), xr = clampi(x + 1, 0, w - 1);
            float gx = -r0[xl] + r0[xr] - 2 * r1[xl] + 2 * r1[xr] - r2[xl] + r2[xr];
            float gy = -r0[xl] - 2 * r0[x] - r0[xr] + r2[xl] + 2 * r2[x] + r2[xr];
            size_t i = static_cast<size_t>(y) * w + x;
            out.mag[i] = std::sqrt(gx * gx + gy * gy);
            float ax = std::fabs(gx), ay = std::fabs(gy);
            out.dir[i] = ay <= 0.41421356f * ax ? 0
                       : ay >= 2.41421356f * ax ? 2
                       : (gx * gy > 0 ? 1 : 3);
        }
    });
    return out;
}

// Non-maximum suppression: keep only local maxima along the gradient direction.
std::vector<float> non_max_suppress(const Gradient& g, Backend b) {
    const int w = g.w, h = g.h;
    static const int dx[4] = {1, 1, 0, 1}, dy[4] = {0, 1, 1, -1};
    std::vector<float> out(static_cast<size_t>(w) * h, 0.f);
    for_rows(b, 1, h - 1, [&](int y) {
        for (int x = 1; x < w - 1; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            int d = g.dir[i];
            float m = g.mag[i];
            float a = g.mag[static_cast<size_t>(y + dy[d]) * w + (x + dx[d])];
            float c = g.mag[static_cast<size_t>(y - dy[d]) * w + (x - dx[d])];
            out[i] = (m >= a && m >= c) ? m : 0.f;
        }
    });
    return out;
}

// Hysteresis: strong seeds flood into connected weak pixels. Inherently serial.
Image hysteresis(const std::vector<float>& nms, int w, int h, float lo, float hi) {
    Image out(w, h, 1);
    std::vector<int> stack;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (nms[static_cast<size_t>(y) * w + x] >= hi) {
                out.row(y)[x] = 255;
                stack.push_back(y * w + x);
            }
    while (!stack.empty()) {
        int p = stack.back(); stack.pop_back();
        int py = p / w, px = p % w;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int ny = py + dy, nx = px + dx;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                uint8_t& o = out.row(ny)[nx];
                if (o == 0 && nms[static_cast<size_t>(ny) * w + nx] >= lo) {
                    o = 255;
                    stack.push_back(ny * w + nx);
                }
            }
    }
    return out;
}

} // namespace

Image sobel(const Image& gray, Backend req) {
    Backend b = choose_backend(Op::Sobel, gray.width, gray.height, req);
    Gradient g = compute_gradient(gray, b);
    Image out(gray.width, gray.height, 1);
    for_rows(b, 0, gray.height, [&](int y) {
        uint8_t* d = out.row(y);
        for (int x = 0; x < gray.width; ++x)
            d[x] = clamp_u8(g.mag[static_cast<size_t>(y) * gray.width + x]);
    });
    return out;
}

Image canny(const Image& in, double low, double high, double sigma, Backend req) {
    Backend b = choose_backend(Op::Canny, in.width, in.height, req);
    Image gray = to_gray(in, b);
    Image blurred = gaussian_blur(gray, sigma, b);
    Gradient g = compute_gradient(blurred, b);
    std::vector<float> nms = non_max_suppress(g, b);
    return hysteresis(nms, g.w, g.h, static_cast<float>(low), static_cast<float>(high));
}

} // namespace pv
