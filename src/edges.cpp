#include "parallel.hpp"
#include <cmath>
#include <vector>

namespace pv {
namespace {

// Gradient pass: float magnitude + quantised direction (0:horiz 1:\ 2:vert 3:/).
struct Gradient {
    int w, h;
    Buf<float>   mag;
    Buf<uint8_t> dir;
};

// Pixels whose magnitude is below `cut` can never become edges (hysteresis needs >= low,
// and NMS only ever compares a candidate >= low against its neighbours), so they are
// zeroed and skip the direction computation. This keeps results identical while
// removing the branchy part for the (typically majority) flat pixels.
Gradient compute_gradient(const Image& g, Backend b, float cut) {
    const int w = g.width, h = g.height;
    Gradient out{w, h, Buf<float>(static_cast<size_t>(w) * h), Buf<uint8_t>(static_cast<size_t>(w) * h)};
    const float cut2 = cut * cut;
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* r0 = g.row(clampi(y - 1, 0, h - 1));
        const uint8_t* r1 = g.row(y);
        const uint8_t* r2 = g.row(clampi(y + 1, 0, h - 1));
        float* mag = &out.mag[static_cast<size_t>(y) * w];
        uint8_t* dir = &out.dir[static_cast<size_t>(y) * w];
        thread_local std::vector<float> gxb, gyb;
        gxb.resize(w); gyb.resize(w);
        float* GX = gxb.data(); float* GY = gyb.data();
        // pass 1: derivatives. Interior is clamp-free and vectorises; edges use clamping.
        auto px = [&](int x) {
            int xl = clampi(x - 1, 0, w - 1), xr = clampi(x + 1, 0, w - 1);
            GX[x] = static_cast<float>(-r0[xl] + r0[xr] - 2 * r1[xl] + 2 * r1[xr] - r2[xl] + r2[xr]);
            GY[x] = static_cast<float>(-r0[xl] - 2 * r0[x] - r0[xr] + r2[xl] + 2 * r2[x] + r2[xr]);
        };
        px(0); if (w > 1) px(w - 1);
        for (int x = 1; x < w - 1; ++x) {
            GX[x] = static_cast<float>(-r0[x - 1] + r0[x + 1] - 2 * r1[x - 1] + 2 * r1[x + 1] - r2[x - 1] + r2[x + 1]);
            GY[x] = static_cast<float>(-r0[x - 1] - 2 * r0[x] - r0[x + 1] + r2[x - 1] + 2 * r2[x] + r2[x + 1]);
        }
        // pass 2: magnitude (vectorised), then direction only where it can matter
        for (int x = 0; x < w; ++x) mag[x] = std::sqrt(GX[x] * GX[x] + GY[x] * GY[x]);
        for (int x = 0; x < w; ++x) {
            if (GX[x] * GX[x] + GY[x] * GY[x] < cut2) { mag[x] = 0.f; dir[x] = 0; continue; }
            float ax = std::fabs(GX[x]), ay = std::fabs(GY[x]);
            dir[x] = ay <= 0.41421356f * ax ? 0
                   : ay >= 2.41421356f * ax ? 2
                   : (GX[x] * GY[x] > 0 ? 1 : 3);
        }
    });
    return out;
}

// Non-maximum suppression: keep only local maxima along the gradient direction.
Buf<float> non_max_suppress(const Gradient& g, Backend b) {
    const int w = g.w, h = g.h;
    static const int dx[4] = {1, 1, 0, 1}, dy[4] = {0, 1, 1, -1};
    Buf<float> out(static_cast<size_t>(w) * h);
    for_rows(b, 0, h, [&](int y) {
        float* o = &out[static_cast<size_t>(y) * w];
        if (y == 0 || y == h - 1) { for (int x = 0; x < w; ++x) o[x] = 0.f; return; }
        o[0] = 0.f; o[w - 1] = 0.f;
        for (int x = 1; x < w - 1; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            float m = g.mag[i];
            if (m == 0.f) { o[x] = 0.f; continue; }   // sub-threshold pixel: nothing to suppress
            int d = g.dir[i];
            float a = g.mag[static_cast<size_t>(y + dy[d]) * w + (x + dx[d])];
            float c = g.mag[static_cast<size_t>(y - dy[d]) * w + (x - dx[d])];
            o[x] = (m >= a && m >= c) ? m : 0.f;
        }
    });
    return out;
}

// Hysteresis: strong seeds flood into connected weak pixels. The seed scan is parallel;
// the flood fill is inherently serial but only touches edge pixels. The result is the
// set of weak pixels connected to a seed, so it does not depend on visit order.
Image hysteresis(const Buf<float>& nms, int w, int h, float lo, float hi, Backend b) {
    Image out(w, h, 1);
    std::vector<int> stack;
    if (b == Backend::OpenMP) {
        #pragma omp parallel
        {
            std::vector<int> local;
            #pragma omp for schedule(static) nowait
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    if (nms[static_cast<size_t>(y) * w + x] >= hi) {
                        out.row(y)[x] = 255;
                        local.push_back(y * w + x);
                    }
            #pragma omp critical
            stack.insert(stack.end(), local.begin(), local.end());
        }
    } else {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if (nms[static_cast<size_t>(y) * w + x] >= hi) {
                    out.row(y)[x] = 255;
                    stack.push_back(y * w + x);
                }
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
    Gradient g = compute_gradient(gray, b, 0.f);
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
    Gradient g = compute_gradient(blurred, b, static_cast<float>(low));
    Buf<float> nms = non_max_suppress(g, b);
    return hysteresis(nms, g.w, g.h, static_cast<float>(low), static_cast<float>(high), b);
}

} // namespace pv
