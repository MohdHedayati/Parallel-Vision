// Internal pipeline pieces shared by the shared-memory and MPI backends.
#pragma once
#include "parallel.hpp"

namespace pv {
namespace detail {

// blur -> gradient -> non-max suppression on a gray image. Labels: 0 none, 1 weak (>= low), 2 strong (>= high).
PV_API Buf<uint8_t> canny_labels(const Image& gray, double low, double high, double sigma, Backend b);

// Connect weak labels to strong ones. Needs the whole image (connectivity is global).
PV_API Image hysteresis_from_labels(const uint8_t* labels, int width, int height, Backend b);

// Least-squares line ms = a + b * pixels through measured (pixels, ms) points.
inline LinearCost linear_fit(const std::vector<double>& px, const std::vector<double>& ms) {
    const double n = static_cast<double>(px.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < px.size(); ++i) { sx += px[i]; sy += ms[i]; sxx += px[i] * px[i]; sxy += px[i] * ms[i]; }
    const double den = n * sxx - sx * sx;
    const double slope = den != 0 ? (n * sxy - sx * sy) / den : 0.0;          // ms per pixel
    LinearCost c;
    c.b_ns_per_px = slope * 1e6;
    c.a_ms = (sy - slope * sx) / n;
    if (c.a_ms < 0 && sxx > 0) {                 // fixed overhead cannot be negative: refit through the origin
        c.a_ms = 0;
        c.b_ns_per_px = sxy / sxx * 1e6;
    }
    return c;
}

} // namespace detail
} // namespace pv
