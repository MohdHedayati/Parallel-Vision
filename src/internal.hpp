// Internal pipeline pieces shared by the shared-memory and MPI backends.
#pragma once
#include "parallel.hpp"

namespace pv {
namespace detail {

// blur -> gradient -> non-max suppression on a gray image. Labels: 0 none, 1 weak (>= low), 2 strong (>= high).
Buf<uint8_t> canny_labels(const Image& gray, double low, double high, double sigma, Backend b);

// Connect weak labels to strong ones. Needs the whole image (connectivity is global).
Image hysteresis_from_labels(const uint8_t* labels, int width, int height, Backend b);

} // namespace detail
} // namespace pv
