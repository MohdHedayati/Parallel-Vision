// Internal: row-parallel loop helper. Kernels are written per-row so the same
// code serves Sequential, OpenMP and (later) MPI strips.
#pragma once
#include "pv/pv.hpp"

namespace pv {

template <class F>
inline void for_rows(Backend b, int y0, int y1, F&& f) {
    if (b == Backend::OpenMP) {
        #pragma omp parallel for schedule(static)
        for (int y = y0; y < y1; ++y) f(y);
    } else {
        for (int y = y0; y < y1; ++y) f(y);
    }
}

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline uint8_t clamp_u8(float v) { return v < 0 ? 0 : (v > 255 ? 255 : static_cast<uint8_t>(v + 0.5f)); }

} // namespace pv
