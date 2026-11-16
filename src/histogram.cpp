// Histogram equalisation. The histogram is a parallel reduction: every thread counts
// into its own 256-bin array and the arrays are summed. Integer counts make the result
// independent of the thread count (bit-identical to the sequential version).
#include "parallel.hpp"
#include <array>

namespace pv {

Image equalize_hist(const Image& gray, Backend req) {
    Backend b = choose_backend(Op::HistEqualize, gray.width, gray.height, req);
    const int w = gray.width, h = gray.height;

    std::array<long long, 256> hist{};
    if (b == Backend::OpenMP) {
        #pragma omp parallel
        {
            std::array<long long, 256> local{};
            #pragma omp for schedule(static) nowait
            for (int y = 0; y < h; ++y) {
                const uint8_t* s = gray.row(y);
                for (int x = 0; x < w; ++x) ++local[s[x]];
            }
            #pragma omp critical
            for (int i = 0; i < 256; ++i) hist[i] += local[i];
        }
    } else {
        for (int y = 0; y < h; ++y) {
            const uint8_t* s = gray.row(y);
            for (int x = 0; x < w; ++x) ++hist[s[x]];
        }
    }

    // cumulative distribution -> lookup table (standard equalisation formula)
    const long long total = static_cast<long long>(w) * h;
    std::array<uint8_t, 256> lut{};
    long long cdf = 0, cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        cdf += hist[i];
        if (cdf_min == 0 && cdf > 0) cdf_min = cdf;
        lut[i] = (total == cdf_min) ? static_cast<uint8_t>(i)       // constant image: leave unchanged
               : static_cast<uint8_t>(((cdf - cdf_min) * 255 + (total - cdf_min) / 2) / (total - cdf_min));
    }

    Image out(w, h, 1);
    for_rows(b, 0, h, [&](int y) {
        const uint8_t* s = gray.row(y);
        uint8_t* d = out.row(y);
        for (int x = 0; x < w; ++x) d[x] = lut[s[x]];
    });
    return out;
}

} // namespace pv
