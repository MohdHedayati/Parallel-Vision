#include "parallel.hpp"

namespace pv {

Image to_gray(const Image& in, Backend req) {
    if (in.channels == 1) return in;
    Backend b = choose_backend(Op::Grayscale, in.width, in.height, req);
    Image out(in.width, in.height, 1);
    const int c = in.channels;
    for_rows(b, 0, in.height, [&](int y) {
        const uint8_t* s = in.row(y);
        uint8_t* d = out.row(y);
        for (int x = 0; x < in.width; ++x, s += c)   // BGR order
            d[x] = static_cast<uint8_t>((114 * s[0] + 587 * s[1] + 299 * s[2] + 500) / 1000);
    });
    return out;
}

} // namespace pv
