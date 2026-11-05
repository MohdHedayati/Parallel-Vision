#include "parallel.hpp"
#include <array>

namespace pv {

Image threshold(const Image& gray, int t, Backend req) {
    Backend b = choose_backend(Op::Threshold, gray.width, gray.height, req);
    Image out(gray.width, gray.height, 1);
    for_rows(b, 0, gray.height, [&](int y) {
        const uint8_t* s = gray.row(y);
        uint8_t* d = out.row(y);
        for (int x = 0; x < gray.width; ++x) d[x] = s[x] > t ? 255 : 0;
    });
    return out;
}

int otsu_threshold(const Image& gray) {
    std::array<double, 256> hist{};
    for (uint8_t v : gray.data) hist[v] += 1;
    double total = static_cast<double>(gray.data.size()), sum = 0;
    for (int i = 0; i < 256; ++i) sum += i * hist[i];
    double wb = 0, sb = 0, best = -1;
    int t = 0;
    for (int i = 0; i < 256; ++i) {
        wb += hist[i];
        if (wb == 0) continue;
        double wf = total - wb;
        if (wf == 0) break;
        sb += i * hist[i];
        double mb = sb / wb, mf = (sum - sb) / wf;
        double between = wb * wf * (mb - mf) * (mb - mf);
        if (between > best) { best = between; t = i; }
    }
    return t;
}

} // namespace pv
