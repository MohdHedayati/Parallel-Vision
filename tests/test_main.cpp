#include "pv/pv.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static pv::Image random_image(int w, int h, int c, unsigned seed) {
    pv::Image im(w, h, c);
    srand(seed);
    for (auto& v : im.data) v = static_cast<uint8_t>(rand() & 255);
    return im;
}

// White square on black: edges must appear on the border of the square only.
static pv::Image square_image(int n, int c = 1) {
    pv::Image im(n, n, c);
    for (int y = n / 4; y < 3 * n / 4; ++y)
        for (int x = n / 4; x < 3 * n / 4; ++x)
            for (int k = 0; k < c; ++k) im.row(y)[x * c + k] = 255;
    return im;
}

int main() {
    using pv::Backend;

    // Sequential and OpenMP must agree bit-for-bit on every operation.
    pv::Image rgb = random_image(257, 131, 3, 1);
    pv::Image gray = pv::to_gray(rgb, Backend::Sequential);
    CHECK(gray.channels == 1);
    CHECK(pv::to_gray(rgb, Backend::OpenMP).data == gray.data);
    CHECK(pv::gaussian_blur(gray, 1.4, Backend::Sequential).data == pv::gaussian_blur(gray, 1.4, Backend::OpenMP).data);
    CHECK(pv::sobel(gray, Backend::Sequential).data == pv::sobel(gray, Backend::OpenMP).data);
    CHECK(pv::threshold(gray, 128, Backend::Sequential).data == pv::threshold(gray, 128, Backend::OpenMP).data);
    CHECK(pv::canny(rgb, 50, 150, 1.4, Backend::Sequential).data == pv::canny(rgb, 50, 150, 1.4, Backend::OpenMP).data);

    // Canny on a square: interior and far background must be empty, border must have edges.
    const int n = 128;
    pv::Image e = pv::canny(square_image(n), 50, 150, 1.4, Backend::OpenMP);
    CHECK(e.row(n / 2)[n / 2] == 0);        // centre of square
    CHECK(e.row(5)[5] == 0);                // background corner
    int edge_pixels = 0;
    for (uint8_t v : e.data) edge_pixels += v == 255;
    CHECK(edge_pixels > 4 * (n / 2) / 2);   // roughly a perimeter's worth
    CHECK(edge_pixels < 4 * (n / 2) * 4);   // but thin lines, not a filled blob

    // Otsu separates a two-level image.
    pv::Image two(10, 10, 1);
    for (size_t i = 0; i < two.data.size(); ++i) two.data[i] = i % 2 ? 200 : 40;
    int t = pv::otsu_threshold(two);
    CHECK(t >= 40 && t < 200);

    // Dispatcher: small -> sequential, large -> OpenMP (when >1 thread).
    CHECK(pv::choose_backend(pv::Op::Canny, 32, 32) == Backend::Sequential);
    CHECK(pv::choose_backend(pv::Op::Canny, 4000, 4000) != Backend::CUDA);
    CHECK(pv::choose_backend(pv::Op::Canny, 32, 32, Backend::CUDA) == Backend::OpenMP);  // unavailable -> fallback

    // ---- new operations: Sequential == OpenMP bit for bit ----
    CHECK(pv::median_filter(gray, 1, Backend::Sequential).data == pv::median_filter(gray, 1, Backend::OpenMP).data);
    CHECK(pv::median_filter(gray, 2, Backend::Sequential).data == pv::median_filter(gray, 2, Backend::OpenMP).data);
    CHECK(pv::prewitt(gray, Backend::Sequential).data == pv::prewitt(gray, Backend::OpenMP).data);
    CHECK(pv::laplacian(gray, Backend::Sequential).data == pv::laplacian(gray, Backend::OpenMP).data);
    CHECK(pv::equalize_hist(gray, Backend::Sequential).data == pv::equalize_hist(gray, Backend::OpenMP).data);
    CHECK(pv::erode(gray, 2, Backend::Sequential).data == pv::erode(gray, 2, Backend::OpenMP).data);
    CHECK(pv::dilate(gray, 2, Backend::Sequential).data == pv::dilate(gray, 2, Backend::OpenMP).data);

    // median 3x3 and 5x5 against a brute-force reference (clamped borders)
    {
        pv::Image g = random_image(41, 29, 1, 5);
        for (int radius : {1, 2}) {
            pv::Image m = pv::median_filter(g, radius, Backend::Sequential);
            bool ok = true;
            for (int y = 0; y < g.height && ok; ++y)
                for (int x = 0; x < g.width && ok; ++x) {
                    std::vector<int> v;
                    for (int dy = -radius; dy <= radius; ++dy)
                        for (int dx = -radius; dx <= radius; ++dx)
                            v.push_back(g.row(std::min(std::max(y + dy, 0), g.height - 1))[std::min(std::max(x + dx, 0), g.width - 1)]);
                    std::sort(v.begin(), v.end());
                    ok = m.row(y)[x] == v[v.size() / 2];
                }
            CHECK(ok);
        }
    }
    // median removes isolated salt noise
    {
        pv::Image g(20, 20, 1);
        for (auto& v : g.data) v = 100;
        g.row(10)[10] = 255;
        pv::Image m = pv::median_filter(g, 1, Backend::OpenMP);
        bool clean = true;
        for (auto v : m.data) clean = clean && v == 100;
        CHECK(clean);
    }
    // erosion shrinks, dilation grows: erode <= original <= dilate; a lone bright pixel becomes a 3x3 block
    {
        pv::Image g = random_image(37, 23, 1, 9);
        pv::Image e = pv::erode(g, 1, Backend::OpenMP), d = pv::dilate(g, 1, Backend::OpenMP);
        bool ordered = true;
        for (size_t i = 0; i < g.data.size(); ++i) ordered = ordered && e.data[i] <= g.data[i] && g.data[i] <= d.data[i];
        CHECK(ordered);
        pv::Image dot(15, 15, 1);
        dot.row(7)[7] = 255;
        pv::Image grown = pv::dilate(dot, 1, Backend::OpenMP), gone = pv::erode(grown, 1, Backend::OpenMP);
        int on = 0;
        for (auto v : grown.data) on += v == 255;
        CHECK(on == 9);
        CHECK(gone.row(7)[7] == 255);               // opening-closing identity for a single dot
        CHECK(pv::erode(dot, 1, Backend::OpenMP).row(7)[7] == 0);
    }
    // histogram equalisation stretches a low-contrast image to the full range; constant image unchanged
    {
        pv::Image g(64, 64, 1);
        for (size_t i = 0; i < g.data.size(); ++i) g.data[i] = static_cast<uint8_t>(100 + (i * 7) % 30);
        pv::Image q = pv::equalize_hist(g, Backend::OpenMP);
        uint8_t lo = 255, hi = 0;
        for (auto v : q.data) { lo = std::min(lo, v); hi = std::max(hi, v); }
        CHECK(lo == 0 && hi == 255);
        pv::Image flat(8, 8, 1);
        for (auto& v : flat.data) v = 77;
        CHECK(pv::equalize_hist(flat, Backend::OpenMP).data == flat.data);
    }
    // Laplacian of a constant image is 0; Prewitt/Laplacian respond to a vertical step edge
    {
        pv::Image flat(16, 16, 1);
        for (auto& v : flat.data) v = 90;
        pv::Image l = pv::laplacian(flat, Backend::OpenMP);
        bool zero = true;
        for (auto v : l.data) zero = zero && v == 0;
        CHECK(zero);
        pv::Image step(16, 16, 1);
        for (int y = 0; y < 16; ++y)
            for (int x = 8; x < 16; ++x) step.row(y)[x] = 200;
        pv::Image p = pv::prewitt(step, Backend::OpenMP), lp = pv::laplacian(step, Backend::OpenMP);
        CHECK(p.row(8)[7] > 150 && p.row(8)[8] > 150 && p.row(8)[2] == 0 && p.row(8)[13] == 0);
        CHECK(lp.row(8)[7] > 150 && lp.row(8)[8] > 150 && lp.row(8)[3] == 0);
    }

    // dispatcher: ops without an MPI/CUDA path fall back to OpenMP
    CHECK(pv::choose_backend(pv::Op::Median, 64, 64, Backend::MPI) == Backend::OpenMP);
    CHECK(pv::choose_backend(pv::Op::Erode, 64, 64, Backend::CUDA) == Backend::OpenMP);
    CHECK(std::string(pv::to_string(pv::Op::HistEqualize)) == "equalize");

    // Calibration round-trip: per-op thresholds survive save/load and drive the dispatcher.
    pv::dispatch_config().omp_min_pixels[static_cast<int>(pv::Op::Sobel)] = 1234;
    pv::dispatch_config().omp_min_pixels[static_cast<int>(pv::Op::Canny)] = 0;
    CHECK(pv::save_calibration("pv_test_cal.txt"));
    pv::reset_dispatch_defaults();
    CHECK(pv::dispatch_config().omp_min_pixels[static_cast<int>(pv::Op::Sobel)] == 100000);
    CHECK(pv::load_calibration("pv_test_cal.txt"));
    CHECK(pv::dispatch_config().omp_min_pixels[static_cast<int>(pv::Op::Sobel)] == 1234);
    CHECK(!pv::load_calibration("does_not_exist.txt"));
    CHECK(pv::choose_backend(pv::Op::Canny, 2, 2) == Backend::OpenMP);      // threshold 0 -> always OpenMP
    CHECK(pv::choose_backend(pv::Op::Sobel, 30, 30) == Backend::Sequential); // 900 < 1234
    pv::reset_dispatch_defaults();
    std::remove("pv_test_cal.txt");

    std::printf(failures ? "%d test(s) FAILED\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
