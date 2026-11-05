#include "pv/pv.hpp"
#include <cstdio>
#include <cstdlib>

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
