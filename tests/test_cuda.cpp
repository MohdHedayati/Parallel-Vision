// Run on a machine with an NVIDIA GPU, built with -DPV_WITH_CUDA=ON:
//   build/bin/pv_cuda_tests
// Checks that the GPU Canny equals the sequential CPU Canny and prints timings.
#include "pv/pv.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

static pv::Image make(int w, int h, unsigned seed) {
    pv::Image im(w, h, 3);
    srand(seed);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * 3; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / 27 + y / 9) % 2) * 160 + rand() % 60);
    return im;
}

template <class F> static double ms(F&& f) {
    f();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3;
}

int main() {
    if (!pv::cuda_available()) { std::printf("no CUDA device available (or built without PV_WITH_CUDA)\n"); return 2; }
    int failures = 0;
    const std::pair<int, int> shapes[] = {{64, 64}, {257, 131}, {640, 480}, {1920, 1080}};
    for (auto [w, h] : shapes) {
        pv::Image in = make(w, h, 3);
        pv::Image ref = pv::canny(in, 50, 150, 1.4, pv::Backend::Sequential);
        pv::Image gpu = pv::canny(in, 50, 150, 1.4, pv::Backend::CUDA);
        bool ok = ref.data == gpu.data;
        failures += !ok;
        std::printf("%dx%d: %s  cpu-seq %.2f ms  cpu-omp %.2f ms  cuda %.2f ms\n", w, h, ok ? "identical" : "MISMATCH",
                    ms([&] { pv::canny(in, 50, 150, 1.4, pv::Backend::Sequential); }),
                    ms([&] { pv::canny(in, 50, 150, 1.4, pv::Backend::OpenMP); }),
                    ms([&] { pv::canny(in, 50, 150, 1.4, pv::Backend::CUDA); }));
    }
    std::printf(failures ? "CUDA tests FAILED\n" : "CUDA tests passed\n");
    return failures ? 1 : 0;
}
