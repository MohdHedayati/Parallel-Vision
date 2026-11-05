// Benchmark: Canny on synthetic images, every backend, CSV on stdout.
//   pv_bench [reps=5] > results.csv
#include "pv/pv.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

static pv::Image make_image(int w, int h) {
    pv::Image im(w, h, 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int k = 0; k < 3; ++k)
                im.row(y)[x * 3 + k] = static_cast<uint8_t>(((x / 16 + y / 16) % 2) * 200 + (x * y + k) % 50);
    return im;
}

static double time_ms(const pv::Image& im, pv::Backend b, int reps) {
    pv::canny(im, 50, 150, 1.4, b);  // warm-up
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        pv::canny(im, 50, 150, 1.4, b);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::printf("size,pixels,threads,seq_ms,omp_ms,speedup,efficiency,auto_choice,auto_ms,best_choice\n");
    for (int n : sizes) {
        pv::Image im = make_image(n, n);
        double seq = time_ms(im, pv::Backend::Sequential, reps);
        double omp = time_ms(im, pv::Backend::OpenMP, reps);
        pv::Backend ac = pv::choose_backend(pv::Op::Canny, n, n);
        double aut = time_ms(im, pv::Backend::Auto, reps);
        int th = omp_get_max_threads();
        std::printf("%d,%ld,%d,%.4f,%.4f,%.3f,%.3f,%s,%.4f,%s\n", n, (long)n * n, th, seq, omp,
                    seq / omp, seq / omp / th, pv::to_string(ac), aut,
                    seq <= omp ? "sequential" : "openmp");
        std::fflush(stdout);
    }
    return 0;
}
