// Sequential vs OpenMP for EVERY operation across image sizes (CSV on stdout).
//   pv_ops_bench [reps=3] [max_side=4096]  > ops.csv
#include "pv/pv.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

using namespace pv;

static Image synth(int w, int h, int c) {
    Image im(w, h, c);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * c; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / (16 * c) + y / 16) % 2) * 200 + (x * y) % 50);
    return im;
}

static double best_ms(Op op, const Image& rgb, const Image& gray, Backend b, int reps) {
    run_op(op, rgb, gray, b);
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        run_op(op, rgb, gray, b);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3;
    const int max_side = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int threads = omp_get_max_threads();
    std::printf("op,size,pixels,threads,seq_ms,omp_ms,speedup,efficiency,auto_choice\n");
    for (int o = 0; o < DispatchConfig::kOps; ++o) {
        Op op = static_cast<Op>(o);
        for (int n : {128, 256, 512, 1024, 2048, 4096}) {
            if (n > max_side) continue;
            Image rgb = synth(n, n, 3), gray = to_gray(rgb, Backend::Sequential);
            double s = best_ms(op, rgb, gray, Backend::Sequential, reps);
            double p = best_ms(op, rgb, gray, Backend::OpenMP, reps);
            std::printf("%s,%d,%ld,%d,%.4f,%.4f,%.3f,%.3f,%s\n", to_string(op), n, (long)n * n, threads, s, p, s / p,
                        s / p / threads, to_string(choose_backend(op, n, n)));
            std::fflush(stdout);
        }
    }
    return 0;
}
