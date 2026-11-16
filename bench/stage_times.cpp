// Where does Canny spend its time? Stage-by-stage timing, sequential vs OpenMP (CSV on stdout).
//   pv_stage_times [reps=5]  > stages.csv
// Stages: gray | blur+gradient+suppression ("filters") | hysteresis (global, partly serial).
// Used to explain the Amdahl limit on Canny and as an input to the CUDA projection model.
#include "pv/pv.hpp"
#include "internal.hpp"          // pipeline pieces (exported from the DLL for this benchmark)
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace pv;

static Image synth(int n) {
    Image im(n, n, 3);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n * 3; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / 48 + y / 16) % 2) * 200 + (x * y) % 50);
    return im;
}

template <class F>
static double best_ms(F&& f, int reps) {
    f();
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    std::printf("size,pixels,backend,gray_ms,filters_ms,hysteresis_ms,total_ms\n");
    for (int n : {512, 1024, 2048, 3072, 4096}) {
        Image rgb = synth(n);
        Image gray = to_gray(rgb, Backend::Sequential);
        for (Backend b : {Backend::Sequential, Backend::OpenMP}) {
            double tg = best_ms([&] { to_gray(rgb, b); }, reps);
            double tf = best_ms([&] { detail::canny_labels(gray, 50, 150, 1.4, b); }, reps);
            Buf<uint8_t> lab = detail::canny_labels(gray, 50, 150, 1.4, b);
            double th = best_ms([&] { detail::hysteresis_from_labels(lab.data(), n, n, b); }, reps);
            std::printf("%d,%ld,%s,%.4f,%.4f,%.4f,%.4f\n", n, (long)n * n, to_string(b), tg, tf, th, tg + tf + th);
            std::fflush(stdout);
        }
    }
    return 0;
}
