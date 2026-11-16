// Compare backend-selection strategies over all operations.
// For each (op, size) we time Sequential and OpenMP, find the oracle (the faster one), and score
// what each dispatcher would have chosen:
//   heuristic  - fixed 100k-pixel switch point
//   calibrated - robust calibration (median-smoothed, interpolated)
//   strict     - the original calibration rule (first size with a >=10 % win at every larger size)
//   pv_dispatch_report [reps=7] > dispatch.csv      (summary goes to stderr)
#include "pv/pv.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <utility>
#include <vector>

using namespace pv;

static Image synth(int w, int h, int c) {
    Image im(w, h, c);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * c; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / (13 * c) + y / 11) % 2) * 180 + (x * 7 + y * 3) % 60);
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
    int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    const int nops = DispatchConfig::kOps;

    std::fprintf(stderr, "calibrating (%d threads)...\n", omp_get_max_threads());
    reset_dispatch_defaults();
    DispatchConfig heur = dispatch_config();
    calibrate(5, true);
    DispatchConfig cal = dispatch_config();
    calibrate(5, false);
    DispatchConfig strict = dispatch_config();
    std::fprintf(stderr, "  %-10s %12s %12s\n", "op", "robust", "strict");
    for (int o = 0; o < nops; ++o)
        std::fprintf(stderr, "  %-10s %12ld %12ld\n", to_string(static_cast<Op>(o)), cal.omp_min_pixels[o], strict.omp_min_pixels[o]);

    // test sizes differ from the calibration grid, incl. non-square
    const std::vector<std::pair<int, int>> sizes = {
        {20, 20}, {40, 30}, {80, 60}, {120, 90}, {160, 160}, {224, 224}, {320, 240}, {448, 300},
        {640, 480}, {800, 600}, {1000, 700}, {1280, 720}, {1920, 1080}, {2560, 1440}};

    std::printf("op,width,height,pixels,seq_ms,omp_ms,oracle,heuristic,calibrated,strict\n");
    double tot_or = 0, tot_h = 0, tot_c = 0, tot_s = 0;
    int ok_h = 0, ok_c = 0, ok_s = 0, n = 0;
    for (int o = 0; o < nops; ++o) {
        Op op = static_cast<Op>(o);
        for (auto [w, h] : sizes) {
            Image rgb = synth(w, h, 3), gray = to_gray(rgb, Backend::Sequential);
            double s = best_ms(op, rgb, gray, Backend::Sequential, reps);
            double p = best_ms(op, rgb, gray, Backend::OpenMP, reps);
            double best = s < p ? s : p;
            dispatch_config() = heur;   Backend ch = choose_backend(op, w, h);
            dispatch_config() = cal;    Backend cc = choose_backend(op, w, h);
            dispatch_config() = strict; Backend cs = choose_backend(op, w, h);
            double th = ch == Backend::Sequential ? s : p;
            double tc = cc == Backend::Sequential ? s : p;
            double ts = cs == Backend::Sequential ? s : p;
            tot_or += best; tot_h += th; tot_c += tc; tot_s += ts;
            // "correct" = within 10 % of the oracle (near-ties are not mistakes)
            ok_h += th <= 1.10 * best;
            ok_c += tc <= 1.10 * best;
            ok_s += ts <= 1.10 * best;
            ++n;
            std::printf("%s,%d,%d,%ld,%.4f,%.4f,%s,%s,%s,%s\n", to_string(op), w, h, (long)w * h, s, p,
                        s <= p ? "sequential" : "openmp", to_string(ch), to_string(cc), to_string(cs));
            std::fflush(stdout);
        }
    }
    std::fprintf(stderr, "\n%-12s %10s %14s\n", "dispatcher", "accuracy", "time vs oracle");
    std::fprintf(stderr, "%-12s %9.1f%% %13.1f%%\n", "heuristic", 100.0 * ok_h / n, 100.0 * tot_h / tot_or);
    std::fprintf(stderr, "%-12s %9.1f%% %13.1f%%\n", "calibrated", 100.0 * ok_c / n, 100.0 * tot_c / tot_or);
    std::fprintf(stderr, "%-12s %9.1f%% %13.1f%%\n", "strict", 100.0 * ok_s / n, 100.0 * tot_s / tot_or);
    std::fprintf(stderr, "(accuracy = choice within 10%% of the faster backend; %d cases)\n", n);
    return 0;
}
