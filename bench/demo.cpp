// Fast live demo (about 20 seconds). Prints three tables and writes the data behind them as CSV.
//   pv_demo [output_dir=results/demo]
//   1. Correctness : every operation gives the SAME image sequentially and with OpenMP
//   2. Dispatcher  : Backend::Auto versus the fastest backend, small to large images
//   3. Speedup     : all operations on a 2048 x 2048 image
#include "pv/pv.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <string>

using namespace pv;

static Image synth(int w, int h, int c, unsigned seed = 1) {
    Image im(w, h, c);
    srand(seed);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * c; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / (16 * c) + y / 16) % 2) * 170 + rand() % 60);
    return im;
}

static Image apply(Op op, const Image& rgb, const Image& gray, Backend b) {
    switch (op) {
        case Op::Grayscale:    return to_gray(rgb, b);
        case Op::GaussianBlur: return gaussian_blur(gray, 1.4, b);
        case Op::Sobel:        return sobel(gray, b);
        case Op::Threshold:    return threshold(gray, 128, b);
        case Op::Canny:        return canny(rgb, 50, 150, 1.4, b);
        case Op::Median:       return median_filter(gray, 1, b);
        case Op::Prewitt:      return prewitt(gray, b);
        case Op::Laplacian:    return laplacian(gray, b);
        case Op::HistEqualize: return equalize_hist(gray, b);
        case Op::Erode:        return erode(gray, 1, b);
        case Op::Dilate:       return dilate(gray, 1, b);
    }
    return Image();
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

static std::string bar(double v, double vmax, int width = 24) {
    int n = vmax > 0 ? static_cast<int>(v / vmax * width + 0.5) : 0;
    return std::string(std::max(0, std::min(width, n)), '#');
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "results/demo";
    const int threads = omp_get_max_threads();
    std::printf("\n=== Parallel-Vision live demo  (%d OpenMP threads; MPI %s; CUDA %s) ===\n", threads,
                backend_available(Backend::MPI) ? "built" : "not built", cuda_available() ? "available" : "not available");

    // ------------------------------------------------------------------ 1. correctness
    std::printf("\n[1] Correctness: sequential vs OpenMP output on a 301x203 image\n");
    {
        Image rgb = synth(301, 203, 3), gray = to_gray(rgb, Backend::Sequential);
        int pass = 0;
        for (int o = 0; o < DispatchConfig::kOps; ++o) {
            Op op = static_cast<Op>(o);
            bool same = apply(op, rgb, gray, Backend::Sequential).data == apply(op, rgb, gray, Backend::OpenMP).data;
            pass += same;
            std::printf("    %-10s %s\n", to_string(op), same ? "identical" : "DIFFERENT");
        }
        std::printf("    -> %d / %d operations identical\n", pass, DispatchConfig::kOps);
    }

    // ------------------------------------------------------------------ 2. dispatcher
    reset_dispatch_defaults();
    const DispatchConfig heur = dispatch_config();
    std::printf("\n[2] Calibrating this machine (measures where OpenMP starts to win) ...\n");
    auto tc0 = std::chrono::steady_clock::now();
    calibrate(3, true);
    const DispatchConfig cal = dispatch_config();
    std::printf("    done in %.1f s. Switch points (pixels): ", std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count());
    for (Op op : {Op::Canny, Op::GaussianBlur, Op::Median, Op::Threshold})
        std::printf("%s=%ld  ", to_string(op), cal.omp_min_pixels[static_cast<int>(op)]);
    std::printf("(heuristic: %ld for all)\n", heur.omp_min_pixels[0]);

    std::printf("\n    Backend::Auto vs the best backend (best of 3 runs); 'ok' = within 10%% of the fastest\n");
    std::printf("    %-9s %-10s %9s %9s   %-10s %-3s  %-10s %-3s\n", "op", "size", "seq ms", "omp ms", "heuristic", "ok", "calibrated", "ok");
    FILE* fd = std::fopen((out + "/demo_dispatch.csv").c_str(), "w");
    if (fd) std::fprintf(fd, "op,size,pixels,seq_ms,omp_ms,fastest,heuristic,heuristic_ok,calibrated,calibrated_ok\n");
    int right_h = 0, right_c = 0, total = 0;
    for (Op op : {Op::Canny, Op::GaussianBlur, Op::Median, Op::Threshold}) {
        for (int n : {32, 64, 128, 256, 512, 1024, 2048}) {
            Image rgb = synth(n, n, 3), gray = to_gray(rgb, Backend::Sequential);
            double s = best_ms([&] { run_op(op, rgb, gray, Backend::Sequential); }, 3);
            double p = best_ms([&] { run_op(op, rgb, gray, Backend::OpenMP); }, 3);
            dispatch_config() = heur; Backend ch = choose_backend(op, n, n);
            dispatch_config() = cal;  Backend cc = choose_backend(op, n, n);
            double best = std::min(s, p);
            bool okh = (ch == Backend::Sequential ? s : p) <= 1.10 * best;
            bool okc = (cc == Backend::Sequential ? s : p) <= 1.10 * best;
            right_h += okh; right_c += okc; ++total;
            std::printf("    %-9s %4dx%-5d %9.3f %9.3f   %-10s %-3s  %-10s %-3s\n", to_string(op), n, n, s, p,
                        to_string(ch), okh ? "yes" : "NO", to_string(cc), okc ? "yes" : "NO");
            if (fd) std::fprintf(fd, "%s,%d,%ld,%.4f,%.4f,%s,%s,%d,%s,%d\n", to_string(op), n, (long)n * n, s, p,
                                 s <= p ? "sequential" : "openmp", to_string(ch), okh, to_string(cc), okc);
        }
    }
    if (fd) std::fclose(fd);
    std::printf("    -> within 10%% of the best backend: heuristic %d / %d, calibrated %d / %d cases\n", right_h, total, right_c, total);

    // ------------------------------------------------------------------ 3. speedups
    std::printf("\n[3] Speedup on a 2048x2048 image, %d threads\n", threads);
    std::printf("    %-10s %10s %10s %8s\n", "op", "seq ms", "omp ms", "speedup");
    FILE* fo = std::fopen((out + "/demo_ops.csv").c_str(), "w");
    if (fo) std::fprintf(fo, "op,seq_ms,omp_ms,speedup\n");
    Image rgb = synth(2048, 2048, 3), gray = to_gray(rgb, Backend::Sequential);
    double smax = 0;
    struct Row { Op op; double s, p; };
    std::vector<Row> rows;
    for (int o = 0; o < DispatchConfig::kOps; ++o) {
        Op op = static_cast<Op>(o);
        double s = best_ms([&] { run_op(op, rgb, gray, Backend::Sequential); }, 3);
        double p = best_ms([&] { run_op(op, rgb, gray, Backend::OpenMP); }, 3);
        rows.push_back({op, s, p});
        smax = std::max(smax, s / p);
    }
    for (auto& r : rows) {
        std::printf("    %-10s %10.2f %10.2f %7.2fx  %s\n", to_string(r.op), r.s, r.p, r.s / r.p, bar(r.s / r.p, smax).c_str());
        if (fo) std::fprintf(fo, "%s,%.4f,%.4f,%.3f\n", to_string(r.op), r.s, r.p, r.s / r.p);
    }
    if (fo) std::fclose(fo);
    std::printf("\nData written to %s/*.csv\n", out.c_str());
    return 0;
}
