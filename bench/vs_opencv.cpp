// Honest baseline: pv vs OpenCV on the same Canny pipeline (gray -> blur 1.4 -> Canny 50/150).
//   pv_vs_opencv [reps=7] [optional_image_for_edge_agreement]  > vs_opencv.csv
#include "pv/pv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

static cv::Mat synth(int w, int h) {
    cv::Mat m(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * 3; ++x)
            m.ptr(y)[x] = static_cast<uint8_t>(((x / 48 + y / 16) % 2) * 200 + (x * y) % 50);
    return m;
}
static pv::Image to_pv(const cv::Mat& m) {
    pv::Image im(m.cols, m.rows, m.channels());
    for (int y = 0; y < m.rows; ++y) std::memcpy(im.row(y), m.ptr(y), (size_t)m.cols * m.channels());
    return im;
}
template <class F> static double best_ms(F&& f, int reps) {
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
static void cv_canny(const cv::Mat& src) {
    cv::Mat g, b, e;
    cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(g, b, cv::Size(0, 0), 1.4);
    cv::Canny(b, e, 50, 150, 3, true);
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    int cvThreads = cv::getNumThreads();
    std::fprintf(stderr, "OpenCV %s, %d threads\n", CV_VERSION, cvThreads);

    if (argc > 2) {   // do the two implementations find the same edges?
        cv::Mat src = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (!src.empty()) {
            cv::Mat g, b, e;
            cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(g, b, cv::Size(0, 0), 1.4);
            cv::Canny(b, e, 50, 150, 3, true);
            pv::Image pe = pv::canny(to_pv(src), 50, 150, 1.4, pv::Backend::OpenMP);
            long both = 0, a = 0, c = 0;
            for (int y = 0; y < src.rows; ++y)
                for (int x = 0; x < src.cols; ++x) {
                    bool p = pe.row(y)[x] != 0, o = e.at<uint8_t>(y, x) != 0;
                    both += p && o; a += p; c += o;
                }
            std::fprintf(stderr, "edge pixels: pv=%ld opencv=%ld  IoU=%.3f\n", a, c, (double)both / (a + c - both));
        }
    }

    const int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    std::printf("size,pixels,pv_seq_ms,pv_omp_ms,pv_auto_ms,cv_1thread_ms,cv_nthreads_ms,pv_auto_vs_cv_n\n");
    for (int n : sizes) {
        cv::Mat src = synth(n, n);
        pv::Image im = to_pv(src);
        double ps = best_ms([&] { pv::canny(im, 50, 150, 1.4, pv::Backend::Sequential); }, reps);
        double po = best_ms([&] { pv::canny(im, 50, 150, 1.4, pv::Backend::OpenMP); }, reps);
        double pa = best_ms([&] { pv::canny(im, 50, 150, 1.4, pv::Backend::Auto); }, reps);
        cv::setNumThreads(1);
        double c1 = best_ms([&] { cv_canny(src); }, reps);
        cv::setNumThreads(cvThreads);
        double cn = best_ms([&] { cv_canny(src); }, reps);
        std::printf("%d,%ld,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n", n, (long)n * n, ps, po, pa, c1, cn, cn / pa);
        std::fflush(stdout);
    }
    return 0;
}
