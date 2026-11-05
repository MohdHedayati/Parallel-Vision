// pv — command-line front end. OpenCV is used only for reading/writing files.
//   pv <canny|sobel|blur|gray|threshold> in out [--backend auto|seq|omp]
//        [--low N] [--high N] [--sigma S] [--t N|otsu]
#include "pv/pv.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static pv::Image from_mat(const cv::Mat& m) {
    pv::Image im(m.cols, m.rows, m.channels());
    for (int y = 0; y < m.rows; ++y)
        std::memcpy(im.row(y), m.ptr(y), static_cast<size_t>(m.cols) * m.channels());
    return im;
}

static cv::Mat to_mat(const pv::Image& im) {
    cv::Mat m(im.height, im.width, im.channels == 1 ? CV_8UC1 : CV_8UC3);
    for (int y = 0; y < im.height; ++y)
        std::memcpy(m.ptr(y), im.row(y), static_cast<size_t>(im.width) * im.channels);
    return m;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: pv <canny|sobel|blur|gray|threshold> in out [--backend auto|seq|omp]\n"
                    "          [--low N] [--high N] [--sigma S] [--t N|otsu]\n"
                    "          [--calibrate | --use-calibration]\n");
        return 1;
    }
    std::string cmd = argv[1];
    pv::Backend backend = pv::Backend::Auto;
    double low = 50, high = 150, sigma = 1.4;
    int thr = -1;  // -1 = otsu
    for (int i = 4; i < argc; ) {
        std::string k = argv[i];
        if (k == "--calibrate") {   // measure switch points now and cache them
            pv::calibrate();
            pv::save_calibration("pv_calibration.txt");
            ++i; continue;
        }
        if (k == "--use-calibration") {   // load cached switch points if present
            if (!pv::load_calibration("pv_calibration.txt"))
                std::fprintf(stderr, "no pv_calibration.txt, using heuristic (run with --calibrate first)\n");
            ++i; continue;
        }
        if (i + 1 >= argc) break;
        std::string v = argv[i + 1];
        i += 2;
        if (k == "--backend") backend = v == "seq" ? pv::Backend::Sequential : v == "omp" ? pv::Backend::OpenMP : pv::Backend::Auto;
        else if (k == "--low") low = std::atof(v.c_str());
        else if (k == "--high") high = std::atof(v.c_str());
        else if (k == "--sigma") sigma = std::atof(v.c_str());
        else if (k == "--t") thr = v == "otsu" ? -1 : std::atoi(v.c_str());
    }

    cv::Mat src = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (src.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    pv::Image in = from_mat(src), out;

    pv::Backend used = pv::choose_backend(pv::Op::Canny, in.width, in.height, backend);
    auto t0 = std::chrono::steady_clock::now();
    if (cmd == "canny") out = pv::canny(in, low, high, sigma, backend);
    else if (cmd == "sobel") out = pv::sobel(pv::to_gray(in, backend), backend);
    else if (cmd == "blur") out = pv::gaussian_blur(pv::to_gray(in, backend), sigma, backend);
    else if (cmd == "gray") out = pv::to_gray(in, backend);
    else if (cmd == "threshold") {
        pv::Image g = pv::to_gray(in, backend);
        out = pv::threshold(g, thr < 0 ? pv::otsu_threshold(g) : thr, backend);
    } else { std::fprintf(stderr, "unknown command %s\n", cmd.c_str()); return 1; }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (!cv::imwrite(argv[3], to_mat(out))) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    std::printf("%s %dx%d  backend=%s  %.2f ms\n", cmd.c_str(), in.width, in.height, pv::to_string(used), ms);
    return 0;
}
