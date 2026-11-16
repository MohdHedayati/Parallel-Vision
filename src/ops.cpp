// Names and a uniform "run one operation" entry point for calibration and benchmarks.
#include "pv/pv.hpp"

namespace pv {

const char* to_string(Op op) {
    switch (op) {
        case Op::Grayscale:    return "grayscale";
        case Op::GaussianBlur: return "blur";
        case Op::Sobel:        return "sobel";
        case Op::Threshold:    return "threshold";
        case Op::Canny:        return "canny";
        case Op::Median:       return "median";
        case Op::Prewitt:      return "prewitt";
        case Op::Laplacian:    return "laplacian";
        case Op::HistEqualize: return "equalize";
        case Op::Erode:        return "erode";
        case Op::Dilate:       return "dilate";
    }
    return "?";
}

void run_op(Op op, const Image& rgb, const Image& gray, Backend b) {
    switch (op) {
        case Op::Grayscale:    to_gray(rgb, b); break;
        case Op::GaussianBlur: gaussian_blur(gray, 1.4, b); break;
        case Op::Sobel:        sobel(gray, b); break;
        case Op::Threshold:    threshold(gray, 128, b); break;
        case Op::Canny:        canny(rgb, 50, 150, 1.4, b); break;
        case Op::Median:       median_filter(gray, 1, b); break;
        case Op::Prewitt:      prewitt(gray, b); break;
        case Op::Laplacian:    laplacian(gray, b); break;
        case Op::HistEqualize: equalize_hist(gray, b); break;
        case Op::Erode:        erode(gray, 1, b); break;
        case Op::Dilate:       dilate(gray, 1, b); break;
    }
}

} // namespace pv
