// Cost model: time(N) ~ a + b*N per backend, fitted from measurements, and the crossover
// where the distributed (MPI) backend would start to pay off.
#include "internal.hpp"
#include <chrono>
#include <cmath>

namespace pv {

LinearCost fit_cost(Backend b, int reps) {
    static const int kSides[] = {1024, 1536, 2048, 2560, 3072};   // large enough to be past cache effects
    std::vector<double> px, ms;
    for (int side : kSides) {
        Image im(side, side, 3);
        for (int y = 0; y < side; ++y)
            for (int x = 0; x < side * 3; ++x)
                im.row(y)[x] = static_cast<uint8_t>(((x / 48 + y / 16) % 2) * 200 + (x * y) % 50);
        canny(im, 50, 150, 1.4, b);   // warm-up
        double best = 1e30;
        for (int i = 0; i < reps; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            canny(im, 50, 150, 1.4, b);
            double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (t < best) best = t;
        }
        px.push_back(static_cast<double>(side) * side);
        ms.push_back(best);
    }
    return detail::linear_fit(px, ms);
}

double mpi_crossover_pixels(const LinearCost& seq, const LinearCost& omp, const LinearCost& mpi) {
    // MPI "pays off" from size N upward if it stays faster than the best single-process backend
    // for every larger image. Scan down from a huge image and stop at the first size where it
    // is no longer ahead; if it is not ahead even at 10^10 pixels it never pays off.
    auto ahead = [&](double n) { return mpi.at(n) < std::fmin(seq.at(n), omp.at(n)); };
    if (!ahead(1e10)) return -1;
    double n = 1e10;
    while (n > 1e3 && ahead(n / 1.02)) n /= 1.02;
    return n;
}

} // namespace pv
