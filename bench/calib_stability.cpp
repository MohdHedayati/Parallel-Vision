// How repeatable is calibration? Run it several times with each rule and print every switch point.
// A good rule gives nearly the same answer every time.
//   pv_calib_stability [runs=5] > calib_stability.csv
#include "pv/pv.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const int runs = argc > 1 ? std::atoi(argv[1]) : 5;
    std::printf("rule,run,op,switch_px\n");
    for (bool robust : {false, true}) {
        for (int r = 0; r < runs; ++r) {
            pv::reset_dispatch_defaults();
            pv::calibrate(5, robust);
            for (int o = 0; o < pv::DispatchConfig::kOps; ++o)
                std::printf("%s,%d,%s,%ld\n", robust ? "robust" : "strict", r, pv::to_string(static_cast<pv::Op>(o)),
                            pv::dispatch_config().omp_min_pixels[o]);
            std::fflush(stdout);
            std::fprintf(stderr, "%s run %d/%d done\n", robust ? "robust" : "strict", r + 1, runs);
        }
    }
    return 0;
}
