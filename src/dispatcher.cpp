#include "pv/pv.hpp"
#include <omp.h>

namespace pv {

const char* to_string(Backend b) {
    switch (b) {
        case Backend::Auto:       return "auto";
        case Backend::Sequential: return "sequential";
        case Backend::OpenMP:     return "openmp";
        case Backend::MPI:        return "mpi";
        case Backend::CUDA:       return "cuda";
    }
    return "?";
}

bool backend_available(Backend b) {
    switch (b) {
        case Backend::Auto:
        case Backend::Sequential:
        case Backend::OpenMP: return true;
#ifdef PV_HAVE_MPI
        case Backend::MPI:    return true;
#endif
        default:              return false;
    }
}

DispatchConfig& dispatch_config() {
    static DispatchConfig cfg;
    return cfg;
}

// Heuristic dispatcher: cost of forking threads is roughly fixed, work grows
// with pixel count, so small images stay sequential.
void reset_dispatch_defaults() { dispatch_config() = DispatchConfig(); }

Backend choose_backend(Op op, int width, int height, Backend requested) {
    if (requested != Backend::Auto)
        return backend_available(requested) ? requested : Backend::OpenMP;
    long pixels = static_cast<long>(width) * height;
    if (omp_get_max_threads() < 2 || pixels < dispatch_config().omp_min_pixels[static_cast<int>(op)])
        return Backend::Sequential;
    return Backend::OpenMP;
}

} // namespace pv
