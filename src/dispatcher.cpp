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
        case Backend::CUDA:   return cuda_available();
        default:              return false;
    }
}

DispatchConfig& dispatch_config() {
    static DispatchConfig cfg;
    return cfg;
}

void reset_dispatch_defaults() { dispatch_config() = DispatchConfig(); }

// Dispatcher: the cost of forking threads is roughly fixed while the work grows with the
// pixel count, so small images stay sequential. Only Canny has MPI and CUDA paths; a request
// for either on another operation falls back to OpenMP. MPI is never picked by Auto because
// it is collective (every process must call it).
Backend choose_backend(Op op, int width, int height, Backend requested) {
    if (requested != Backend::Auto) {
        if ((requested == Backend::MPI || requested == Backend::CUDA) && op != Op::Canny) return Backend::OpenMP;
        return backend_available(requested) ? requested : Backend::OpenMP;
    }
    const long pixels = static_cast<long>(width) * height;
    if (op == Op::Canny && cuda_available() && pixels >= dispatch_config().cuda_canny_min_pixels)
        return Backend::CUDA;
    if (omp_get_max_threads() < 2 || pixels < dispatch_config().omp_min_pixels[static_cast<int>(op)])
        return Backend::Sequential;
    return Backend::OpenMP;
}

} // namespace pv
