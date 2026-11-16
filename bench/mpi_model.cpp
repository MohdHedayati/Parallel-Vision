// Cost model for the MPI backend.
//   mpiexec -n <P> pv_mpi_model [reps=3]      -> CSV on rank 0
//
// Fits time(N) = a + b*N for sequential, OpenMP and MPI Canny from sizes 1024..3072, then checks the
// model on LARGER sizes it was not fitted on (4096, 5120), and reports the image size at which MPI would
// start to beat both single-process backends (-1 = never, on this machine).
// Set PV_HYBRID=1 to use OpenMP inside every rank.
#include "pv/pv.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

static pv::Image make_image(int n) {
    pv::Image im(n, n, 3);
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
    pv::mpi_init(&argc, &argv);
    const int rank = pv::mpi_rank(), size = pv::mpi_size();
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3;
    const pv::Backend local = std::getenv("PV_HYBRID") ? pv::Backend::OpenMP : pv::Backend::Sequential;

    // The single-process OpenMP baseline must use ALL hardware threads. In hybrid mode the ranks are
    // started with OMP_NUM_THREADS = threads per rank, which would otherwise handicap the baseline.
    const int per_rank_threads = omp_get_max_threads();
    const int all_threads = omp_get_num_procs();

    // single-process backends are measured on rank 0 while the others sleep (not spin) in a barrier
    pv::LinearCost seq, omp;
    if (rank == 0) {
        omp_set_num_threads(all_threads);
        seq = pv::fit_cost(pv::Backend::Sequential, reps);
        omp = pv::fit_cost(pv::Backend::OpenMP, reps);
    }
    pv::mpi_sleep_barrier();
    omp_set_num_threads(per_rank_threads);
    pv::LinearCost mpi = pv::fit_mpi_cost(local, reps);                 // collective

    const double cross = pv::mpi_crossover_pixels(seq, omp, mpi);
    if (rank == 0) std::printf("ranks,local,series,size,pixels,predicted_ms,measured_ms,a_ms,b_ns_per_px,mpi_crossover_px\n");

    for (int n : {2048, 3072, 4096, 5120}) {                            // 4096 and 5120 are larger than any fitted size
        pv::Image in = rank == 0 ? make_image(n) : pv::Image();
        if (rank != 0) { in.width = n; in.height = n; in.channels = 3; }
        double px = static_cast<double>(n) * n;

        double m_seq = 0, m_omp = 0;
        if (rank == 0) {
            omp_set_num_threads(all_threads);
            m_seq = best_ms([&] { pv::canny(in, 50, 150, 1.4, pv::Backend::Sequential); }, reps);
            m_omp = best_ms([&] { pv::canny(in, 50, 150, 1.4, pv::Backend::OpenMP); }, reps);
        }
        pv::mpi_sleep_barrier();
        omp_set_num_threads(per_rank_threads);

        pv::mpi_canny(in, 50, 150, 1.4, local);                          // warm-up
        double best = 1e30;
        for (int i = 0; i < reps; ++i) {
            pv::mpi_barrier();
            auto t0 = std::chrono::steady_clock::now();
            pv::mpi_canny(in, 50, 150, 1.4, local);
            pv::mpi_barrier();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms < best) best = ms;
        }
        if (rank == 0) {
            auto row = [&](const char* name, const pv::LinearCost& c, double measured) {
                std::printf("%d,%s,%s,%d,%.0f,%.3f,%.3f,%.4f,%.4f,%.0f\n", size, pv::to_string(local), name, n, px,
                            c.at(px), measured, c.a_ms, c.b_ns_per_px, cross);
            };
            row("sequential", seq, m_seq);
            row("openmp", omp, m_omp);
            row("mpi", mpi, best);
            std::fflush(stdout);
        }
    }
    pv::mpi_finalize();
    return 0;
}
