// Canny on synthetic images with the MPI backend (data starts and ends on rank 0).
//   mpiexec -n <P> pv_mpi_bench [reps=5]      -> CSV rows on rank 0
// Run with different -n and compare with pv_bench / pv_vs_opencv.
// Set PV_HYBRID=1 to use OpenMP inside each rank (MPI+OpenMP hybrid).
#include "pv/pv.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

static pv::Image make_image(int w, int h) {
    pv::Image im(w, h, 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * 3; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / 48 + y / 16) % 2) * 200 + (x * y) % 50);
    return im;
}

int main(int argc, char** argv) {
    pv::mpi_init(&argc, &argv);
    const int rank = pv::mpi_rank(), size = pv::mpi_size();
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const pv::Backend local = std::getenv("PV_HYBRID") ? pv::Backend::OpenMP : pv::Backend::Sequential;
    if (rank == 0) std::printf("ranks,local_backend,size,pixels,mpi_ms\n");
    for (int n : {256, 512, 1024, 2048, 4096}) {
        pv::Image in = rank == 0 ? make_image(n, n) : pv::Image();
        if (rank != 0) { in.width = n; in.height = n; in.channels = 3; }
        pv::mpi_canny(in, 50, 150, 1.4, local);  // warm-up
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
            std::printf("%d,%s,%d,%ld,%.4f\n", size, pv::to_string(local), n, (long)n * n, best);
            std::fflush(stdout);
        }
    }
    pv::mpi_finalize();
    return 0;
}
