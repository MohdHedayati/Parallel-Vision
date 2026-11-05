// Run under MPI:  mpiexec -n 4 pv_mpi_tests        (set PV_HYBRID=1 to use OpenMP inside each rank)
// Checks that the distributed Canny is bit-identical to the sequential one for several
// image shapes (including heights too small to give every rank a full halo).
#include "pv/pv.hpp"
#include <cstdio>
#include <cstdlib>

static pv::Image make(int w, int h, int c, unsigned seed) {
    pv::Image im(w, h, c);
    srand(seed);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * c; ++x)
            im.row(y)[x] = static_cast<uint8_t>(((x / (9 * c) + y / 7) % 2) * 160 + rand() % 60);
    return im;
}

int main(int argc, char** argv) {
    pv::mpi_init(&argc, &argv);
    const int rank = pv::mpi_rank(), size = pv::mpi_size();
    int failures = 0;
    const pv::Backend local = std::getenv("PV_HYBRID") ? pv::Backend::OpenMP : pv::Backend::Sequential;
    const int shapes[][2] = {{64, 64}, {257, 131}, {100, 9}, {300, 500}, {33, 1000}, {16, 16}};
    for (auto& s : shapes) {
        pv::Image in = rank == 0 ? make(s[0], s[1], 3, 7) : pv::Image();
        if (rank != 0) { in.width = s[0]; in.height = s[1]; in.channels = 3; }   // ignored off-root
        pv::Image got = pv::mpi_canny(in, 50, 150, 1.4, local);
        if (rank == 0) {
            pv::Image ref = pv::canny(in, 50, 150, 1.4, pv::Backend::Sequential);
            bool ok = got.data == ref.data;
            long n = 0;
            for (auto v : ref.data) n += v == 255;
            std::printf("%dx%d on %d ranks: %s (%ld edge px)\n", s[0], s[1], size, ok ? "identical" : "MISMATCH", n);
            failures += !ok;
        }
    }
    pv::mpi_finalize();
    if (rank == 0) std::printf(failures ? "MPI tests FAILED\n" : "MPI tests passed\n");
    return failures ? 1 : 0;
}
