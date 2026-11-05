// MPI backend: row-strip decomposition with halo exchange. See pv.hpp for the contract.
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef PV_HAVE_MPI
#include <mpi.h>

namespace pv {

bool mpi_init(int* argc, char*** argv) {
    int f = 0;
    MPI_Initialized(&f);
    if (!f) MPI_Init(argc, argv);
    return true;
}
void mpi_finalize() {
    int f = 0;
    MPI_Finalized(&f);
    if (!f) MPI_Finalize();
}
int mpi_rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
int mpi_size() { int n = 1; MPI_Comm_size(MPI_COMM_WORLD, &n); return n; }
void mpi_barrier() { MPI_Barrier(MPI_COMM_WORLD); }

Image mpi_canny(const Image& in, double low, double high, double sigma, Backend local) {
    const int rank = mpi_rank(), size = mpi_size();
    const MPI_Comm comm = MPI_COMM_WORLD;

    int dims[3] = {in.width, in.height, in.channels};          // only rank 0's values are meaningful
    MPI_Bcast(dims, 3, MPI_INT, 0, comm);
    const int w = dims[0], h = dims[1], c = dims[2];

    // Halo depth: blur radius + 1 (gradient) + 1 (non-max suppression). With exactly this much
    // context every owned row is computed from the same data as in the sequential run.
    const int r = std::max(1, static_cast<int>(std::ceil(3 * sigma)));
    const int depth = r + 2;

    // A strip must hold at least `depth` rows so halos only ever come from the adjacent rank.
    const int active = std::max(1, std::min(size, h / depth));
    std::vector<int> rows(size, 0), start(size, 0);
    for (int i = 0, y = 0; i < active; ++i) {
        rows[i] = h / active + (i < h % active ? 1 : 0);
        start[i] = y;
        y += rows[i];
    }
    const int my_rows = rows[rank];

    // 1. scatter colour strips from rank 0
    std::vector<int> cnt(size), dsp(size);
    for (int i = 0; i < size; ++i) { cnt[i] = rows[i] * w * c; dsp[i] = start[i] * w * c; }
    Image strip(w, my_rows, c);
    MPI_Scatterv(rank == 0 ? in.data.data() : nullptr, cnt.data(), dsp.data(), MPI_BYTE,
                 strip.data.data(), cnt[rank], MPI_BYTE, 0, comm);

    std::vector<uint8_t> own_labels;                            // labels for this rank's rows
    if (my_rows > 0) {
        // 2. grayscale on the owned rows (pointwise, no halo needed)
        Image gray = to_gray(strip, local);

        // 3. halo exchange (gray, 1 byte/pixel): extended strip = top halo | own rows | bottom halo
        const bool has_up = rank > 0, has_down = rank < active - 1;
        const int top = has_up ? depth : 0, bot = has_down ? depth : 0;
        Image ext(w, top + my_rows + bot, 1);
        std::copy(gray.data.begin(), gray.data.end(), ext.row(top));
        const int up = has_up ? rank - 1 : MPI_PROC_NULL, down = has_down ? rank + 1 : MPI_PROC_NULL;
        const int n = depth * w;
        // my first rows -> upper neighbour's bottom halo; my bottom halo <- lower neighbour's first rows
        MPI_Sendrecv(gray.row(0), has_up ? n : 0, MPI_BYTE, up, 10,
                     ext.row(top + my_rows), has_down ? n : 0, MPI_BYTE, down, 10, comm, MPI_STATUS_IGNORE);
        // my last rows -> lower neighbour's top halo; my top halo <- upper neighbour's last rows
        MPI_Sendrecv(gray.row(std::max(0, my_rows - depth)), has_down ? n : 0, MPI_BYTE, down, 11,
                     ext.row(0), has_up ? n : 0, MPI_BYTE, up, 11, comm, MPI_STATUS_IGNORE);

        // 4. blur -> gradient -> NMS on the extended strip; keep only the owned rows
        Buf<uint8_t> lab = detail::canny_labels(ext, low, high, sigma, local);
        own_labels.assign(lab.data() + static_cast<size_t>(top) * w,
                          lab.data() + static_cast<size_t>(top + my_rows) * w);
    }

    // 5. gather labels on rank 0; hysteresis needs global connectivity so it runs there
    std::vector<int> gcnt(size), gdsp(size);
    for (int i = 0; i < size; ++i) { gcnt[i] = rows[i] * w; gdsp[i] = start[i] * w; }
    std::vector<uint8_t> all;
    if (rank == 0) all.resize(static_cast<size_t>(w) * h);
    MPI_Gatherv(own_labels.data(), gcnt[rank], MPI_BYTE, all.data(), gcnt.data(), gdsp.data(), MPI_BYTE, 0, comm);
    if (rank != 0) return Image();
    return detail::hysteresis_from_labels(all.data(), w, h,
                                          local == Backend::OpenMP ? Backend::OpenMP : Backend::Sequential);
}

} // namespace pv

#else  // built without MPI

namespace pv {
bool  mpi_init(int*, char***) { return false; }
void  mpi_finalize() {}
int   mpi_rank() { return 0; }
int   mpi_size() { return 1; }
void  mpi_barrier() {}
Image mpi_canny(const Image&, double, double, double, Backend) {
    throw std::runtime_error("Parallel-Vision was built without MPI (configure with -DPV_WITH_MPI=ON)");
}
} // namespace pv

#endif
