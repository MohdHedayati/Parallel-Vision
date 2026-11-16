// Used when the library is built WITHOUT -DPV_WITH_CUDA=ON.
#include "pv/pv.hpp"
#include <stdexcept>

namespace pv {
bool cuda_available() { return false; }
Image cuda_canny(const Image&, double, double, double) {
    throw std::runtime_error("Parallel-Vision was built without CUDA (configure with -DPV_WITH_CUDA=ON)");
}
} // namespace pv
