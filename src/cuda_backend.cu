// CUDA backend for Canny (compiled only with -DPV_WITH_CUDA=ON).
//
// STATUS: written to mirror the CPU code operation-for-operation (same tap order, same
// direction rules; built with --fmad=false so floating-point rounding matches), but it has
// NOT been run: the development machine has no NVIDIA GPU. Run tests/test_cuda.cpp on a
// CUDA machine (Linux, Windows+MSVC, or Google Colab) to check it is bit-identical.
//
// Pipeline on the GPU: gray -> blur (horizontal, vertical) -> gradient -> non-max suppression.
// Hysteresis needs global connectivity and only touches edge pixels, so it stays on the CPU.
#include "internal.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace pv {
namespace {

#define PV_CUDA_CHECK(call)                                                                      \
    do {                                                                                         \
        cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess)                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_));    \
    } while (0)

// RAII device buffer so every early exit / exception frees GPU memory.
template <class T>
struct DevBuf {
    T* p = nullptr;
    explicit DevBuf(size_t n) { PV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

__device__ __forceinline__ int dclamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ unsigned char dclamp_u8(float v) {
    return v < 0 ? 0 : (v > 255 ? 255 : static_cast<unsigned char>(v + 0.5f));
}

// BGR (or already gray) -> gray, same integer formula as the CPU.
__global__ void k_gray(const unsigned char* in, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned char* s = in + (static_cast<size_t>(y) * w + x) * c;
    out[static_cast<size_t>(y) * w + x] =
        c == 1 ? s[0] : static_cast<unsigned char>((114 * s[0] + 587 * s[1] + 299 * s[2] + 500) / 1000);
}

// Horizontal blur pass: taps accumulated in ascending order starting from 0, like the CPU.
__global__ void k_blur_h(const unsigned char* gray, float* tmp, const float* k, int r, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned char* row = gray + static_cast<size_t>(y) * w;
    float acc = 0.f;
    for (int i = 0; i <= 2 * r; ++i) acc += k[i] * static_cast<float>(row[dclamp(x + i - r, 0, w - 1)]);
    tmp[static_cast<size_t>(y) * w + x] = acc;
}

__global__ void k_blur_v(const float* tmp, unsigned char* out, const float* k, int r, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float acc = 0.f;
    for (int i = 0; i <= 2 * r; ++i) acc += k[i] * tmp[static_cast<size_t>(dclamp(y + i - r, 0, h - 1)) * w + x];
    out[static_cast<size_t>(y) * w + x] = dclamp_u8(acc);
}

// Sobel gradient: magnitude and quantised direction (0 horiz, 1 '\', 2 vert, 3 '/').
// Pixels with magnitude below `cut` are zeroed (they can never become edges).
__global__ void k_gradient(const unsigned char* g, float* mag, unsigned char* dir, int w, int h, float cut2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned char* r0 = g + static_cast<size_t>(dclamp(y - 1, 0, h - 1)) * w;
    const unsigned char* r1 = g + static_cast<size_t>(y) * w;
    const unsigned char* r2 = g + static_cast<size_t>(dclamp(y + 1, 0, h - 1)) * w;
    int xl = dclamp(x - 1, 0, w - 1), xr = dclamp(x + 1, 0, w - 1);
    float gx = static_cast<float>(-r0[xl] + r0[xr] - 2 * r1[xl] + 2 * r1[xr] - r2[xl] + r2[xr]);
    float gy = static_cast<float>(-r0[xl] - 2 * r0[x] - r0[xr] + r2[xl] + 2 * r2[x] + r2[xr]);
    size_t i = static_cast<size_t>(y) * w + x;
    float s2 = gx * gx + gy * gy;
    mag[i] = sqrtf(s2);
    if (s2 < cut2) { mag[i] = 0.f; dir[i] = 0; return; }
    float ax = fabsf(gx), ay = fabsf(gy);
    dir[i] = ay <= 0.41421356f * ax ? 0 : ay >= 2.41421356f * ax ? 2 : (gx * gy > 0 ? 1 : 3);
}

// Non-maximum suppression -> labels 0 none / 1 weak / 2 strong.
__global__ void k_nms(const float* mag, const unsigned char* dir, unsigned char* lab, int w, int h, float hi) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t i = static_cast<size_t>(y) * w + x;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) { lab[i] = 0; return; }
    float m = mag[i];
    if (m == 0.f) { lab[i] = 0; return; }
    const int dxs[4] = {1, 1, 0, 1}, dys[4] = {0, 1, 1, -1};
    int d = dir[i];
    float a = mag[static_cast<size_t>(y + dys[d]) * w + (x + dxs[d])];
    float c = mag[static_cast<size_t>(y - dys[d]) * w + (x - dxs[d])];
    lab[i] = (m >= a && m >= c) ? (m >= hi ? 2 : 1) : 0;
}

} // namespace

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

Image cuda_canny(const Image& in, double low, double high, double sigma) {
    if (in.empty()) return Image();
    const int w = in.width, h = in.height, c = in.channels;
    const size_t npx = static_cast<size_t>(w) * h;

    // Gaussian taps computed exactly like the CPU (filters.cpp).
    int r = static_cast<int>(std::ceil(3 * sigma));
    if (r < 1) r = 1;
    std::vector<float> k(2 * r + 1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = std::exp(-(i * i) / (2 * sigma * sigma));
    for (auto& v : k) v /= sum;

    DevBuf<unsigned char> d_in(npx * c), d_gray(npx), d_blur(npx), d_dir(npx), d_lab(npx);
    DevBuf<float> d_tmp(npx), d_mag(npx), d_k(k.size());
    PV_CUDA_CHECK(cudaMemcpy(d_in.p, in.data.data(), npx * c, cudaMemcpyHostToDevice));
    PV_CUDA_CHECK(cudaMemcpy(d_k.p, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    k_gray<<<grid, block>>>(d_in.p, d_gray.p, w, h, c);
    k_blur_h<<<grid, block>>>(d_gray.p, d_tmp.p, d_k.p, r, w, h);
    k_blur_v<<<grid, block>>>(d_tmp.p, d_blur.p, d_k.p, r, w, h);
    const float cut = static_cast<float>(low);
    k_gradient<<<grid, block>>>(d_blur.p, d_mag.p, d_dir.p, w, h, cut * cut);
    k_nms<<<grid, block>>>(d_mag.p, d_dir.p, d_lab.p, w, h, static_cast<float>(high));
    PV_CUDA_CHECK(cudaGetLastError());

    std::vector<unsigned char> lab(npx);
    PV_CUDA_CHECK(cudaMemcpy(lab.data(), d_lab.p, npx, cudaMemcpyDeviceToHost));   // also synchronises
    return detail::hysteresis_from_labels(lab.data(), w, h, Backend::OpenMP);
}

} // namespace pv
