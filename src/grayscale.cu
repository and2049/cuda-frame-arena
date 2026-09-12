#include "frame_arena/grayscale.hpp"
#include "frame_arena/cuda_error.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace frame_arena {

namespace {

// Blocks per SM for the grid-stride loop. The kernel is bound by its shared
// histogram flush (256 global atomics per block), so the grid is sized by the
// device, not by the pixel count: fewer blocks means fewer flushes.
constexpr unsigned kBlocksPerSm = 4;
constexpr unsigned kBlockThreads = 256;

__device__ __forceinline__ std::uint8_t byte(std::uint32_t word, unsigned index) {
  return static_cast<std::uint8_t>(word >> (8 * index));
}

__global__ void grayscale_kernel(const std::uint8_t* __restrict__ rgb, std::uint8_t* __restrict__ gray,
                                 std::uint32_t* histogram, std::uint32_t pixels) {
  __shared__ std::uint32_t local[kHistogramBins];
  for (unsigned i = threadIdx.x; i < kHistogramBins; i += blockDim.x) local[i] = 0;
  __syncthreads();

  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  // Four pixels are 12 input bytes and 4 output bytes: three 32-bit loads and
  // one 32-bit store per thread instead of twelve byte loads and four byte stores.
  const std::uint32_t quads = pixels / 4;
  const auto* rgb_words = reinterpret_cast<const std::uint32_t*>(rgb);
  auto* gray_words = reinterpret_cast<std::uint32_t*>(gray);
  for (std::uint32_t q = tid; q < quads; q += stride) {
    std::uint32_t w0 = rgb_words[3 * q];
    std::uint32_t w1 = rgb_words[3 * q + 1];
    std::uint32_t w2 = rgb_words[3 * q + 2];
    std::uint8_t y0 = luma(byte(w0, 0), byte(w0, 1), byte(w0, 2));
    std::uint8_t y1 = luma(byte(w0, 3), byte(w1, 0), byte(w1, 1));
    std::uint8_t y2 = luma(byte(w1, 2), byte(w1, 3), byte(w2, 0));
    std::uint8_t y3 = luma(byte(w2, 1), byte(w2, 2), byte(w2, 3));
    gray_words[q] = y0 | (y1 << 8) | (y2 << 16) | (y3 << 24);
    atomicAdd(&local[y0], 1u);
    atomicAdd(&local[y1], 1u);
    atomicAdd(&local[y2], 1u);
    atomicAdd(&local[y3], 1u);
  }
  for (std::uint32_t idx = 4 * quads + tid; idx < pixels; idx += stride) {
    const std::uint8_t* px = rgb + 3 * idx;
    std::uint8_t y = luma(px[0], px[1], px[2]);
    gray[idx] = y;
    atomicAdd(&local[y], 1u);
  }
  __syncthreads();

  for (unsigned i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    if (local[i] != 0) atomicAdd(&histogram[i], local[i]);
  }
}

unsigned multiprocessor_count() {
  static const unsigned count = [] {
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return static_cast<unsigned>(sms);
  }();
  return count;
}

bool aligned(const void* p, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

}

void launch_grayscale(const std::uint8_t* rgb, std::uint8_t* gray, std::uint32_t* histogram,
                      std::uint32_t pixels, cudaStream_t stream) {
  if (!aligned(rgb, 4) || !aligned(gray, 4)) throw std::invalid_argument("rgb and gray must be 4-byte aligned");
  unsigned work_blocks = (pixels / 4 + kBlockThreads - 1) / kBlockThreads;
  unsigned grid = std::max(1u, std::min(kBlocksPerSm * multiprocessor_count(), work_blocks));
  grayscale_kernel<<<grid, kBlockThreads, 0, stream>>>(rgb, gray, histogram, pixels);
  CUDA_CHECK(cudaGetLastError());
}

}
