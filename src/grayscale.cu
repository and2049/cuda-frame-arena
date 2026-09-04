#include "frame_arena/grayscale.hpp"
#include "frame_arena/cuda_error.hpp"

namespace frame_arena {

namespace {

__global__ void grayscale_kernel(const std::uint8_t* rgb, std::uint8_t* gray,
                                 std::uint32_t* histogram, std::uint32_t pixels) {
  __shared__ std::uint32_t local[kHistogramBins];
  for (unsigned i = threadIdx.x; i < kHistogramBins; i += blockDim.x) local[i] = 0;
  __syncthreads();

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pixels) {
    const std::uint8_t* px = rgb + 3u * idx;
    std::uint8_t y = (77u * px[0] + 150u * px[1] + 29u * px[2]) >> 8;
    gray[idx] = y;
    atomicAdd(&local[y], 1u);
  }
  __syncthreads();

  for (unsigned i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    if (local[i] != 0) atomicAdd(&histogram[i], local[i]);
  }
}

}

void launch_grayscale(const std::uint8_t* rgb, std::uint8_t* gray, std::uint32_t* histogram,
                      std::uint32_t pixels, cudaStream_t stream) {
  constexpr unsigned block = 256;
  unsigned grid = (pixels + block - 1) / block;
  grayscale_kernel<<<grid, block, 0, stream>>>(rgb, gray, histogram, pixels);
  CUDA_CHECK(cudaGetLastError());
}

}
