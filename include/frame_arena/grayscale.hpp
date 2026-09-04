#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace frame_arena {

constexpr std::size_t kHistogramBins = 256;

struct FrameMetadata {
  std::uint32_t frame_id;
  std::uint16_t width;
  std::uint16_t height;
};

inline std::uint8_t luma(std::uint8_t r, std::uint8_t g, std::uint8_t b) noexcept {
  return static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b) >> 8);
}

void launch_grayscale(const std::uint8_t* rgb, std::uint8_t* gray, std::uint32_t* histogram,
                      std::uint32_t pixels, cudaStream_t stream);

}
