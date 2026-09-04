#pragma once

#include "frame_arena/arena.hpp"
#include "frame_arena/cuda_error.hpp"
#include "frame_arena/grayscale.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace frame_arena {

struct FrameDims {
  std::uint16_t width;
  std::uint16_t height;
  constexpr std::size_t pixels() const noexcept { return std::size_t{width} * height; }
};

struct FrameRegions {
  std::uint8_t* rgb;
  std::uint8_t* gray;
  std::uint32_t* histogram;
  FrameMetadata* metadata;
};

constexpr std::size_t kHistogramBytes = kHistogramBins * sizeof(std::uint32_t);

constexpr std::size_t frame_slot_bytes(FrameDims dims) noexcept {
  return dims.pixels() * 4 + kHistogramBytes + 256 + 128 + 256 + alignof(FrameMetadata) + sizeof(FrameMetadata);
}

inline std::optional<FrameRegions> carve_frame(Arena& arena, FrameDims dims) noexcept {
  FrameRegions r;
  if (!(r.rgb = static_cast<std::uint8_t*>(arena.allocate(dims.pixels() * 3, 256)))) return std::nullopt;
  if (!(r.gray = static_cast<std::uint8_t*>(arena.allocate(dims.pixels(), 128)))) return std::nullopt;
  if (!(r.histogram = static_cast<std::uint32_t*>(arena.allocate(kHistogramBytes, 256)))) return std::nullopt;
  if (!(r.metadata = arena.allocate<FrameMetadata>())) return std::nullopt;
  return r;
}

inline void fill_synthetic_rgb(std::uint8_t* rgb, FrameDims dims, std::uint32_t frame_id) noexcept {
  std::uint32_t state = frame_id * 2654435761u + 1u;
  for (std::size_t i = 0; i < dims.pixels() * 3; ++i) {
    state = state * 1664525u + 1013904223u;
    rgb[i] = static_cast<std::uint8_t>(state >> 24);
  }
}

inline void enqueue_frame(const FrameRegions& host, const FrameRegions& device, FrameDims dims,
                          cudaStream_t stream) {
  std::size_t pixels = dims.pixels();
  CUDA_CHECK(cudaMemcpyAsync(device.rgb, host.rgb, pixels * 3, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(device.histogram, 0, kHistogramBytes, stream));
  launch_grayscale(device.rgb, device.gray, device.histogram, static_cast<std::uint32_t>(pixels), stream);
  CUDA_CHECK(cudaMemcpyAsync(host.gray, device.gray, pixels, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(host.histogram, device.histogram, kHistogramBytes, cudaMemcpyDeviceToHost, stream));
}

inline bool verify_grayscale(const FrameRegions& host, FrameDims dims) noexcept {
  std::vector<std::uint32_t> expected(kHistogramBins, 0);
  for (std::size_t i = 0; i < dims.pixels(); ++i) {
    std::uint8_t y = luma(host.rgb[3 * i], host.rgb[3 * i + 1], host.rgb[3 * i + 2]);
    if (host.gray[i] != y) return false;
    ++expected[y];
  }
  for (std::size_t bin = 0; bin < kHistogramBins; ++bin) {
    if (host.histogram[bin] != expected[bin]) return false;
  }
  return true;
}

}
