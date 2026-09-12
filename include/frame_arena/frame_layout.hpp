#pragma once

#include "frame_arena/arena.hpp"
#include "frame_arena/cuda_error.hpp"
#include "frame_arena/grayscale.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
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

inline std::uint32_t synthetic_word(std::uint32_t index, std::uint32_t frame_id) noexcept {
  std::uint32_t word = (index + frame_id * 2654435761u + 1u) * 2246822519u;
  return word ^ (word >> 15);
}

inline std::uint8_t synthetic_rgb_byte(std::size_t index, std::uint32_t frame_id) noexcept {
  std::uint32_t word = synthetic_word(static_cast<std::uint32_t>(index / 4), frame_id);
  return static_cast<std::uint8_t>(word >> (8 * (index % 4)));
}

inline void fill_synthetic_rgb(std::uint8_t* rgb, FrameDims dims, std::uint32_t frame_id) noexcept {
  std::size_t bytes = dims.pixels() * 3;
  std::size_t words = bytes / 4;
  for (std::size_t i = 0; i < words; ++i) {
    std::uint32_t word = synthetic_word(static_cast<std::uint32_t>(i), frame_id);
    std::memcpy(rgb + 4 * i, &word, 4);
  }
  for (std::size_t i = 4 * words; i < bytes; ++i) rgb[i] = synthetic_rgb_byte(i, frame_id);
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

// Recomputes the expected output from the synthetic generator rather than
// reading host.rgb back: the input region is written by the CPU and read by
// the copy engine, and nothing in the pipeline should need to read it again.
inline bool verify_grayscale(const FrameRegions& host, FrameDims dims, std::uint32_t frame_id) noexcept {
  std::vector<std::uint32_t> expected(kHistogramBins, 0);
  for (std::size_t i = 0; i < dims.pixels(); ++i) {
    std::uint8_t y = luma(synthetic_rgb_byte(3 * i, frame_id), synthetic_rgb_byte(3 * i + 1, frame_id),
                          synthetic_rgb_byte(3 * i + 2, frame_id));
    if (host.gray[i] != y) return false;
    ++expected[y];
  }
  for (std::size_t bin = 0; bin < kHistogramBins; ++bin) {
    if (host.histogram[bin] != expected[bin]) return false;
  }
  return true;
}

}
