#pragma once

#include "frame_arena/arena.hpp"
#include "frame_arena/cuda_error.hpp"
#include "frame_arena/frame_ring.hpp"
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

// Per-region alignments, chosen to vary on purpose so padding shows up in used().
constexpr std::size_t kRgbAlignment = 256;
constexpr std::size_t kGrayAlignment = 128;
constexpr std::size_t kHistogramAlignment = 256;

constexpr std::size_t frame_upload_bytes(FrameDims dims) noexcept {
  return dims.pixels() * 3 + kRgbAlignment;
}

constexpr std::size_t frame_download_bytes(FrameDims dims) noexcept {
  return dims.pixels() + kGrayAlignment + kHistogramBytes + kHistogramAlignment + alignof(FrameMetadata) +
         sizeof(FrameMetadata);
}

// The device arena holds both directions: the kernel reads rgb and writes gray
// and histogram from the same slab.
constexpr SlotLayout frame_slot_layout(FrameDims dims) noexcept {
  return {frame_upload_bytes(dims), frame_download_bytes(dims), frame_upload_bytes(dims) + frame_download_bytes(dims)};
}

// rgb comes from `input`, which the CPU fills and the copy engine reads; gray,
// histogram and metadata come from `output`, which the CPU reads back. On the
// host these are the upload and download arenas; on the device both are the
// slot's single device arena.
inline std::optional<FrameRegions> carve_frame(Arena& input, Arena& output, FrameDims dims) noexcept {
  FrameRegions r;
  if (!(r.rgb = static_cast<std::uint8_t*>(input.allocate(dims.pixels() * 3, kRgbAlignment)))) return std::nullopt;
  if (!(r.gray = static_cast<std::uint8_t*>(output.allocate(dims.pixels(), kGrayAlignment)))) return std::nullopt;
  if (!(r.histogram = static_cast<std::uint32_t*>(output.allocate(kHistogramBytes, kHistogramAlignment)))) return std::nullopt;
  if (!(r.metadata = output.allocate<FrameMetadata>())) return std::nullopt;
  return r;
}

inline std::optional<FrameRegions> carve_frame(Arena& arena, FrameDims dims) noexcept {
  return carve_frame(arena, arena, dims);
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
inline bool verify_grayscale(const FrameRegions& host, FrameDims dims, std::uint32_t frame_id) {
  std::vector<std::uint8_t> rgb(dims.pixels() * 3);
  fill_synthetic_rgb(rgb.data(), dims, frame_id);
  std::vector<std::uint32_t> expected(kHistogramBins, 0);
  for (std::size_t i = 0; i < dims.pixels(); ++i) {
    std::uint8_t y = luma(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
    if (host.gray[i] != y) return false;
    ++expected[y];
  }
  for (std::size_t bin = 0; bin < kHistogramBins; ++bin) {
    if (host.histogram[bin] != expected[bin]) return false;
  }
  return true;
}

}
