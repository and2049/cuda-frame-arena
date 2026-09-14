#pragma once

#include "frame_arena/arena.hpp"
#include "frame_arena/cuda_resource.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace frame_arena {

enum class SlotState { Available, HostWriting, InFlight };

// Host memory is split by direction. The upload arena is the source of H2D
// copies: the CPU only writes it, so it can live in write-combined pinned
// memory, which is uncached and very slow to read back. The download arena
// receives D2H copies and anything the CPU reads, so it stays cacheable.
enum class UploadMemory { WriteCombined, Cacheable };

struct SlotLayout {
  std::size_t upload_bytes;
  std::size_t download_bytes;
  std::size_t device_bytes;
};

struct FrameSlot {
  Arena upload_arena;
  Arena download_arena;
  Arena device_arena;
  CudaStream stream;
  CudaEvent completion;
  SlotState state = SlotState::Available;
  std::function<void()> on_retire;
};

struct RingStats {
  std::size_t acquired = 0;
  std::size_t blocked = 0;
};

class FrameRing;

class FrameLease {
public:
  ~FrameLease() noexcept;

  FrameLease(const FrameLease&) = delete;
  FrameLease& operator=(const FrameLease&) = delete;

  FrameLease(FrameLease&& other) noexcept;
  FrameLease& operator=(FrameLease&& other) noexcept;

  Arena& upload_arena() noexcept;
  Arena& download_arena() noexcept;
  Arena& device_arena() noexcept;
  cudaStream_t stream() const noexcept;
  std::size_t index() const noexcept { return index_; }

  void on_retire(std::function<void()> callback);
  void submit();

private:
  friend class FrameRing;
  FrameLease(FrameRing& ring, std::size_t index) noexcept : ring_(&ring), index_(index) {}
  void release() noexcept;

  FrameRing* ring_;
  std::size_t index_;
};

class FrameRing {
public:
  FrameRing(std::size_t depth, SlotLayout layout, UploadMemory upload = UploadMemory::WriteCombined);
  FrameRing(std::size_t depth, std::size_t slot_bytes)
      : FrameRing(depth, SlotLayout{slot_bytes, slot_bytes, slot_bytes}) {}

  [[nodiscard]] std::optional<FrameLease> try_acquire();
  [[nodiscard]] FrameLease acquire();
  void drain();

  std::size_t depth() const noexcept { return slots_.size(); }
  const SlotLayout& layout() const noexcept { return layout_; }
  std::size_t next_index() const noexcept { return next_; }
  const FrameSlot& slot(std::size_t index) const noexcept { return slots_[index]; }
  const RingStats& stats() const noexcept { return stats_; }

private:
  friend class FrameLease;
  FrameLease lease(std::size_t index);
  void retire(FrameSlot& slot);
  void submit(std::size_t index);

  SlotLayout layout_;
  PinnedBuffer upload_slab_;
  PinnedBuffer download_slab_;
  DeviceBuffer device_slab_;
  std::vector<FrameSlot> slots_;
  std::size_t next_ = 0;
  RingStats stats_;
};

}
