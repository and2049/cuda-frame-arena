#pragma once

#include "frame_arena/arena.hpp"
#include "frame_arena/cuda_resource.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace frame_arena {

enum class SlotState { Available, HostWriting, InFlight };

struct FrameSlot {
  Arena host_arena;
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

  Arena& host_arena() noexcept;
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
  FrameRing(std::size_t depth, std::size_t slot_bytes);

  [[nodiscard]] std::optional<FrameLease> try_acquire();
  [[nodiscard]] FrameLease acquire();
  void drain();

  std::size_t depth() const noexcept { return slots_.size(); }
  std::size_t slot_bytes() const noexcept { return slot_bytes_; }
  std::size_t next_index() const noexcept { return next_; }
  const FrameSlot& slot(std::size_t index) const noexcept { return slots_[index]; }
  const RingStats& stats() const noexcept { return stats_; }

private:
  friend class FrameLease;
  FrameLease lease(std::size_t index);
  void retire(FrameSlot& slot);
  void submit(std::size_t index);

  std::size_t slot_bytes_;
  PinnedBuffer host_slab_;
  DeviceBuffer device_slab_;
  std::vector<FrameSlot> slots_;
  std::size_t next_ = 0;
  RingStats stats_;
};

}
