#pragma once

#include "frame_arena/arena.hpp"
#include "frame_arena/cuda_resource.hpp"

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace frame_arena {

enum class SlotState { Available, HostWriting, InFlight };

// Write-combined pinned memory suits the upload arena, which the CPU only writes;
// it is uncached, so reading it back is very slow.
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

// mutex_ covers the ring order, slot states and stats, so a lease can be acquired on
// one thread and submitted on another. What a lease exposes belongs to its holder.
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
  std::size_t next_index() const;
  const FrameSlot& slot(std::size_t index) const noexcept { return slots_[index]; }
  RingStats stats() const;

private:
  friend class FrameLease;
  bool next_slot_ready();
  void wait_for_next_slot(std::unique_lock<std::mutex>& lock);
  FrameLease lease(std::unique_lock<std::mutex>& lock);
  void retire(FrameSlot& slot);
  void submit(std::size_t index);

  mutable std::mutex mutex_;
  std::condition_variable submitted_;
  SlotLayout layout_;
  PinnedBuffer upload_slab_;
  PinnedBuffer download_slab_;
  DeviceBuffer device_slab_;
  std::vector<FrameSlot> slots_;
  std::size_t next_ = 0;
  RingStats stats_;
};

}
