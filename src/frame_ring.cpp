#include "frame_arena/frame_ring.hpp"
#include "frame_arena/checked_math.hpp"
#include "frame_arena/cuda_error.hpp"

#include <cstdio>
#include <stdexcept>

namespace frame_arena {

namespace {

constexpr std::size_t kSlotAlignment = 256;

std::size_t padded_slot_bytes(std::size_t slot_bytes) {
  auto padded = align_up(slot_bytes, kSlotAlignment);
  if (!padded || *padded == 0) throw std::invalid_argument("invalid slot size");
  return *padded;
}

SlotLayout padded_layout(SlotLayout layout) {
  return {padded_slot_bytes(layout.upload_bytes), padded_slot_bytes(layout.download_bytes),
          padded_slot_bytes(layout.device_bytes)};
}

unsigned upload_flags(UploadMemory upload) {
  return upload == UploadMemory::WriteCombined ? cudaHostAllocWriteCombined : cudaHostAllocDefault;
}

Arena slot_arena(void* slab, std::size_t index, std::size_t slot_bytes) {
  return Arena(static_cast<std::byte*>(slab) + index * slot_bytes, slot_bytes);
}

std::size_t slab_bytes(std::size_t depth, std::size_t slot_bytes) {
  auto total = checked_mul(depth, slot_bytes);
  if (!total || depth == 0) throw std::invalid_argument("invalid ring depth");
  return *total;
}

}

FrameLease::~FrameLease() noexcept { release(); }

FrameLease::FrameLease(FrameLease&& other) noexcept
    : ring_(std::exchange(other.ring_, nullptr)), index_(other.index_) {}

FrameLease& FrameLease::operator=(FrameLease&& other) noexcept {
  if (this != &other) {
    release();
    ring_ = std::exchange(other.ring_, nullptr);
    index_ = other.index_;
  }
  return *this;
}

Arena& FrameLease::upload_arena() noexcept { return ring_->slots_[index_].upload_arena; }

Arena& FrameLease::download_arena() noexcept { return ring_->slots_[index_].download_arena; }

Arena& FrameLease::device_arena() noexcept { return ring_->slots_[index_].device_arena; }

cudaStream_t FrameLease::stream() const noexcept { return ring_->slots_[index_].stream.get(); }

void FrameLease::on_retire(std::function<void()> callback) {
  ring_->slots_[index_].on_retire = std::move(callback);
}

void FrameLease::submit() {
  if (ring_ == nullptr) throw std::logic_error("lease already submitted");
  ring_->submit(index_);
  ring_ = nullptr;
}

void FrameLease::release() noexcept {
  if (ring_ == nullptr) return;
  try {
    submit();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "implicit submit failed: %s\n", e.what());
    ring_ = nullptr;
  }
}

FrameRing::FrameRing(std::size_t depth, SlotLayout layout, UploadMemory upload)
    : layout_(padded_layout(layout)),
      upload_slab_(slab_bytes(depth, layout_.upload_bytes), upload_flags(upload)),
      download_slab_(slab_bytes(depth, layout_.download_bytes)),
      device_slab_(slab_bytes(depth, layout_.device_bytes)) {
  slots_.reserve(depth);
  for (std::size_t i = 0; i < depth; ++i) {
    FrameSlot& slot = slots_.emplace_back();
    slot.upload_arena = slot_arena(upload_slab_.data(), i, layout_.upload_bytes);
    slot.download_arena = slot_arena(download_slab_.data(), i, layout_.download_bytes);
    slot.device_arena = slot_arena(device_slab_.data(), i, layout_.device_bytes);
  }
}

std::optional<FrameLease> FrameRing::try_acquire() {
  std::unique_lock lock(mutex_);
  if (!next_slot_ready()) return std::nullopt;
  return lease(lock);
}

FrameLease FrameRing::acquire() {
  std::unique_lock lock(mutex_);
  if (next_slot_ready()) return lease(lock);
  ++stats_.blocked;
  do {
    wait_for_next_slot(lock);
  } while (!next_slot_ready());
  return lease(lock);
}

void FrameRing::drain() {
  std::lock_guard lock(mutex_);
  for (FrameSlot& slot : slots_) {
    if (slot.state != SlotState::InFlight) continue;
    slot.completion.synchronize();
    retire(slot);
    slot.state = SlotState::Available;
  }
}

std::size_t FrameRing::next_index() const {
  std::lock_guard lock(mutex_);
  return next_;
}

RingStats FrameRing::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

// Called with mutex_ held.
bool FrameRing::next_slot_ready() {
  FrameSlot& slot = slots_[next_];
  if (slot.state == SlotState::HostWriting) return false;
  cudaError_t status = slot.completion.query();
  if (status == cudaErrorNotReady) return false;
  check_cuda(status, "cudaEventQuery");
  return true;
}

// The event wait drops the lock because the submit that records the event needs it.
// Either wait may lose the slot to another thread, so the caller checks again.
void FrameRing::wait_for_next_slot(std::unique_lock<std::mutex>& lock) {
  FrameSlot& slot = slots_[next_];
  if (slot.state == SlotState::HostWriting) {
    submitted_.wait(lock);
    return;
  }
  lock.unlock();
  slot.completion.synchronize();
  lock.lock();
}

// Claims the slot under the lock and retires it after, since on_retire is caller code
// and may be slow. If it throws, the lease's destructor submits the slot.
FrameLease FrameRing::lease(std::unique_lock<std::mutex>& lock) {
  std::size_t index = next_;
  FrameSlot& slot = slots_[index];
  slot.state = SlotState::HostWriting;
  next_ = (index + 1) % slots_.size();
  ++stats_.acquired;
  lock.unlock();
  FrameLease claimed(*this, index);
  retire(slot);
  return claimed;
}

void FrameRing::retire(FrameSlot& slot) {
  if (slot.on_retire) std::exchange(slot.on_retire, nullptr)();
  slot.upload_arena.reset();
  slot.download_arena.reset();
  slot.device_arena.reset();
}

void FrameRing::submit(std::size_t index) {
  std::lock_guard lock(mutex_);
  FrameSlot& slot = slots_[index];
  slot.completion.record(slot.stream.get());
  slot.state = SlotState::InFlight;
  submitted_.notify_all();
}

}
