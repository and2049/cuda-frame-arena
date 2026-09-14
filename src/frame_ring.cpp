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
  FrameSlot& slot = slots_[next_];
  if (slot.state == SlotState::HostWriting) throw std::logic_error("next slot is still leased");
  cudaError_t status = slot.completion.query();
  if (status == cudaErrorNotReady) return std::nullopt;
  check_cuda(status, "cudaEventQuery");
  return lease(next_);
}

FrameLease FrameRing::acquire() {
  if (auto ready = try_acquire()) return std::move(*ready);
  ++stats_.blocked;
  slots_[next_].completion.synchronize();
  return lease(next_);
}

void FrameRing::drain() {
  for (FrameSlot& slot : slots_) {
    if (slot.state != SlotState::InFlight) continue;
    slot.completion.synchronize();
    retire(slot);
  }
}

FrameLease FrameRing::lease(std::size_t index) {
  FrameSlot& slot = slots_[index];
  retire(slot);
  slot.state = SlotState::HostWriting;
  next_ = (index + 1) % slots_.size();
  ++stats_.acquired;
  return FrameLease(*this, index);
}

void FrameRing::retire(FrameSlot& slot) {
  if (slot.on_retire) std::exchange(slot.on_retire, nullptr)();
  slot.upload_arena.reset();
  slot.download_arena.reset();
  slot.device_arena.reset();
  slot.state = SlotState::Available;
}

void FrameRing::submit(std::size_t index) {
  FrameSlot& slot = slots_[index];
  slot.completion.record(slot.stream.get());
  slot.state = SlotState::InFlight;
}

}
