#include "frame_arena/arena.hpp"
#include "frame_arena/checked_math.hpp"

namespace frame_arena {

Arena::Arena(void* base, std::size_t capacity) noexcept
    : base_(static_cast<std::byte*>(base)), capacity_(capacity) {}

void* Arena::allocate(std::size_t bytes, std::size_t alignment) noexcept {
  if (base_ == nullptr) return nullptr;
  auto start = align_up(offset_, alignment);
  if (!start) return nullptr;
  auto end = checked_add(*start, bytes);
  if (!end || *end > capacity_) return nullptr;
  offset_ = *end;
  return base_ + *start;
}

void* Arena::allocate_array(std::size_t elem_size, std::size_t alignment, std::size_t count) noexcept {
  auto bytes = checked_mul(elem_size, count);
  if (!bytes) return nullptr;
  return allocate(*bytes, alignment);
}

}
