#pragma once

#include <cstddef>
#include <new>
#include <type_traits>

namespace frame_arena {

class Arena {
public:
  Arena() noexcept = default;
  Arena(void* base, std::size_t capacity) noexcept;

  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment) noexcept;

  template <class T>
  [[nodiscard]] T* allocate(std::size_t count = 1) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    return static_cast<T*>(allocate_array(sizeof(T), alignof(T), count));
  }

  void reset() noexcept { offset_ = 0; }

  void* base() const noexcept { return base_; }
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t used() const noexcept { return offset_; }
  std::size_t remaining() const noexcept { return capacity_ - offset_; }

private:
  void* allocate_array(std::size_t elem_size, std::size_t alignment, std::size_t count) noexcept;

  std::byte* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t offset_ = 0;
};

}
