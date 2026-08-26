#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace frame_arena {

constexpr bool is_pow2(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

constexpr std::optional<std::size_t> checked_add(std::size_t a, std::size_t b) noexcept {
  if (a > SIZE_MAX - b) return std::nullopt;
  return a + b;
}

constexpr std::optional<std::size_t> checked_mul(std::size_t a, std::size_t b) noexcept {
  if (a != 0 && b > SIZE_MAX / a) return std::nullopt;
  return a * b;
}

constexpr std::optional<std::size_t> align_up(std::size_t value, std::size_t alignment) noexcept {
  if (!is_pow2(alignment)) return std::nullopt;
  auto bumped = checked_add(value, alignment - 1);
  if (!bumped) return std::nullopt;
  return *bumped & ~(alignment - 1);
}

}
