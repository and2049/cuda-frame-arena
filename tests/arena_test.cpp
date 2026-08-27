#include "frame_arena/arena.hpp"
#include "frame_arena/checked_math.hpp"
#include "check.hpp"

#include <cstdint>
#include <vector>

using namespace frame_arena;

namespace {

constexpr std::size_t kCapacity = 4096;

struct Backing {
  alignas(256) std::byte bytes[kCapacity];
};

std::uintptr_t addr(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

void test_align_up() {
  CHECK(align_up(0, 1) == 0);
  CHECK(align_up(1, 256) == 256);
  CHECK(align_up(256, 256) == 256);
  CHECK(align_up(257, 128) == 384);
  CHECK(!align_up(5, 0));
  CHECK(!align_up(5, 3));
  CHECK(!align_up(SIZE_MAX, 2));
  CHECK(!checked_add(SIZE_MAX, 1));
  CHECK(!checked_mul(SIZE_MAX / 2 + 1, 2));
  CHECK(checked_mul(0, SIZE_MAX) == 0);
}

void test_first_allocation_aligned() {
  Backing backing;
  for (std::size_t alignment = 1; alignment <= 256; alignment *= 2) {
    Arena arena(backing.bytes, kCapacity);
    void* p = arena.allocate(1, alignment);
    CHECK(p == backing.bytes);
    CHECK(addr(p) % alignment == 0);
  }
}

void test_successive_allocations_do_not_overlap() {
  Backing backing;
  Arena arena(backing.bytes, kCapacity);
  std::vector<std::pair<std::uintptr_t, std::size_t>> regions;
  std::size_t sizes[] = {3, 100, 1, 257, 64};
  std::size_t alignments[] = {1, 256, 128, 8, 64};
  for (int i = 0; i < 5; ++i) {
    void* p = arena.allocate(sizes[i], alignments[i]);
    CHECK(p != nullptr);
    CHECK(addr(p) % alignments[i] == 0);
    for (auto [start, size] : regions) {
      CHECK(addr(p) >= start + size || addr(p) + sizes[i] <= start);
    }
    regions.emplace_back(addr(p), sizes[i]);
  }
}

void test_padding_accounted() {
  Backing backing;
  Arena arena(backing.bytes, kCapacity);
  CHECK(arena.allocate(1, 1) != nullptr);
  CHECK(arena.used() == 1);
  CHECK(arena.allocate(1, 256) != nullptr);
  CHECK(arena.used() == 257);
  CHECK(arena.remaining() == kCapacity - 257);
}

void test_capacity_boundaries() {
  Backing backing;
  Arena exact(backing.bytes, kCapacity);
  CHECK(exact.allocate(kCapacity, 1) != nullptr);
  CHECK(exact.remaining() == 0);
  CHECK(exact.allocate(1, 1) == nullptr);

  Arena over(backing.bytes, kCapacity);
  CHECK(over.allocate(kCapacity + 1, 1) == nullptr);
  CHECK(over.used() == 0);

  Arena padded(backing.bytes, kCapacity);
  CHECK(padded.allocate(1, 1) != nullptr);
  CHECK(padded.allocate(kCapacity - 255, 256) == nullptr);
  CHECK(padded.used() == 1);
  CHECK(padded.allocate(kCapacity - 255, 1) != nullptr);
}

void test_invalid_alignment() {
  Backing backing;
  Arena arena(backing.bytes, kCapacity);
  CHECK(arena.allocate(8, 0) == nullptr);
  CHECK(arena.allocate(8, 3) == nullptr);
  CHECK(arena.allocate(8, 96) == nullptr);
  CHECK(arena.used() == 0);
}

void test_zero_size() {
  Backing backing;
  Arena arena(backing.bytes, kCapacity);
  CHECK(arena.allocate(5, 1) != nullptr);
  void* p = arena.allocate(0, 64);
  CHECK(p == backing.bytes + 64);
  CHECK(arena.used() == 64);
  CHECK(arena.allocate(kCapacity - 64, 1) != nullptr);
  CHECK(arena.allocate(0, 1) == backing.bytes + kCapacity);
}

void test_typed_overflow() {
  Backing backing;
  Arena arena(backing.bytes, kCapacity);
  CHECK(arena.allocate<std::uint64_t>(SIZE_MAX / 4) == nullptr);
  CHECK(arena.allocate(SIZE_MAX, 1) == nullptr);
  CHECK(arena.used() == 0);
  auto* ints = arena.allocate<std::uint32_t>(4);
  CHECK(ints != nullptr);
  CHECK(addr(ints) % alignof(std::uint32_t) == 0);
  CHECK(arena.used() == 16);
}

void test_reset_reproduces_sequence() {
  Backing backing;
  Arena arena(backing.bytes, kCapacity);
  void* a = arena.allocate(10, 1);
  void* b = arena.allocate(300, 256);
  void* c = arena.allocate(7, 128);
  arena.reset();
  CHECK(arena.used() == 0);
  CHECK(arena.allocate(10, 1) == a);
  CHECK(arena.allocate(300, 256) == b);
  CHECK(arena.allocate(7, 128) == c);
}

void test_empty_arena() {
  Arena arena;
  CHECK(arena.allocate(0, 1) == nullptr);
  CHECK(arena.capacity() == 0);
}

void test_alignment_sweep() {
  Backing backing;
  for (std::size_t start = 0; start < 512; ++start) {
    for (std::size_t alignment = 1; alignment <= 256; alignment *= 2) {
      Arena arena(backing.bytes, kCapacity);
      if (start > 0) CHECK(arena.allocate(start, 1) != nullptr);
      void* p = arena.allocate(16, alignment);
      CHECK(p != nullptr);
      CHECK(addr(p) % alignment == 0);
      CHECK(addr(p) >= addr(backing.bytes) + start);
      CHECK(addr(p) - addr(backing.bytes) - start < alignment);
    }
  }
}

}

int main() {
  test_align_up();
  test_first_allocation_aligned();
  test_successive_allocations_do_not_overlap();
  test_padding_accounted();
  test_capacity_boundaries();
  test_invalid_alignment();
  test_zero_size();
  test_typed_overflow();
  test_reset_reproduces_sequence();
  test_empty_arena();
  test_alignment_sweep();
  return test::finish("arena_test");
}
