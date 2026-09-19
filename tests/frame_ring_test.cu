#include "frame_arena/cuda_error.hpp"
#include "frame_arena/frame_layout.hpp"
#include "frame_arena/frame_ring.hpp"
#include "check.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

using namespace frame_arena;

namespace {

__global__ void spin_kernel(unsigned milliseconds) {
  for (unsigned i = 0; i < milliseconds; ++i) __nanosleep(1000000u);
}

void launch_spin(cudaStream_t stream, unsigned milliseconds) {
  spin_kernel<<<1, 1, 0, stream>>>(milliseconds);
  CUDA_CHECK(cudaGetLastError());
}

std::uintptr_t addr(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

bool disjoint(const Arena& a, const Arena& b) {
  return addr(a.base()) + a.capacity() <= addr(b.base()) || addr(b.base()) + b.capacity() <= addr(a.base());
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void test_slots_are_disjoint() {
  FrameRing ring(4, SlotLayout{1000, 300, 2000});
  CHECK(ring.layout().upload_bytes == 1024);
  CHECK(ring.layout().download_bytes == 512);
  CHECK(ring.layout().device_bytes == 2048);
  for (std::size_t i = 0; i < ring.depth(); ++i) {
    CHECK(addr(ring.slot(i).upload_arena.base()) % 256 == 0);
    CHECK(addr(ring.slot(i).download_arena.base()) % 256 == 0);
    CHECK(addr(ring.slot(i).device_arena.base()) % 256 == 0);
    CHECK(disjoint(ring.slot(i).upload_arena, ring.slot(i).download_arena));
    for (std::size_t j = i + 1; j < ring.depth(); ++j) {
      CHECK(disjoint(ring.slot(i).upload_arena, ring.slot(j).upload_arena));
      CHECK(disjoint(ring.slot(i).download_arena, ring.slot(j).download_arena));
      CHECK(disjoint(ring.slot(i).device_arena, ring.slot(j).device_arena));
    }
  }
  FrameLease lease = ring.acquire();
  void* p = lease.device_arena().allocate(512, 256);
  CHECK(p == ring.slot(lease.index()).device_arena.base());
  CHECK(lease.device_arena().allocate(1537, 1) == nullptr);
}

void test_moved_owners_are_empty() {
  DeviceBuffer device(1024);
  DeviceBuffer device_moved(std::move(device));
  CHECK(device.data() == nullptr && device.size() == 0);
  CHECK(device_moved.data() != nullptr && device_moved.size() == 1024);

  PinnedBuffer pinned(1024);
  PinnedBuffer pinned_moved;
  pinned_moved = std::move(pinned);
  CHECK(pinned.data() == nullptr && pinned.size() == 0);
  CHECK(pinned_moved.size() == 1024);

  CudaStream stream;
  CudaStream stream_moved(std::move(stream));
  CHECK(stream.get() == nullptr && stream_moved.get() != nullptr);

  CudaEvent event;
  CudaEvent event_moved(std::move(event));
  CHECK(event.get() == nullptr && event_moved.get() != nullptr);
}

void test_error_reports_operation() {
  void* p = nullptr;
  bool thrown = false;
  try {
    CUDA_CHECK(cudaMalloc(&p, ~std::size_t{0}));
  } catch (const CudaError& e) {
    thrown = true;
    std::string what = e.what();
    CHECK(e.code() == cudaErrorMemoryAllocation);
    CHECK(what.find("cudaMalloc") != std::string::npos);
    CHECK(what.find(cudaGetErrorString(e.code())) != std::string::npos);
    CHECK(what.find("frame_ring_test.cu") != std::string::npos);
  }
  CHECK(thrown);
  cudaGetLastError();
}

void test_busy_slot_is_not_reset() {
  FrameRing ring(2, 4096);
  {
    FrameLease lease = ring.acquire();
    CHECK(lease.upload_arena().allocate(100, 1) != nullptr);
    CHECK(lease.download_arena().allocate(200, 1) != nullptr);
    launch_spin(lease.stream(), 150);
    lease.submit();
  }
  ring.acquire().submit();

  CHECK(!ring.try_acquire().has_value());
  CHECK(ring.slot(0).state == SlotState::InFlight);
  CHECK(ring.slot(0).upload_arena.used() == 100);
  CHECK(ring.slot(0).download_arena.used() == 200);

  auto start = std::chrono::steady_clock::now();
  FrameLease lease = ring.acquire();
  CHECK(elapsed_ms(start) > 50);
  CHECK(lease.index() == 0);
  CHECK(lease.upload_arena().used() == 0);
  CHECK(lease.download_arena().used() == 0);
  CHECK(ring.stats().blocked == 1);
}

void test_slots_run_concurrently() {
  FrameRing ring(3, 1024);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 3; ++i) {
    FrameLease lease = ring.acquire();
    launch_spin(lease.stream(), 100);
  }
  ring.drain();
  double concurrent = elapsed_ms(start);
  CHECK(concurrent < 220);
  for (std::size_t i = 0; i < ring.depth(); ++i) CHECK(ring.slot(i).state == SlotState::Available);
}

void test_wraparound_preserves_correctness() {
  constexpr FrameDims dims[] = {{64, 48}, {320, 200}, {17, 33}, {256, 256}};
  FrameRing ring(3, frame_slot_layout({320, 256}));
  std::size_t verified = 0, failed = 0;
  for (std::uint32_t frame_id = 0; frame_id < 12; ++frame_id) {
    FrameLease lease = ring.acquire();
    auto host = carve_frame(lease.upload_arena(), lease.download_arena(), dims[frame_id % 4]);
    auto device = carve_frame(lease.device_arena(), dims[frame_id % 4]);
    CHECK(host && device);
    fill_synthetic_rgb(host->rgb, dims[frame_id % 4], frame_id);
    enqueue_frame(*host, *device, dims[frame_id % 4], lease.stream());
    lease.on_retire([host = *host, d = dims[frame_id % 4], frame_id, &verified, &failed] {
      verify_grayscale(host, d, frame_id) ? ++verified : ++failed;
    });
    lease.submit();
  }
  ring.drain();
  CHECK(verified == 12 && failed == 0);
  CHECK(ring.stats().acquired == 12);
}

void test_exhaustion_is_clean() {
  FrameRing ring(1, 1024);
  FrameLease lease = ring.acquire();
  CHECK(!carve_frame(lease.upload_arena(), lease.download_arena(), {640, 480}).has_value());
  CHECK(lease.upload_arena().used() == 0);
  CHECK(lease.download_arena().used() == 0);
  CHECK(lease.device_arena().allocate(2048, 1) == nullptr);
  CHECK(lease.device_arena().used() == 0);
  CHECK(lease.device_arena().allocate(1024, 1) != nullptr);
}

void test_unsubmitted_lease_is_submitted_on_drop() {
  FrameRing ring(1, 1024);
  { FrameLease lease = ring.acquire(); }
  CHECK(ring.slot(0).state == SlotState::InFlight);
  CHECK(ring.try_acquire().has_value());
}

// At depth 2 the producer blocks in acquire() on an event that only this thread's
// submit records, so the test deadlocks if the ring waits with its lock held.
void test_leases_move_between_threads() {
  constexpr int kFrames = 32;
  FrameRing ring(2, 1024);
  std::mutex mutex;
  std::condition_variable handed_over;
  std::deque<FrameLease> pending;
  std::size_t retired = 0;

  std::thread producer([&] {
    for (int i = 0; i < kFrames; ++i) {
      FrameLease lease = ring.acquire();
      launch_spin(lease.stream(), 2);
      lease.on_retire([&retired] { ++retired; });
      std::lock_guard lock(mutex);
      pending.push_back(std::move(lease));
      handed_over.notify_one();
    }
  });
  for (int i = 0; i < kFrames; ++i) {
    std::unique_lock lock(mutex);
    handed_over.wait(lock, [&] { return !pending.empty(); });
    FrameLease lease = std::move(pending.front());
    pending.pop_front();
    lock.unlock();
    launch_spin(lease.stream(), 2);
    lease.submit();
  }
  producer.join();
  ring.drain();
  CHECK(ring.stats().acquired == kFrames);
  CHECK(ring.stats().blocked > 0);
  CHECK(retired == kFrames);
}

void test_leased_slot_is_not_ready() {
  FrameRing ring(1, 1024);
  FrameLease lease = ring.acquire();
  CHECK(!ring.try_acquire().has_value());
  lease.submit();
  CHECK(ring.try_acquire().has_value());
}

}

int main() {
  test_slots_are_disjoint();
  test_moved_owners_are_empty();
  test_error_reports_operation();
  test_busy_slot_is_not_reset();
  test_slots_run_concurrently();
  test_wraparound_preserves_correctness();
  test_exhaustion_is_clean();
  test_unsubmitted_lease_is_submitted_on_drop();
  test_leases_move_between_threads();
  test_leased_slot_is_not_ready();
  return test::finish("frame_ring_test");
}
