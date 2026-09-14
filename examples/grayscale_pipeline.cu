#include "frame_arena/frame_layout.hpp"
#include "frame_arena/frame_ring.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace frame_arena;

namespace {

constexpr FrameDims kDimsCycle[] = {{640, 480}, {1280, 720}, {1920, 1080}, {800, 600}};

struct Counters {
  std::size_t verified = 0;
  std::size_t failed = 0;
  std::size_t dropped = 0;
};

void process_frame(FrameLease& lease, FrameDims dims, std::uint32_t frame_id, Counters& counters) {
  auto host = carve_frame(lease.upload_arena(), lease.download_arena(), dims);
  auto device = carve_frame(lease.device_arena(), dims);
  if (!host || !device) throw std::runtime_error("frame does not fit in slot arena");

  fill_synthetic_rgb(host->rgb, dims, frame_id);
  *host->metadata = {frame_id, dims.width, dims.height};
  enqueue_frame(*host, *device, dims, lease.stream());

  lease.on_retire([host = *host, dims, frame_id, &counters] {
    verify_grayscale(host, dims, frame_id) ? ++counters.verified : ++counters.failed;
  });
  lease.submit();
}

}

int main(int argc, char** argv) {
  std::size_t depth = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 3;
  std::size_t frames = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 64;
  bool drop_when_busy = argc > 3 && std::strcmp(argv[3], "drop") == 0;

  FrameRing ring(depth, frame_slot_layout({1920, 1080}));
  Counters counters;

  auto start = std::chrono::steady_clock::now();
  for (std::uint32_t frame_id = 0; frame_id < frames; ++frame_id) {
    FrameDims dims = kDimsCycle[frame_id % 4];
    if (drop_when_busy) {
      auto lease = ring.try_acquire();
      lease ? process_frame(*lease, dims, frame_id, counters) : void(++counters.dropped);
    } else {
      FrameLease lease = ring.acquire();
      process_frame(lease, dims, frame_id, counters);
    }
  }
  ring.drain();
  auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);

  std::printf("depth=%zu frames=%zu verified=%zu failed=%zu dropped=%zu blocked=%zu\n", depth, frames,
              counters.verified, counters.failed, counters.dropped, ring.stats().blocked);
  std::printf("elapsed=%.2f ms  %.1f frames/s\n", elapsed.count(), counters.verified / (elapsed.count() / 1000.0));
  return counters.failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
