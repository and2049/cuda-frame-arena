#include "frame_arena/cuda_error.hpp"
#include "frame_arena/frame_layout.hpp"
#include "frame_arena/frame_ring.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace frame_arena;
using Clock = std::chrono::steady_clock;

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

bool spot_check(const FrameRegions& host, FrameDims dims, std::uint32_t frame_id) {
  std::uint32_t total = 0;
  for (std::size_t bin = 0; bin < kHistogramBins; ++bin) total += host.histogram[bin];
  if (total != dims.pixels()) return false;
  for (std::size_t i = 0; i < dims.pixels(); i += 61) {
    std::uint8_t y = luma(synthetic_rgb_byte(3 * i, frame_id), synthetic_rgb_byte(3 * i + 1, frame_id),
                          synthetic_rgb_byte(3 * i + 2, frame_id));
    if (host.gray[i] != y) return false;
  }
  return true;
}

class Policy {
public:
  virtual ~Policy() = default;
  virtual const char* name() const = 0;
  virtual FrameRegions host(FrameLease& lease, FrameDims dims) = 0;
  virtual FrameRegions device(FrameLease& lease, FrameDims dims) = 0;
  virtual void enqueue_release(FrameLease&, const FrameRegions&) {}
  virtual void release(std::size_t, const FrameRegions&, const FrameRegions&) {}
};

FrameRegions require(std::optional<FrameRegions> regions) {
  if (!regions) throw std::runtime_error("slot arena exhausted");
  return *regions;
}

std::array<void*, 4> pointers(const FrameRegions& r) {
  return {r.rgb, r.gray, r.histogram, r.metadata};
}

std::array<std::size_t, 4> region_sizes(FrameDims dims) {
  return {dims.pixels() * 3, dims.pixels(), kHistogramBytes, sizeof(FrameMetadata)};
}

// Region 0 (rgb) is upload-only, so the separate-allocation policies give it
// the same write-combined memory the ring's upload slab uses.
unsigned host_flags(std::size_t region) {
  return region == 0 ? cudaHostAllocWriteCombined : cudaHostAllocDefault;
}

FrameRegions regions_from(const std::array<void*, 4>& p) {
  return {static_cast<std::uint8_t*>(p[0]), static_cast<std::uint8_t*>(p[1]),
          static_cast<std::uint32_t*>(p[2]), static_cast<FrameMetadata*>(p[3])};
}

class ArenaPolicy : public Policy {
public:
  const char* name() const override { return "preallocated arena"; }
  FrameRegions host(FrameLease& lease, FrameDims dims) override {
    return require(carve_frame(lease.upload_arena(), lease.download_arena(), dims));
  }
  FrameRegions device(FrameLease& lease, FrameDims dims) override { return require(carve_frame(lease.device_arena(), dims)); }
};

class PerFramePolicy : public Policy {
public:
  const char* name() const override { return "cudaMalloc/cudaFree per frame"; }

  FrameRegions host(FrameLease&, FrameDims dims) override {
    std::array<void*, 4> p;
    for (std::size_t i = 0; i < 4; ++i) CUDA_CHECK(cudaHostAlloc(&p[i], region_sizes(dims)[i], host_flags(i)));
    return regions_from(p);
  }

  FrameRegions device(FrameLease&, FrameDims dims) override {
    std::array<void*, 4> p;
    for (std::size_t i = 0; i < 4; ++i) CUDA_CHECK(cudaMalloc(&p[i], region_sizes(dims)[i]));
    return regions_from(p);
  }

  void release(std::size_t, const FrameRegions& host, const FrameRegions& device) override {
    for (void* p : pointers(host)) CUDA_CHECK(cudaFreeHost(p));
    for (void* p : pointers(device)) CUDA_CHECK(cudaFree(p));
  }
};

class MallocAsyncPolicy : public Policy {
public:
  // By default the pool returns memory to the driver at every synchronization, so a
  // per-frame cudaMallocAsync would remap each time the ring blocks.
  MallocAsyncPolicy() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t threshold = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  }

  const char* name() const override { return "cudaMallocAsync/cudaFreeAsync"; }
  FrameRegions host(FrameLease& lease, FrameDims dims) override {
    return require(carve_frame(lease.upload_arena(), lease.download_arena(), dims));
  }

  FrameRegions device(FrameLease& lease, FrameDims dims) override {
    std::array<void*, 4> p;
    for (std::size_t i = 0; i < 4; ++i) CUDA_CHECK(cudaMallocAsync(&p[i], region_sizes(dims)[i], lease.stream()));
    return regions_from(p);
  }

  void enqueue_release(FrameLease& lease, const FrameRegions& device) override {
    for (void* p : pointers(device)) CUDA_CHECK(cudaFreeAsync(p, lease.stream()));
  }
};

class ReusedBuffersPolicy : public Policy {
public:
  ReusedBuffersPolicy(std::size_t depth, FrameDims max_dims) {
    for (std::size_t i = 0; i < depth; ++i) {
      host_.push_back(make_regions(pinned_, max_dims, [](std::size_t i, std::size_t bytes) { return PinnedBuffer(bytes, host_flags(i)); }));
      device_.push_back(make_regions(devices_, max_dims, [](std::size_t, std::size_t bytes) { return DeviceBuffer(bytes); }));
    }
  }

  const char* name() const override { return "reused separate allocations"; }
  FrameRegions host(FrameLease& lease, FrameDims) override { return host_[lease.index()]; }
  FrameRegions device(FrameLease& lease, FrameDims) override { return device_[lease.index()]; }

private:
  template <class Buffer, class Make>
  static FrameRegions make_regions(std::vector<Buffer>& owners, FrameDims dims, Make make) {
    std::array<void*, 4> p;
    for (std::size_t i = 0; i < 4; ++i) p[i] = owners.emplace_back(make(i, region_sizes(dims)[i])).data();
    return regions_from(p);
  }

  std::vector<PinnedBuffer> pinned_;
  std::vector<DeviceBuffer> devices_;
  std::vector<FrameRegions> host_, device_;
};

struct Sample {
  double alloc_ms, h2d_ms, kernel_ms, d2h_ms, latency_ms;
  Clock::time_point begin, end;
};

struct Pending {
  bool active = false;
  std::uint32_t frame_id = 0;
  FrameRegions host, device;
  Sample sample;
  CudaEvent marks[4]{CudaEvent(cudaEventDefault), CudaEvent(cudaEventDefault), CudaEvent(cudaEventDefault), CudaEvent(cudaEventDefault)};
};

struct Result {
  std::string name;
  std::size_t depth, blocked;
  double throughput_fps;
  std::vector<Sample> samples;
};

class Runner {
public:
  Runner(Policy& policy, std::size_t depth, FrameDims dims, std::size_t frames, std::size_t warmup,
         UploadMemory upload = UploadMemory::WriteCombined, const char* label = nullptr)
      : policy_(policy), label_(label ? label : policy.name()), dims_(dims), frames_(frames), warmup_(warmup),
        ring_(depth, frame_slot_layout(dims), upload), pending_(depth) {}

  Result run() {
    for (std::uint32_t id = 0; id < frames_; ++id) submit(id);
    while (in_flight_ > 0) poll();
    std::vector<Sample> steady(samples_.begin() + warmup_, samples_.end());
    auto last_end = std::max_element(steady.begin(), steady.end(), [](auto& a, auto& b) { return a.end < b.end; })->end;
    double fps = steady.size() / (ms_between(steady.front().begin, last_end) / 1000.0);
    return {label_, ring_.depth(), blocked_, fps, std::move(steady)};
  }

private:
  void submit(std::uint32_t id) {
    auto begin = Clock::now();
    std::optional<FrameLease> lease = ring_.try_acquire();
    if (!lease) ++blocked_;
    while (!lease) {
      poll();
      lease = ring_.try_acquire();
    }
    Pending& p = pending_[lease->index()];
    if (p.active) finalize(lease->index());
    auto alloc_start = Clock::now();
    p.host = policy_.host(*lease, dims_);
    p.device = policy_.device(*lease, dims_);
    p.sample.alloc_ms = ms_between(alloc_start, Clock::now());
    fill_synthetic_rgb(p.host.rgb, dims_, id);
    // Completions are only observed in poll(), so latency is quantized by one submit
    // iteration; polling after the fill, the longest host step, halves that.
    poll();
    cudaStream_t s = lease->stream();
    p.marks[0].record(s);
    CUDA_CHECK(cudaMemcpyAsync(p.device.rgb, p.host.rgb, dims_.pixels() * 3, cudaMemcpyHostToDevice, s));
    p.marks[1].record(s);
    CUDA_CHECK(cudaMemsetAsync(p.device.histogram, 0, kHistogramBytes, s));
    launch_grayscale(p.device.rgb, p.device.gray, p.device.histogram, static_cast<std::uint32_t>(dims_.pixels()), s);
    p.marks[2].record(s);
    CUDA_CHECK(cudaMemcpyAsync(p.host.gray, p.device.gray, dims_.pixels(), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(p.host.histogram, p.device.histogram, kHistogramBytes, cudaMemcpyDeviceToHost, s));
    p.marks[3].record(s);
    policy_.enqueue_release(*lease, p.device);
    p.sample.begin = begin;
    p.frame_id = id;
    p.active = true;
    ++in_flight_;
    lease->submit();
    poll();
  }

  void poll() {
    for (std::size_t i = 0; i < pending_.size(); ++i) {
      if (!pending_[i].active) continue;
      cudaError_t status = ring_.slot(i).completion.query();
      if (status == cudaErrorNotReady) continue;
      check_cuda(status, "cudaEventQuery");
      finalize(i);
    }
  }

  void finalize(std::size_t i) {
    Pending& p = pending_[i];
    p.sample.end = Clock::now();
    p.sample.latency_ms = ms_between(p.sample.begin, p.sample.end);
    p.sample.h2d_ms = p.marks[1].elapsed_ms_since(p.marks[0]);
    p.sample.kernel_ms = p.marks[2].elapsed_ms_since(p.marks[1]);
    p.sample.d2h_ms = p.marks[3].elapsed_ms_since(p.marks[2]);
    if (!spot_check(p.host, dims_, p.frame_id)) throw std::runtime_error("verification failed");
    policy_.release(i, p.host, p.device);
    samples_.push_back(p.sample);
    p.active = false;
    --in_flight_;
  }

  Policy& policy_;
  std::string label_;
  FrameDims dims_;
  std::size_t frames_, warmup_;
  FrameRing ring_;
  std::vector<Pending> pending_;
  std::vector<Sample> samples_;
  std::size_t in_flight_ = 0;
  std::size_t blocked_ = 0;
};

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<std::size_t>(p * values.size()))];
}

double mean(const std::vector<Sample>& samples, double Sample::*field) {
  double total = 0;
  for (const Sample& s : samples) total += s.*field;
  return total / samples.size();
}

void print(const Result& r) {
  std::vector<double> latency;
  for (const Sample& s : r.samples) latency.push_back(s.latency_ms);
  std::printf("%-32s depth=%zu alloc=%7.3f h2d=%6.3f kernel=%6.3f d2h=%6.3f | latency p50=%6.2f p90=%6.2f p99=%6.2f | %7.1f fps blocked=%zu\n",
              r.name.c_str(), r.depth, mean(r.samples, &Sample::alloc_ms), mean(r.samples, &Sample::h2d_ms),
              mean(r.samples, &Sample::kernel_ms), mean(r.samples, &Sample::d2h_ms), percentile(latency, 0.5),
              percentile(latency, 0.9), percentile(latency, 0.99), r.throughput_fps, r.blocked);
}

void print_environment(FrameDims dims, std::size_t frames) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int runtime = 0, driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  std::printf("device=%s runtime=%d driver=%d frame=%ux%u frames=%zu (times in ms, means; latency percentiles)\n\n",
              prop.name, runtime, driver, dims.width, dims.height, frames);
}

}

int main(int argc, char** argv) {
  std::size_t frames = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 200;
  FrameDims dims{static_cast<std::uint16_t>(argc > 3 ? std::atoi(argv[2]) : 1920),
                 static_cast<std::uint16_t>(argc > 3 ? std::atoi(argv[3]) : 1080)};
  std::size_t warmup = 20;
  print_environment(dims, frames);

  std::size_t depths[] = {1, 2, 3, 4, 8};
  std::printf("== ring depth sweep (preallocated arena) ==\n");
  for (std::size_t depth : depths) {
    ArenaPolicy policy;
    print(Runner(policy, depth, dims, frames, warmup).run());
  }

  std::printf("\n== upload memory at depth 3 (preallocated arena) ==\n");
  {
    ArenaPolicy policy;
    print(Runner(policy, 3, dims, frames, warmup, UploadMemory::Cacheable, "cacheable upload slab").run());
    print(Runner(policy, 3, dims, frames, warmup, UploadMemory::WriteCombined, "write-combined upload slab").run());
  }

  std::printf("\n== allocation strategies at depth 3 ==\n");
  PerFramePolicy per_frame;
  ArenaPolicy arena;
  MallocAsyncPolicy malloc_async;
  ReusedBuffersPolicy reused(3, dims);
  for (Policy* policy : {static_cast<Policy*>(&per_frame), static_cast<Policy*>(&arena), static_cast<Policy*>(&malloc_async), static_cast<Policy*>(&reused)}) {
    print(Runner(*policy, 3, dims, frames, warmup).run());
  }
  return EXIT_SUCCESS;
}
