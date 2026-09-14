#include "frame_arena/cuda_resource.hpp"
#include "frame_arena/cuda_error.hpp"

#include <cassert>
#include <cstdio>

namespace frame_arena {

namespace {

void warn_on_failure(cudaError_t code, const char* operation) noexcept {
  if (code == cudaSuccess) return;
  std::fprintf(stderr, "%s failed during cleanup: %s\n", operation, cudaGetErrorString(code));
  assert(false && "CUDA cleanup failed");
}

}

PinnedBuffer::PinnedBuffer(std::size_t bytes, unsigned flags) : size_(bytes) {
  CUDA_CHECK(cudaHostAlloc(&data_, bytes, flags));
}

PinnedBuffer::~PinnedBuffer() noexcept { release(); }

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void PinnedBuffer::release() noexcept {
  if (data_ != nullptr) warn_on_failure(cudaFreeHost(data_), "cudaFreeHost");
  data_ = nullptr;
  size_ = 0;
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) : size_(bytes) {
  CUDA_CHECK(cudaMalloc(&data_, bytes));
}

DeviceBuffer::~DeviceBuffer() noexcept { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void DeviceBuffer::release() noexcept {
  if (data_ != nullptr) warn_on_failure(cudaFree(data_), "cudaFree");
  data_ = nullptr;
  size_ = 0;
}

CudaStream::CudaStream() {
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CudaStream::~CudaStream() noexcept { release(); }

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    release();
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

void CudaStream::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void CudaStream::release() noexcept {
  if (stream_ != nullptr) warn_on_failure(cudaStreamDestroy(stream_), "cudaStreamDestroy");
  stream_ = nullptr;
}

CudaEvent::CudaEvent(unsigned flags) {
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() noexcept { release(); }

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  if (this != &other) {
    release();
    event_ = std::exchange(other.event_, nullptr);
  }
  return *this;
}

void CudaEvent::record(cudaStream_t stream) const { CUDA_CHECK(cudaEventRecord(event_, stream)); }

cudaError_t CudaEvent::query() const noexcept { return cudaEventQuery(event_); }

void CudaEvent::synchronize() const { CUDA_CHECK(cudaEventSynchronize(event_)); }

float CudaEvent::elapsed_ms_since(const CudaEvent& earlier) const {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, earlier.event_, event_));
  return ms;
}

void CudaEvent::release() noexcept {
  if (event_ != nullptr) warn_on_failure(cudaEventDestroy(event_), "cudaEventDestroy");
  event_ = nullptr;
}

}
