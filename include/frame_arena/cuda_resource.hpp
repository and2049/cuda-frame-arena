#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace frame_arena {

class PinnedBuffer {
public:
  PinnedBuffer() noexcept = default;
  explicit PinnedBuffer(std::size_t bytes, unsigned flags = cudaHostAllocDefault);
  ~PinnedBuffer() noexcept;

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

  void* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

private:
  void release() noexcept;

  void* data_ = nullptr;
  std::size_t size_ = 0;
};

class DeviceBuffer {
public:
  DeviceBuffer() noexcept = default;
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer() noexcept;

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

private:
  void release() noexcept;

  void* data_ = nullptr;
  std::size_t size_ = 0;
};

class CudaStream {
public:
  CudaStream();
  ~CudaStream() noexcept;

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  cudaStream_t get() const noexcept { return stream_; }
  void synchronize() const;

private:
  void release() noexcept;

  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
  explicit CudaEvent(unsigned flags = cudaEventDisableTiming);
  ~CudaEvent() noexcept;

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept;
  CudaEvent& operator=(CudaEvent&& other) noexcept;

  cudaEvent_t get() const noexcept { return event_; }
  void record(cudaStream_t stream) const;
  cudaError_t query() const noexcept;
  void synchronize() const;
  float elapsed_ms_since(const CudaEvent& earlier) const;

private:
  void release() noexcept;

  cudaEvent_t event_ = nullptr;
};

}
