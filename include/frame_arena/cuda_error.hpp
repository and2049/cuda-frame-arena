#pragma once

#include <cuda_runtime.h>

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace frame_arena {

class CudaError : public std::runtime_error {
public:
  CudaError(cudaError_t code, std::string_view operation, std::source_location where);

  cudaError_t code() const noexcept { return code_; }

private:
  cudaError_t code_;
};

inline void check_cuda(cudaError_t code, std::string_view operation,
                       std::source_location where = std::source_location::current()) {
  if (code != cudaSuccess) throw CudaError(code, operation, where);
}

}

#define CUDA_CHECK(expr) ::frame_arena::check_cuda((expr), #expr)
