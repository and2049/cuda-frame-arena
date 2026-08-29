#include "frame_arena/cuda_error.hpp"

namespace frame_arena {

namespace {

std::string describe(cudaError_t code, std::string_view operation, std::source_location where) {
  std::string message(operation);
  message += " failed: ";
  message += cudaGetErrorName(code);
  message += " (";
  message += cudaGetErrorString(code);
  message += ") at ";
  message += where.file_name();
  message += ':';
  message += std::to_string(where.line());
  return message;
}

}

CudaError::CudaError(cudaError_t code, std::string_view operation, std::source_location where)
    : std::runtime_error(describe(code, operation, where)), code_(code) {}

}
