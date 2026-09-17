cmake_minimum_required(VERSION 3.24)
project(cuda_frame_arena LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(frame_arena_core src/arena.cpp)
target_include_directories(frame_arena_core PUBLIC include)
target_compile_options(frame_arena_core PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

add_executable(arena_test tests/arena_test.cpp)
target_link_libraries(arena_test PRIVATE frame_arena_core)
target_include_directories(arena_test PRIVATE tests)
add_test(NAME arena_test COMMAND arena_test)

# The gpu dialect version of the pipeline is checked only when an MLIR build is
# available: point MLIR_OPT (or PATH) at mlir-opt from a build with the mlir
# project and the NVPTX target.
find_program(MLIR_OPT mlir-opt)
if(MLIR_OPT)
  add_test(NAME gpu_dialect_pipeline
           COMMAND ${CMAKE_COMMAND} -E env LLVM_BIN=$<PATH:GET_PARENT_PATH,${MLIR_OPT}>
                   ${CMAKE_SOURCE_DIR}/tools/run_gpu_dialect.sh ${CMAKE_BINARY_DIR}/gpu_dialect)
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler not found; building CPU-only targets")
  return()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
find_package(CUDAToolkit REQUIRED)

add_library(frame_arena src/cuda_error.cpp src/cuda_resource.cpp src/frame_ring.cpp src/grayscale.cu)
target_link_libraries(frame_arena PUBLIC frame_arena_core CUDA::cudart)
target_compile_options(frame_arena PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)

add_executable(frame_pipeline examples/grayscale_pipeline.cu)
target_link_libraries(frame_pipeline PRIVATE frame_arena)

add_executable(frame_ring_test tests/frame_ring_test.cu)
target_link_libraries(frame_ring_test PRIVATE frame_arena)
target_include_directories(frame_ring_test PRIVATE tests)
add_test(NAME frame_ring_test COMMAND frame_ring_test)

add_executable(allocation_benchmark benchmarks/allocation_benchmark.cu)
target_link_libraries(allocation_benchmark PRIVATE frame_arena)

