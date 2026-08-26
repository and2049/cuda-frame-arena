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
