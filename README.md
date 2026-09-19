# CUDA Asynchronous Frame Arena

A small frame pipeline that preallocates everything it needs up front: two pinned host slabs (upload and download) and one device slab, all carved into fixed ring slots. A slot only goes back to the caller once the CUDA event recorded on its stream reports that the GPU is done with it.

This is not an attempt at a faster allocator — CUDA already ships a stream-ordered one, and `cudaMallocAsync` matches this design on this machine (see `benchmarks/results/`). The point is to make the lifetime rules of asynchronous work explicit: host program order, stream order, submission and completion are four different things, and a ring index tracks none of them.

```
Upload slab (pinned, write-combined)   Download slab (pinned, cacheable)   Device slab (cudaMalloc)
+--------+--------+--------+          +--------+--------+--------+      +--------+--------+--------+
| slot 0 | slot 1 | slot 2 |          | slot 0 | slot 1 | slot 2 |      | slot 0 | slot 1 | slot 2 |
+--------+--------+--------+          +--------+--------+--------+      +--------+--------+--------+
  stream i, event i                     stream i, event i                   stream i, event i

CPU writes RGB into an upload slot ──cudaMemcpyAsync──▶ device slot
                                                        │ grayscale + histogram kernel
CPU reads gray, histogram and metadata ◀──cudaMemcpyAsync── device slot
the slot is reusable once cudaEventQuery(completion) == cudaSuccess
```

## Layout

```
include/frame_arena/
  checked_math.hpp    is_pow2, checked_add, checked_mul, overflow-safe align_up
  arena.hpp           non-owning bump arena over a [base, base+capacity) byte range
  cuda_error.hpp      CudaError (cudaError_t + operation + source_location), CUDA_CHECK
  cuda_resource.hpp   PinnedBuffer, DeviceBuffer, CudaStream, CudaEvent (rule of five)
  frame_ring.hpp      FrameRing, FrameLease, FrameSlot, SlotState, SlotLayout, UploadMemory
  grayscale.hpp       kernel launcher and FrameMetadata
  frame_layout.hpp    per-frame region carving, synthetic input, CPU reference check
src/                  implementations
examples/grayscale_pipeline.cu    demo: frame_pipeline [depth] [frames] [drop]
tests/arena_test.cpp              CPU-only, no GPU needed
tests/frame_ring_test.cu          CUDA integration tests
benchmarks/allocation_benchmark.cu
benchmarks/results/               recorded runs, one directory per machine
mlir/, tools/                     side exercises on the same kernel, described below
```

`mlir/grayscale_pipeline.mlir` is the same pipeline written in the gpu dialect, and the two scripts under `tools/` produce the device-side LLVM IR, PTX and SASS of `grayscale.cu` and lower the MLIR through `gpu-async-region`, `gpu-to-llvm` and NVVM. Both need an LLVM build with the NVPTX target and the `mlir` project (`clang`, `llc` and `mlir-opt` on `PATH` or in `$LLVM_BIN`). They are reading exercises, not part of the pipeline.

## Build and run

Requires CMake 3.24+, a C++20 compiler and CUDA 12+ (developed on 13.0). Without `nvcc` only `arena_test` is built; if `nvcc` is not on `PATH`, pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure

./build/frame_pipeline 3 64          # ring depth 3, 64 frames, block when the ring is full
./build/frame_pipeline 3 200 drop    # drop frames instead of blocking
./build/allocation_benchmark 300     # 300 frames per configuration
```

The demo and the tests report zero errors and zero hazards under `compute-sanitizer --tool memcheck` and `--tool racecheck`.

Recorded runs are kept in `benchmarks/results/`, one directory per machine, next to the hardware and toolkit they were taken on.

## Slot lifecycle

```
Available ──acquire()──▶ HostWriting ──submit()──▶ InFlight ──event completes──▶ Available
```

`try_acquire()` looks at the next slot in ring order and queries its completion event: `cudaSuccess` runs the slot's `on_retire` callback (if any), resets the arenas and returns a `FrameLease`; `cudaErrorNotReady` returns `std::nullopt`; anything else throws `CudaError`. The blocking policy lives in `acquire()`, which synchronizes on the event and counts the wait in `RingStats`.

`FrameLease::submit()` records the completion event on the slot's stream after everything the caller enqueued, and dropping a lease without submitting submits it implicitly, so a slot can never be stranded in `HostWriting`. A slot that is still `HostWriting` is not ready either: `try_acquire()` returns `std::nullopt` and `acquire()` waits for its `submit()`.

The `on_retire` callback exists because the best moment to verify a frame is right before its slot is recycled: the event has fired, the D2H copy has landed in pinned memory, and the arenas have not been reset yet. The demo uses it to run the CPU reference check.

## Threads

A decoder thread and an inference thread can share one ring: the decoder acquires a lease, fills the upload arena and hands the lease over, and the inference thread enqueues its work and submits. One mutex in `FrameRing` guards what the two contend for: the ring index, the slot states and the stats. Everything reachable through a lease (its arenas, stream and `on_retire` callback) belongs to whoever holds the lease and is not synchronized, the same way the memory behind a `unique_ptr` is not.

The lock is never held across a wait. `acquire()` blocks on the completion event with the lock released, because the `submit()` that records that event needs the lock, and because a `try_acquire()` on the other thread should keep returning `std::nullopt` rather than queue behind a GPU wait. A leased slot is a different wait: with two threads it means the other stage has not submitted yet, so `acquire()` sleeps on a condition variable that `submit()` notifies. After either wait the slot may have gone to another caller, so the next slot is checked again. The retire callback runs after the slot has been claimed and the lock dropped, so a slow verification does not stall the other thread. `drain()` is the exception and holds the lock until every in-flight slot has retired.

Two things stay with the caller. Ring order is shared, so a thread that holds the next slot's lease and calls `acquire()` on the same ring waits for itself. And `slot(i)` returns the slot as is, for the tests and the benchmark; reading its `state` from another thread races with the ring.

`test_leases_move_between_threads` is the deadlock check: with depth 2 the producer blocks in `acquire()` on an event that only a `submit()` from the consumer thread will record.

## Arena

`Arena` is a bump allocator over memory it does not own. `allocate(bytes, alignment)` rejects alignments that are not a nonzero power of two, rounds the offset up with an overflow-checked `align_up`, adds `bytes` with another overflow check, compares against capacity, and only then advances the offset. A failed allocation leaves the offset untouched, and a zero-byte allocation returns the aligned cursor without advancing past the padding. `allocate<T>(count)` checks `count * sizeof(T)` for overflow and requires `T` to be trivially copyable, since nothing is ever constructed or destroyed in arena storage.

One offset-based implementation serves the host and the device slabs. `std::align` would work on the host, where the pointer really is host-addressable, but the device arena only ever hands opaque addresses to CUDA APIs; keeping the arithmetic in integer offsets makes that distinction visible and avoids repeated `uintptr_t` round trips.

Each frame is carved with deliberately varied alignments so the padding shows up in `used()`:

```
region             alignment               host arena   device arena
RGB input          256                     upload       device
grayscale output   128                     download     device
histogram          256                     download     device
metadata           alignof(FrameMetadata)  download     device
```

## Host memory by direction

Each slot owns two host arenas rather than one: `carve_frame` takes the RGB region from the lease's `upload_arena()` and everything else from its `download_arena()` (on the device both are the same `device_arena()`). The pinned footprint is unchanged, `pixels * 3` upload plus `pixels + 1 KB` download per slot, but the caching attribute is now chosen per direction.

The two regions have opposite access patterns: the CPU writes the upload region and the copy engine reads it, while the copy engine writes the download region and the CPU reads it. That makes upload a candidate for write-combined pinned memory (`cudaHostAllocWriteCombined`), which bypasses the CPU caches and is documented as faster to transfer on some systems. The catch is that CPU reads from it are uncached — a strided read of the 6 MB frame takes 27 ms from write-combined memory against 0.2 ms from cacheable memory on this machine — so the split makes the rule enforceable: nothing reads `upload_arena()` back, and the CPU reference check regenerates the synthetic input instead.

`FrameRing` takes an `UploadMemory` argument, default `WriteCombined`, and both settings are measured in `benchmarks/results/`. The H2D copy runs at the same 6.5 GB/s either way on this laptop, so here the split is a layout decision rather than a speedup; the enum is where a different machine changes its mind. Pinned memory is a limited system resource and `cudaHostAlloc` is slow, so keep the ring's footprint (`depth * (upload + download)` on the host, `depth * device` on the GPU) as small as the pipeline needs.

## Kernel

The kernel computes luma and a 256-bin histogram in one pass, accumulating the histogram in shared memory with `atomicAdd` and flushing it to global memory once per block. That flush, not memory bandwidth, is what bounds the kernel: the first version launched one thread per pixel (8100 blocks at 1920x1080, up to 256 bins flushed each, roughly one global atomic per pixel), while sizing the grid by the device (four blocks per SM with a grid-stride loop) cuts the flushes by two orders of magnitude. Reading four pixels per thread as three 32-bit words replaces twelve `LDG.E.U8` and four `STG.E.U8` with three `LDG.E` and one `STG.E`. The three versions are timed side by side in `benchmarks/results/`.

The 32-bit path needs `rgb` and `gray` to be 4-byte aligned, which the arenas guarantee and `launch_grayscale` checks; pixels past the last multiple of four take the byte path.

## Ownership and errors

`PinnedBuffer`, `DeviceBuffer`, `CudaStream` and `CudaEvent` are move-only; a moved-from object is empty and its destructor does nothing. Destructors never throw — a failing `cudaFree`, `cudaFreeHost`, `cudaStreamDestroy` or `cudaEventDestroy` is written to stderr and asserted in debug builds.

`CudaError` carries the `cudaError_t`, the expression that failed and the `std::source_location` of the check, so messages read like `cudaMalloc(&p, ~std::size_t{0}) failed: cudaErrorMemoryAllocation (out of memory) at tests/frame_ring_test.cu:74`. After a kernel launch, `cudaGetLastError()` is checked separately, because a launch failure and an asynchronous execution failure surface at different points.

## Limitations / TODO

- one CUDA device, one stream per slot
- fixed number and size of slots, chosen at construction
- no individual deallocation; arenas reset only when a slot is retired
- trivially copyable byte buffers only; no constructors or destructors run in arena storage
- a failed multi-region carve leaves its partial allocations in place until the next reset
- no sharing an allocation between unrelated streams, and no attempt to replace the CUDA stream-ordered allocator

TODO: `std::pmr::memory_resource` adapters, multi-GPU, free lists, CUDA Graphs, mapped or unified memory. The gpu dialect version is a lowering exercise, not a runnable replacement: it has no ring, no verification, and its kernel is the unvectorized one.
