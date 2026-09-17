// The frame pipeline of frame_layout.hpp::enqueue_frame, written in the gpu
// dialect. The host function is deliberately synchronous: every gpu op runs
// in stream order and nothing carries a token. The gpu-async-region pass adds
// the !gpu.async.token dependencies, which is the same lifetime rule the
// FrameRing enforces with one stream and one completion event per slot.
//
//   tools/run_gpu_dialect.sh              # async-region, then gpu-to-llvm
//
// The kernel is the original one-pixel-per-thread grayscale_kernel: a shared
// (workgroup) histogram filled with atomics and flushed to global memory.

module attributes {gpu.container_module} {

  gpu.module @kernels {
    gpu.func @grayscale(%rgb: memref<?xi8>, %gray: memref<?xi8>,
                        %histogram: memref<256xi32>, %pixels: index)
        workgroup(%local: memref<256xi32, #gpu.address_space<workgroup>>)
        kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c256 = arith.constant 256 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32

      %tid = gpu.thread_id x
      %bdim = gpu.block_dim x
      %bid = gpu.block_id x

      scf.for %i = %tid to %c256 step %bdim {
        memref.store %zero, %local[%i] : memref<256xi32, #gpu.address_space<workgroup>>
      }
      gpu.barrier

      %base = arith.muli %bid, %bdim : index
      %idx = arith.addi %base, %tid : index
      %in_range = arith.cmpi ult, %idx, %pixels : index
      scf.if %in_range {
        %off = arith.muli %idx, %c3 : index
        %off_g = arith.addi %off, %c1 : index
        %off_b = arith.addi %off, %c2 : index
        %r8 = memref.load %rgb[%off] : memref<?xi8>
        %g8 = memref.load %rgb[%off_g] : memref<?xi8>
        %b8 = memref.load %rgb[%off_b] : memref<?xi8>
        %r = arith.extui %r8 : i8 to i32
        %g = arith.extui %g8 : i8 to i32
        %b = arith.extui %b8 : i8 to i32
        %wr = arith.constant 77 : i32
        %wg = arith.constant 150 : i32
        %wb = arith.constant 29 : i32
        %shift = arith.constant 8 : i32
        %tr = arith.muli %r, %wr : i32
        %tg = arith.muli %g, %wg : i32
        %tb = arith.muli %b, %wb : i32
        %s0 = arith.addi %tr, %tg : i32
        %s1 = arith.addi %s0, %tb : i32
        %y = arith.shrui %s1, %shift : i32
        %y8 = arith.trunci %y : i32 to i8
        memref.store %y8, %gray[%idx] : memref<?xi8>
        %bin = arith.index_cast %y : i32 to index
        %old = memref.atomic_rmw addi %one, %local[%bin] : (i32, memref<256xi32, #gpu.address_space<workgroup>>) -> i32
      }
      gpu.barrier

      scf.for %i = %tid to %c256 step %bdim {
        %count = memref.load %local[%i] : memref<256xi32, #gpu.address_space<workgroup>>
        %nonzero = arith.cmpi ne, %count, %zero : i32
        scf.if %nonzero {
          %old = memref.atomic_rmw addi %count, %histogram[%i] : (i32, memref<256xi32>) -> i32
        }
      }
      gpu.return
    }
  }

  // One frame: H2D copy of rgb, histogram clear, kernel, D2H copies of gray
  // and histogram. Host memrefs are the slot's upload and download arenas,
  // device memrefs its device arena.
  func.func @enqueue_frame(%host_rgb: memref<?xi8>, %host_gray: memref<?xi8>,
                           %host_histogram: memref<256xi32>,
                           %device_rgb: memref<?xi8>, %device_gray: memref<?xi8>,
                           %device_histogram: memref<256xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c255 = arith.constant 255 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0 : i32

    %pixels = memref.dim %host_gray, %c0 : memref<?xi8>
    %rounded = arith.addi %pixels, %c255 : index
    %grid = arith.divui %rounded, %c256 : index

    gpu.memcpy %device_rgb, %host_rgb : memref<?xi8>, memref<?xi8>
    gpu.memset %device_histogram, %zero : memref<256xi32>, i32
    gpu.launch_func @kernels::@grayscale
        blocks in (%grid, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%device_rgb : memref<?xi8>, %device_gray : memref<?xi8>,
             %device_histogram : memref<256xi32>, %pixels : index)
    gpu.memcpy %host_gray, %device_gray : memref<?xi8>, memref<?xi8>
    gpu.memcpy %host_histogram, %device_histogram : memref<256xi32>, memref<256xi32>
    return
  }
}
