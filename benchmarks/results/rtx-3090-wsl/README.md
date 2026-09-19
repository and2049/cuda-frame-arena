# NVIDIA GeForce RTX 3090

82 SMs, sm_86, 6 MB L2, PCIe 4.0 x16, driver 13.4 (Windows 616.92), CUDA runtime 12.8, Ubuntu 24.04 under WSL2 on Windows 11, gcc 13.3, Ryzen 5 9600X host.

1920x1080 RGB frames (6.2 MB in, 2.1 MB + 1 KB out), 300 frames per row with the first 20 discarded as warm-up. Phase times are GPU event means in ms; latency is host-observed, from before slot acquisition until the completion event was seen, in ms. Output of `./build/allocation_benchmark 300`:

```
== ring depth sweep (preallocated arena) ==
depth=1  h2d=0.237 kernel=0.051 d2h=0.107 | latency p50=1.14 p90=1.23 p99=1.39 | 1313.4 fps blocked=299
depth=2  h2d=0.236 kernel=0.045 d2h=0.106 | latency p50=0.71 p90=0.80 p99=1.05 | 2721.8 fps blocked=153
depth=3  h2d=0.236 kernel=0.049 d2h=0.106 | latency p50=0.98 p90=1.04 p99=1.19 | 2743.9 fps blocked=5
depth=4  h2d=0.236 kernel=0.047 d2h=0.107 | latency p50=0.98 p90=1.33 p99=1.42 | 2737.2 fps blocked=0
depth=8  h2d=0.236 kernel=0.049 d2h=0.108 | latency p50=0.98 p90=1.29 p99=1.99 | 2642.0 fps blocked=0

== upload memory at depth 3 (preallocated arena) ==
cacheable upload slab       h2d=0.240 kernel=0.047 d2h=0.106 | latency p50=0.99 p90=1.04 p99=1.10 | 2733.8 fps blocked=2
write-combined upload slab  h2d=0.239 kernel=0.048 d2h=0.107 | latency p50=0.99 p90=1.06 p99=1.18 | 2717.0 fps blocked=16

== allocation strategies at depth 3 ==
cudaMalloc/cudaFree per frame  alloc=2.530 | latency p50= 6.78 p90= 7.19 p99=16.38 |  241.0 fps
preallocated arena             alloc=0.000 | latency p50= 0.98 p90= 1.05 p99= 1.26 | 2742.2 fps
cudaMallocAsync/cudaFreeAsync  alloc=0.002 | latency p50= 0.94 p90= 1.04 p99= 1.10 | 2648.0 fps
reused separate allocations    alloc=0.000 | latency p50= 0.96 p90= 1.04 p99= 1.17 | 2657.1 fps
```

Host and GPU are nearly balanced here: the copies run at 26 GB/s H2D and 20 GB/s D2H, so a frame occupies the GPU for about 0.39 ms, while the host loop (mostly the synthetic fill of the 6.2 MB input) takes about 0.37 ms per frame. From depth 3 the ring almost never blocks and throughput stays flat at roughly 2700 fps; deeper rings only add queueing to the tail latency. Write-combined upload memory makes no difference to the H2D rate here either.

Kernel times, best of 200 runs at 1920x1080 on the same device buffer (the 6.2 MB input exceeds the 6 MB L2, so it is not fully resident):

```
one pixel per thread, 8100 blocks                   0.096 ms
grid-stride, 328 blocks                             0.018 ms
grid-stride, 328 blocks, 4 pixels/thread via u32    0.015 ms
```
