# NVIDIA GeForce RTX 4050 Laptop GPU

20 SMs, sm_89, driver 13.2, CUDA runtime 13.0, Ubuntu 24.04, gcc 13.3.

1920x1080 RGB frames (6.2 MB in, 2.1 MB + 1 KB out), 300 frames per row with the first 20 discarded as warm-up. Phase times are GPU event means in ms; latency is host-observed, from before slot acquisition until the completion event was seen, in ms. Output of `./build/allocation_benchmark 300`:

```
== ring depth sweep (preallocated arena) ==
depth=1  h2d=0.944 kernel=0.025 d2h=0.342 | latency p50=3.58 p90=3.59 p99=3.90 | 435.9 fps blocked=299
depth=2  h2d=1.125 kernel=0.029 d2h=0.438 | latency p50=2.83 p90=3.46 p99=3.50 | 738.1 fps blocked=153
depth=3  h2d=1.728 kernel=0.048 d2h=0.542 | latency p50=3.68 p90=3.71 p99=3.80 | 848.0 fps blocked=290
depth=4  h2d=2.828 kernel=0.090 d2h=0.546 | latency p50=4.81 p90=4.84 p99=5.10 | 850.8 fps blocked=282
depth=8  h2d=4.076 kernel=0.098 d2h=0.548 | latency p50=9.65 p90=9.70 p99=9.77 | 833.2 fps blocked=259

== upload memory at depth 3 (preallocated arena) ==
cacheable upload slab       h2d=1.756 kernel=0.049 d2h=0.551 | latency p50=3.73 p90=3.76 p99=3.82 | 836.3 fps blocked=297
write-combined upload slab  h2d=1.724 kernel=0.048 d2h=0.541 | latency p50=3.68 p90=3.71 p99=4.10 | 845.5 fps blocked=288

== allocation strategies at depth 3 ==
cudaMalloc/cudaFree per frame  alloc=3.115 | latency p50=10.66 p90=11.28 p99=14.59 | 159.4 fps
preallocated arena             alloc=0.000 | latency p50= 3.69 p90= 3.73 p99= 3.82 | 838.3 fps
cudaMallocAsync/cudaFreeAsync  alloc=0.003 | latency p50= 3.67 p90= 3.71 p99= 4.22 | 834.4 fps
reused separate allocations    alloc=0.000 | latency p50= 3.65 p90= 3.68 p99= 3.86 | 853.3 fps
```

Kernel times, best of 200 runs at 1920x1080 with the input resident in L2:

```
one pixel per thread, 8100 blocks                   0.081 ms
grid-stride, 80 blocks                              0.026 ms
grid-stride, 80 blocks, 4 pixels/thread via u32     0.018 ms
```
