#!/bin/bash
# Run the gpu dialect pipeline in mlir/grayscale_pipeline.mlir through the
# passes that matter for this project and keep each stage:
#
#   1. gpu-async-region      adds !gpu.async.token dependencies between the
#                            memcpy/memset/launch ops (the stream order)
#   2. convert-gpu-to-nvvm,  lowers the kernel to NVVM and the host side to
#      gpu-to-llvm           GPU runtime calls (mgpuMemcpy, mgpuStreamCreate, ...)
#   3. gpu-module-to-binary  with the NVPTX target built, turns the NVVM
#                            kernel into PTX embedded in a gpu.binary
#
#   tools/run_gpu_dialect.sh [out_dir]
#
# Needs mlir-opt from an LLVM build with the mlir project (on PATH or in
# $LLVM_BIN).
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$repo/build/gpu_dialect}
bin=${LLVM_BIN:+$LLVM_BIN/}
arch=${ARCH:-sm_89}
input=$repo/mlir/grayscale_pipeline.mlir

mkdir -p "$out"
# 1. Tokens only: the host function is still in func/memref/gpu dialects.
"${bin}mlir-opt" "$input" --gpu-async-region -o "$out/1-async.mlir"

# 2. Kernel to NVVM first (gpu-to-llvm checks launch operands against the
#    kernel signature, and memref arguments expand to descriptors on both
#    sides), then host to LLVM with runtime calls. convert-gpu-to-nvvm covers
#    arith, memref and cf but not scf, so the loops are lowered to cf first.
kernel_passes="convert-scf-to-cf,convert-gpu-to-nvvm,reconcile-unrealized-casts"
"${bin}mlir-opt" "$out/1-async.mlir" \
  --pass-pipeline="builtin.module(gpu.module($kernel_passes),gpu-to-llvm)" \
  -o "$out/2-llvm.mlir"

# 3. Same, plus a target attribute and PTX generation through the NVPTX backend.
"${bin}mlir-opt" "$out/1-async.mlir" \
  --pass-pipeline="builtin.module(gpu.module($kernel_passes),nvvm-attach-target{chip=$arch},gpu-to-llvm,gpu-module-to-binary{format=isa})" \
  -o "$out/3-binary.mlir"

printf '%s\n' "$out"/*.mlir
