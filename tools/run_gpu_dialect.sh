#!/bin/bash
# Lower mlir/grayscale_pipeline.mlir through gpu-async-region, gpu-to-llvm and
# gpu-module-to-binary, keeping each stage: tools/run_gpu_dialect.sh [out_dir]
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$repo/build/gpu_dialect}
bin=${LLVM_BIN:+$LLVM_BIN/}
arch=${ARCH:-sm_89}
input=$repo/mlir/grayscale_pipeline.mlir

mkdir -p "$out"
# 1. Tokens only: the host function is still in func/memref/gpu dialects.
"${bin}mlir-opt" "$input" --gpu-async-region -o "$out/1-async.mlir"

# 2. Kernel first, since gpu-to-llvm checks launch operands against the kernel
#    signature; convert-gpu-to-nvvm does not cover scf, hence scf-to-cf before it.
kernel_passes="convert-scf-to-cf,convert-gpu-to-nvvm,reconcile-unrealized-casts"
"${bin}mlir-opt" "$out/1-async.mlir" \
  --pass-pipeline="builtin.module(gpu.module($kernel_passes),gpu-to-llvm)" \
  -o "$out/2-llvm.mlir"

# 3. Same, plus a target attribute and PTX generation through the NVPTX backend.
"${bin}mlir-opt" "$out/1-async.mlir" \
  --pass-pipeline="builtin.module(gpu.module($kernel_passes),nvvm-attach-target{chip=$arch},gpu-to-llvm,gpu-module-to-binary{format=isa})" \
  -o "$out/3-binary.mlir"

printf '%s\n' "$out"/*.mlir
