#!/bin/bash
# Emit the device-side LLVM IR, PTX and SASS for src/grayscale.cu.
#
#   tools/emit_device_ir.sh [out_dir] [git_rev]
#
# Needs an LLVM build with the NVPTX target (clang, llc, opt on PATH or in
# $LLVM_BIN) and the CUDA toolkit for headers and ptxas ($CUDA_PATH). With a
# git revision the sources are taken from that commit, so two runs give a
# before/after pair, for example:
#
#   tools/emit_device_ir.sh build/ir/before 637c67d
#   tools/emit_device_ir.sh build/ir/after
#   diff build/ir/{before,after}/grayscale.O2.ll
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$repo/build/device_ir}
rev=${2:-}
arch=${ARCH:-sm_89}
cuda=${CUDA_PATH:-/usr/local/cuda}
bin=${LLVM_BIN:+$LLVM_BIN/}

mkdir -p "$out"
src=$repo
if [[ -n $rev ]]; then
  src=$out/src
  rm -rf "$src" && mkdir -p "$src"
  git -C "$repo" archive "$rev" include src | tar -x -C "$src"
fi

clang_flags=(-x cuda --cuda-device-only --cuda-gpu-arch="$arch" --cuda-path="$cuda" -std=c++20
             -I"$src/include" -Wno-unknown-cuda-version)

"${bin}clang++" "${clang_flags[@]}" -O0 -S -emit-llvm -o "$out/grayscale.O0.ll" "$src/src/grayscale.cu"
"${bin}clang++" "${clang_flags[@]}" -O2 -S -emit-llvm -o "$out/grayscale.O2.ll" "$src/src/grayscale.cu"
"${bin}llc" -march=nvptx64 -mcpu="$arch" -O2 -o "$out/grayscale.ptx" "$out/grayscale.O2.ll"

if [[ -x $cuda/bin/ptxas ]]; then
  "$cuda/bin/ptxas" -arch="$arch" -v -o "$out/grayscale.cubin" "$out/grayscale.ptx" 2> "$out/ptxas.txt"
  "$cuda/bin/cuobjdump" -sass "$out/grayscale.cubin" > "$out/grayscale.sass"
fi

printf '%s\n' "$out"/grayscale.*
