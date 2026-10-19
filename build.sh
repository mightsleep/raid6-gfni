#!/bin/sh
# build into ~/.cache/raid6-gfni-build (not /tmp, which is tmpfs in RAM); OUT and CC can be
# overridden from the environment (CI: CC=gcc from the distribution)
set -eu
cd "$(dirname "$0")"
CC=${CC:-/nix/store/j1938gsbvbhkyz4kbmvjk56r479mkw3s-gcc-wrapper-16.2.0/bin/gcc}
OUT=${OUT:-$HOME/.cache/raid6-gfni-build}
mkdir -p "$OUT"
COMMON="-O2 -g -Wall -Wno-unused-function -DCONFIG_X86_64 -Ifakeinc -include kshim.h"
# the kernel code writes ymm/zmm in inline asm without clobbers; as in the kernel, the
# compiler must not use SIMD registers on its own
for f in avx2 recov_avx2 avx512 recov_avx512; do
	$CC $COMMON -mgeneral-regs-only -c kernel/$f.c -o "$OUT/$f.o"
done
# harness and reference without SIMD flags, SIMD only in files picked at run time
$CC $COMMON -c raid6.c -o "$OUT/raid6.o"
$CC $COMMON -mavx512f -mavx512bw -mavx512vl -mavx512dq -mgfni -c raid6_zmm.c -o "$OUT/raid6_zmm.o"
# VEX only: without avx512 flags the compiler cannot emit EVEX
$CC $COMMON -mavx2 -mgfni -c raid6_ymm.c -o "$OUT/raid6_ymm.o"
$CC "$OUT"/avx2.o "$OUT"/recov_avx2.o "$OUT"/avx512.o "$OUT"/recov_avx512.o \
	"$OUT"/raid6.o "$OUT"/raid6_zmm.o "$OUT"/raid6_ymm.o -o "$OUT/raid6"
