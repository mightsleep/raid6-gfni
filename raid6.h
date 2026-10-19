/* Shared declarations for raid6.c (harness without SIMD), raid6_zmm.c and raid6_ymm.c. */
#ifndef RAID6_H
#define RAID6_H
#include "kshim.h"
#include "kernel/algos.h"

#define MAX_BYTES (64 * 1024)
#define MAX_DISKS 24

/* matrices for vgf2p8affineqb; in the kernel this would be a 2 KiB table from mktables.c */
extern u64 raid6_gfmat[256];
#define GFMAT(c) raid6_gfmat[(u8)(c)]

u8 gfmul(u8 a, u8 b);

/* raid6_zmm.c: -mavx512f/bw/vl/dq -mgfni */
void bw_gen_syndrome(int disks, size_t bytes, void **ptrs);
void gfni_gen_syndrome(int disks, size_t bytes, void **ptrs);
void gfni_gen_syndrome_t(int disks, size_t bytes, void **ptrs);
void gfni_direct_gen_syndrome(int disks, size_t bytes, void **ptrs);
void gfni_h1_gen_syndrome(int disks, size_t bytes, void **ptrs);
void gfni_xor_syndrome(int disks, int start, int stop, size_t bytes, void **ptrs);
void gfni_recov_2data(int disks, size_t bytes, int faila, int failb, void **ptrs);
void gfni_recov_datap(int disks, size_t bytes, int faila, void **ptrs);
void gfni_recov_2data_fused(int disks, size_t bytes, int faila, int failb, void **ptrs);
int zmm_affine_errors(void);
void zmm_warmup(double ns);
void zmm_bench_insn(double tcyc);

/* raid6_ymm.c: only -mavx2 -mgfni, so VEX encoding only */
void gfni256x2_gen_syndrome(int disks, size_t bytes, void **ptrs);
void gfni256x4_gen_syndrome(int disks, size_t bytes, void **ptrs);
void gfni256x2_xor_syndrome(int disks, int start, int stop, size_t bytes, void **ptrs);
void gfni256x4_xor_syndrome(int disks, int start, int stop, size_t bytes, void **ptrs);
void gfni256_recov_2data(int disks, size_t bytes, int faila, int failb, void **ptrs);
void gfni256_recov_datap(int disks, size_t bytes, int faila, void **ptrs);
int ymm_affine_errors(void);
u8 xmm_gf2p8mulb(u8 a, u8 b);

#endif
