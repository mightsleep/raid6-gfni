/* Userspace shim so that kernel/avx512.c and kernel/recov_avx512.c compile unchanged. */
#ifndef KSHIM_H
#define KSHIM_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef uint8_t u8;
typedef uint64_t u64;
#define __aligned(x) __attribute__((aligned(x)))
#define __init

static inline void kernel_fpu_begin(void) {}
static inline void kernel_fpu_end(void) {}

extern u8 raid6_gfmul[256][256];
extern u8 raid6_vgfmul[256][32];
extern u8 raid6_gfexp[256];
extern u8 raid6_gflog[256];
extern u8 raid6_gfinv[256];
extern u8 raid6_gfexi[256];

/* ZERO_PAGE: static zero buffer, large enough even for the bench above 64 KiB */
extern u8 zero_page_buf[];
#define ZERO_PAGE(x) ((void *)zero_page_buf)
#define page_address(x) (x)

/* recov_*.c calls raid6_gen_syndrome, which is a static_call in the kernel */
void raid6_gen_syndrome(int disks, size_t bytes, void **ptrs);

#endif
