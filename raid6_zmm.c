/*
 * AVX-512 part: bw-tuned (AVX-512BW without GFNI) and GFNI with zmm. Compiled
 * with -mavx512f/bw/vl/dq -mgfni, called only when __builtin_cpu_supports
 * avx512bw (+ gfni).
 */
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include "raid6.h"

/* ---------- (b') AVX-512BW without GFNI: multiplication by 2 in 4 instructions ---------- */

#define LD(ptr) _mm512_load_si512((const void *)(ptr))
#define XOR3 0x96

static inline __m512i mul2_bw(__m512i q, __m512i c1d)
{
	__mmask64 k = _mm512_movepi8_mask(q);		/* vpmovb2m */
	__m512i t = _mm512_maskz_mov_epi8(k, c1d);	/* vmovdqu8{z} */
	return _mm512_xor_si512(_mm512_add_epi8(q, q), t);
}

__attribute__((noinline)) void bw_gen_syndrome(int disks, size_t bytes,
						      void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[z0 + 1], *q = d[z0 + 2];
	const __m512i c1d = _mm512_set1_epi8(0x1d);

	for (size_t o = 0; o < bytes; o += 128) {
		__m512i p0 = LD(d[z0] + o), p1 = LD(d[z0] + o + 64);
		__m512i q0 = p0, q1 = p1;
		for (int z = z0 - 1; z >= 0; z--) {
			__m512i a0 = LD(d[z] + o), a1 = LD(d[z] + o + 64);
			__mmask64 k0 = _mm512_movepi8_mask(q0);
			__mmask64 k1 = _mm512_movepi8_mask(q1);
			__m512i t0 = _mm512_maskz_mov_epi8(k0, c1d);
			__m512i t1 = _mm512_maskz_mov_epi8(k1, c1d);
			q0 = _mm512_ternarylogic_epi64(_mm512_add_epi8(q0, q0), t0, a0, XOR3);
			q1 = _mm512_ternarylogic_epi64(_mm512_add_epi8(q1, q1), t1, a1, XOR3);
			p0 = _mm512_xor_si512(p0, a0);
			p1 = _mm512_xor_si512(p1, a1);
		}
		_mm512_stream_si512((void *)(p + o), p0);
		_mm512_stream_si512((void *)(p + o + 64), p1);
		_mm512_stream_si512((void *)(q + o), q0);
		_mm512_stream_si512((void *)(q + o + 64), q1);
	}
	_mm_sfence();
}

/* ---------- (c) GFNI ---------- */

#define AFF(x, m) _mm512_gf2p8affine_epi64_epi8((x), (m), 0)

#define STORE512(nt, ptr, v) \
	((nt) ? _mm512_stream_si512((ptr), (v)) : _mm512_store_si512((ptr), (v)))

/*
 * Horner scheme over two disks:
 *   Q = 4*Q ^ 2*d[z] ^ d[z-1],  P = P ^ d[z] ^ d[z-1]
 * so 2x vgf2p8affineqb + 2x vpternlogq per two disks and 64 B.
 */
static inline __attribute__((always_inline)) void gfni_gen_body(int disks, size_t bytes,
								 void **ptrs, int nt)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[z0 + 1], *q = d[z0 + 2];
	const __m512i m2 = _mm512_set1_epi64(GFMAT(2));
	const __m512i m4 = _mm512_set1_epi64(GFMAT(4));

	for (size_t o = 0; o < bytes; o += 128) {
		__m512i p0 = LD(d[z0] + o), p1 = LD(d[z0] + o + 64);
		__m512i q0 = p0, q1 = p1;
		int z = z0 - 1;
		for (; z >= 1; z -= 2) {
			__m512i a0 = LD(d[z] + o), a1 = LD(d[z] + o + 64);
			__m512i b0 = LD(d[z - 1] + o), b1 = LD(d[z - 1] + o + 64);
			p0 = _mm512_ternarylogic_epi64(p0, a0, b0, XOR3);
			p1 = _mm512_ternarylogic_epi64(p1, a1, b1, XOR3);
			q0 = _mm512_ternarylogic_epi64(AFF(q0, m4), AFF(a0, m2), b0, XOR3);
			q1 = _mm512_ternarylogic_epi64(AFF(q1, m4), AFF(a1, m2), b1, XOR3);
		}
		if (z == 0) {
			__m512i a0 = LD(d[0] + o), a1 = LD(d[0] + o + 64);
			p0 = _mm512_xor_si512(p0, a0);
			p1 = _mm512_xor_si512(p1, a1);
			q0 = _mm512_xor_si512(AFF(q0, m2), a0);
			q1 = _mm512_xor_si512(AFF(q1, m2), a1);
		}
		STORE512(nt, (void *)(p + o), p0);
		STORE512(nt, (void *)(p + o + 64), p1);
		STORE512(nt, (void *)(q + o), q0);
		STORE512(nt, (void *)(q + o + 64), q1);
	}
	if (nt)
		_mm_sfence();
}

/*
 * Direct sum Q = sum g^z * d[z]: each disk has its own matrix (embedded
 * broadcast from the table), the affines are independent and the chain through Q is
 * only vpternlogq (1 cycle) per two disks. Same instruction count as Horner over
 * two disks.
 */
__attribute__((noinline)) void gfni_direct_gen_syndrome(int disks, size_t bytes,
							       void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[z0 + 1], *q = d[z0 + 2];
	u64 mt[MAX_DISKS];

	for (int z = 0; z <= z0; z++)
		mt[z] = GFMAT(raid6_gfexp[z]);

	for (size_t o = 0; o < bytes; o += 128) {
		__m512i p0 = LD(d[z0] + o), p1 = LD(d[z0] + o + 64);
		__m512i mz0 = _mm512_set1_epi64(mt[z0]);
		__m512i q0 = AFF(p0, mz0), q1 = AFF(p1, mz0);
		int z = z0 - 1;
		for (; z >= 2; z -= 2) {
			__m512i ma = _mm512_set1_epi64(mt[z]), mb = _mm512_set1_epi64(mt[z - 1]);
			__m512i a0 = LD(d[z] + o), a1 = LD(d[z] + o + 64);
			__m512i b0 = LD(d[z - 1] + o), b1 = LD(d[z - 1] + o + 64);
			p0 = _mm512_ternarylogic_epi64(p0, a0, b0, XOR3);
			p1 = _mm512_ternarylogic_epi64(p1, a1, b1, XOR3);
			q0 = _mm512_ternarylogic_epi64(q0, AFF(a0, ma), AFF(b0, mb), XOR3);
			q1 = _mm512_ternarylogic_epi64(q1, AFF(a1, ma), AFF(b1, mb), XOR3);
		}
		__m512i e0 = LD(d[0] + o), e1 = LD(d[0] + o + 64);
		if (z == 1) {
			__m512i m1 = _mm512_set1_epi64(mt[1]);
			__m512i a0 = LD(d[1] + o), a1 = LD(d[1] + o + 64);
			p0 = _mm512_ternarylogic_epi64(p0, a0, e0, XOR3);
			p1 = _mm512_ternarylogic_epi64(p1, a1, e1, XOR3);
			q0 = _mm512_ternarylogic_epi64(q0, AFF(a0, m1), e0, XOR3);
			q1 = _mm512_ternarylogic_epi64(q1, AFF(a1, m1), e1, XOR3);
		} else {
			p0 = _mm512_xor_si512(p0, e0);
			p1 = _mm512_xor_si512(p1, e1);
			q0 = _mm512_xor_si512(q0, e0);
			q1 = _mm512_xor_si512(q1, e1);
		}
		_mm512_stream_si512((void *)(p + o), p0);
		_mm512_stream_si512((void *)(p + o + 64), p1);
		_mm512_stream_si512((void *)(q + o), q0);
		_mm512_stream_si512((void *)(q + o + 64), q1);
	}
	_mm_sfence();
}

/*
 * ISA-L pq_gen_avx512_gfni style: Horner one disk at a time, Q = 2*(Q ^ d),
 * 1 affine + 2 vpxorq per disk, chain affine + xor (4 cycles) per disk.
 */
__attribute__((noinline)) void gfni_h1_gen_syndrome(int disks, size_t bytes,
							   void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[z0 + 1], *q = d[z0 + 2];
	const __m512i m2 = _mm512_set1_epi64(GFMAT(2));

	for (size_t o = 0; o < bytes; o += 128) {
		__m512i p0 = LD(d[z0] + o), p1 = LD(d[z0] + o + 64);
		__m512i q0 = p0, q1 = p1;
		for (int z = z0 - 1; z >= 0; z--) {
			__m512i a0 = LD(d[z] + o), a1 = LD(d[z] + o + 64);
			p0 = _mm512_xor_si512(p0, a0);
			p1 = _mm512_xor_si512(p1, a1);
			q0 = _mm512_xor_si512(AFF(q0, m2), a0);
			q1 = _mm512_xor_si512(AFF(q1, m2), a1);
		}
		_mm512_stream_si512((void *)(p + o), p0);
		_mm512_stream_si512((void *)(p + o + 64), p1);
		_mm512_stream_si512((void *)(q + o), q0);
		_mm512_stream_si512((void *)(q + o + 64), q1);
	}
	_mm_sfence();
}

/* like the kernel: P and Q with non-temporal stores */
__attribute__((noinline)) void gfni_gen_syndrome(int disks, size_t bytes, void **ptrs)
{
	gfni_gen_body(disks, bytes, ptrs, 1);
}

/* plain stores: for recov, which reads P and Q right back */
__attribute__((noinline)) void gfni_gen_syndrome_t(int disks, size_t bytes, void **ptrs)
{
	gfni_gen_body(disks, bytes, ptrs, 0);
}

/*
 * Right side as in gen, left side (disks below start) one multiplication by the
 * constant 2^start instead of start multiplications by 2.
 */
__attribute__((noinline)) void gfni_xor_syndrome(int disks, int start,
							int stop, size_t bytes,
							void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	u8 *p = d[disks - 2], *q = d[disks - 1];
	const __m512i m2 = _mm512_set1_epi64(GFMAT(2));
	const __m512i m4 = _mm512_set1_epi64(GFMAT(4));
	const __m512i ms = _mm512_set1_epi64(GFMAT(raid6_gfexp[start]));

	for (size_t o = 0; o < bytes; o += 128) {
		__m512i q0 = LD(d[stop] + o), q1 = LD(d[stop] + o + 64);
		__m512i p0 = _mm512_xor_si512(q0, LD(p + o));
		__m512i p1 = _mm512_xor_si512(q1, LD(p + o + 64));
		int z = stop - 1;
		for (; z >= start + 1; z -= 2) {
			__m512i a0 = LD(d[z] + o), a1 = LD(d[z] + o + 64);
			__m512i b0 = LD(d[z - 1] + o), b1 = LD(d[z - 1] + o + 64);
			p0 = _mm512_ternarylogic_epi64(p0, a0, b0, XOR3);
			p1 = _mm512_ternarylogic_epi64(p1, a1, b1, XOR3);
			q0 = _mm512_ternarylogic_epi64(AFF(q0, m4), AFF(a0, m2), b0, XOR3);
			q1 = _mm512_ternarylogic_epi64(AFF(q1, m4), AFF(a1, m2), b1, XOR3);
		}
		if (z == start) {
			__m512i a0 = LD(d[z] + o), a1 = LD(d[z] + o + 64);
			p0 = _mm512_xor_si512(p0, a0);
			p1 = _mm512_xor_si512(p1, a1);
			q0 = _mm512_xor_si512(AFF(q0, m2), a0);
			q1 = _mm512_xor_si512(AFF(q1, m2), a1);
		}
		if (start) {
			q0 = AFF(q0, ms);
			q1 = AFF(q1, ms);
		}
		_mm512_store_si512((void *)(q + o), _mm512_xor_si512(q0, LD(q + o)));
		_mm512_store_si512((void *)(q + o + 64), _mm512_xor_si512(q1, LD(q + o + 64)));
		_mm512_store_si512((void *)(p + o), p0);
		_mm512_store_si512((void *)(p + o + 64), p1);
	}
}

/* Same procedure as recov_avx512.c, only the constant multiplication is one affine. */
__attribute__((noinline)) void gfni_recov_2data(int disks, size_t bytes,
						       int faila, int failb,
						       void **ptrs)
{
	u8 *p = ptrs[disks - 2], *q = ptrs[disks - 1];
	u8 *dp = ptrs[faila], *dq = ptrs[failb];

	ptrs[faila] = zero_page_buf;
	ptrs[disks - 2] = dp;
	ptrs[failb] = zero_page_buf;
	ptrs[disks - 1] = dq;
	raid6_gen_syndrome(disks, bytes, ptrs);
	ptrs[faila] = dp;
	ptrs[failb] = dq;
	ptrs[disks - 2] = p;
	ptrs[disks - 1] = q;

	const __m512i mpb = _mm512_set1_epi64(GFMAT(raid6_gfexi[failb - faila]));
	const __m512i mq = _mm512_set1_epi64(GFMAT(
		raid6_gfinv[raid6_gfexp[faila] ^ raid6_gfexp[failb]]));

	for (size_t o = 0; o < bytes; o += 64) {
		__m512i px = _mm512_xor_si512(LD(p + o), LD(dp + o));
		__m512i qx = _mm512_xor_si512(LD(q + o), LD(dq + o));
		__m512i db = _mm512_xor_si512(AFF(px, mpb), AFF(qx, mq));
		_mm512_store_si512((void *)(dq + o), db);
		_mm512_store_si512((void *)(dp + o), _mm512_xor_si512(db, px));
	}
}

__attribute__((noinline)) void gfni_recov_datap(int disks, size_t bytes,
						       int faila, void **ptrs)
{
	u8 *p = ptrs[disks - 2], *q = ptrs[disks - 1];
	u8 *dq = ptrs[faila];

	ptrs[faila] = zero_page_buf;
	ptrs[disks - 1] = dq;
	raid6_gen_syndrome(disks, bytes, ptrs);
	ptrs[faila] = dq;
	ptrs[disks - 1] = q;

	const __m512i mq = _mm512_set1_epi64(GFMAT(raid6_gfinv[raid6_gfexp[faila]]));

	for (size_t o = 0; o < bytes; o += 64) {
		__m512i x = AFF(_mm512_xor_si512(LD(q + o), LD(dq + o)), mq);
		_mm512_store_si512((void *)(dq + o), x);
		_mm512_store_si512((void *)(p + o), _mm512_xor_si512(x, LD(p + o)));
	}
}

/*
 * Single-pass recovery of two data disks: the syndrome without the failed disks is
 * computed in registers and Da, Db are finished right away. The failed disks are
 * neither read nor written as an intermediate (the kernel uses them as temporary dp/dq).
 */
__attribute__((noinline)) void gfni_recov_2data_fused(int disks, size_t bytes,
							     int faila, int failb,
							     void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[disks - 2], *q = d[disks - 1];
	u8 *da = d[faila], *db = d[failb];
	const u8 *src[MAX_DISKS];
	const __m512i m2 = _mm512_set1_epi64(GFMAT(2));
	const __m512i m4 = _mm512_set1_epi64(GFMAT(4));
	const __m512i mpb = _mm512_set1_epi64(GFMAT(raid6_gfexi[failb - faila]));
	const __m512i mq = _mm512_set1_epi64(GFMAT(
		raid6_gfinv[raid6_gfexp[faila] ^ raid6_gfexp[failb]]));

	for (int z = 0; z <= z0; z++)
		src[z] = (z == faila || z == failb) ? zero_page_buf : d[z];

	for (size_t o = 0; o < bytes; o += 128) {
		__m512i p0 = _mm512_xor_si512(LD(p + o), LD(src[z0] + o));
		__m512i p1 = _mm512_xor_si512(LD(p + o + 64), LD(src[z0] + o + 64));
		__m512i q0 = LD(src[z0] + o), q1 = LD(src[z0] + o + 64);
		int z = z0 - 1;
		for (; z >= 1; z -= 2) {
			__m512i a0 = LD(src[z] + o), a1 = LD(src[z] + o + 64);
			__m512i b0 = LD(src[z - 1] + o), b1 = LD(src[z - 1] + o + 64);
			p0 = _mm512_ternarylogic_epi64(p0, a0, b0, XOR3);
			p1 = _mm512_ternarylogic_epi64(p1, a1, b1, XOR3);
			q0 = _mm512_ternarylogic_epi64(AFF(q0, m4), AFF(a0, m2), b0, XOR3);
			q1 = _mm512_ternarylogic_epi64(AFF(q1, m4), AFF(a1, m2), b1, XOR3);
		}
		if (z == 0) {
			__m512i a0 = LD(src[0] + o), a1 = LD(src[0] + o + 64);
			p0 = _mm512_xor_si512(p0, a0);
			p1 = _mm512_xor_si512(p1, a1);
			q0 = _mm512_xor_si512(AFF(q0, m2), a0);
			q1 = _mm512_xor_si512(AFF(q1, m2), a1);
		}
		/* p0 = px, q0 ^ Q = qx */
		q0 = _mm512_xor_si512(q0, LD(q + o));
		q1 = _mm512_xor_si512(q1, LD(q + o + 64));
		__m512i b0 = _mm512_xor_si512(AFF(p0, mpb), AFF(q0, mq));
		__m512i b1 = _mm512_xor_si512(AFF(p1, mpb), AFF(q1, mq));
		_mm512_store_si512((void *)(db + o), b0);
		_mm512_store_si512((void *)(db + o + 64), b1);
		_mm512_store_si512((void *)(da + o), _mm512_xor_si512(b0, p0));
		_mm512_store_si512((void *)(da + o + 64), _mm512_xor_si512(b1, p1));
	}
}

/* ---------- helpers for tests and bench ---------- */

/* EVEX vgf2p8affineqb zmm: all 256 matrices against scalar gfmul */
int zmm_affine_errors(void)
{
	int bad = 0;
	for (int c = 0; c < 256; c++) {
		__m512i m = _mm512_set1_epi64(GFMAT(c));
		for (int x0 = 0; x0 < 256; x0 += 64) {
			u8 in[64] __aligned(64), out[64] __aligned(64);
			for (int i = 0; i < 64; i++)
				in[i] = x0 + i;
			_mm512_store_si512(out, AFF(_mm512_load_si512(in), m));
			for (int i = 0; i < 64; i++)
				bad += out[i] != gfmul(c, in[i]);
		}
	}
	return bad;
}

static double zmm_now_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e9 + t.tv_nsec;
}

/* warm up the cores, including powering up the upper halves of zmm */
void zmm_warmup(double ns)
{
	double t0 = zmm_now_ns();
	__m512i x = _mm512_set1_epi8(1);
	while (zmm_now_ns() - t0 < ns)
		for (int i = 0; i < 10000; i++)
			__asm__ volatile("vpaddb %0,%0,%0" : "+v"(x));
	__asm__ volatile("vzeroupper");
}

/* latency and throughput of the key zmm instructions in core cycles */
void zmm_bench_insn(double tcyc)
{
	const long n = 20000000;
	double t0, tadd, tlat, tthr, tter, tshuf;
	__m512i x = _mm512_set1_epi8(3), m = _mm512_set1_epi64(GFMAT(2));

	t0 = zmm_now_ns();
	for (long i = 0; i < n; i++)
		__asm__ volatile("vpaddb %%zmm0,%%zmm0,%%zmm0\n\tvpaddb %%zmm0,%%zmm0,%%zmm0\n\t"
				 "vpaddb %%zmm0,%%zmm0,%%zmm0\n\tvpaddb %%zmm0,%%zmm0,%%zmm0" ::: "xmm0");
	tadd = (zmm_now_ns() - t0) / (4.0 * n);

	t0 = zmm_now_ns();
	for (long i = 0; i < n; i++)
		__asm__ volatile("vgf2p8affineqb $0,%1,%0,%0\n\tvgf2p8affineqb $0,%1,%0,%0\n\t"
				 "vgf2p8affineqb $0,%1,%0,%0\n\tvgf2p8affineqb $0,%1,%0,%0"
				 : "+v"(x) : "v"(m));
	tlat = (zmm_now_ns() - t0) / (4.0 * n);

	t0 = zmm_now_ns();
	for (long i = 0; i < n; i++)
		__asm__ volatile(
			"vgf2p8affineqb $0,%%zmm16,%%zmm1,%%zmm1\n\tvgf2p8affineqb $0,%%zmm16,%%zmm2,%%zmm2\n\t"
			"vgf2p8affineqb $0,%%zmm16,%%zmm3,%%zmm3\n\tvgf2p8affineqb $0,%%zmm16,%%zmm4,%%zmm4\n\t"
			"vgf2p8affineqb $0,%%zmm16,%%zmm5,%%zmm5\n\tvgf2p8affineqb $0,%%zmm16,%%zmm6,%%zmm6\n\t"
			"vgf2p8affineqb $0,%%zmm16,%%zmm7,%%zmm7\n\tvgf2p8affineqb $0,%%zmm16,%%zmm8,%%zmm8"
			::: "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8");
	tthr = (zmm_now_ns() - t0) / (8.0 * n);

	t0 = zmm_now_ns();
	for (long i = 0; i < n; i++)
		__asm__ volatile(
			"vpternlogq $0x96,%%zmm16,%%zmm17,%%zmm1\n\tvpternlogq $0x96,%%zmm16,%%zmm17,%%zmm2\n\t"
			"vpternlogq $0x96,%%zmm16,%%zmm17,%%zmm3\n\tvpternlogq $0x96,%%zmm16,%%zmm17,%%zmm4\n\t"
			"vpternlogq $0x96,%%zmm16,%%zmm17,%%zmm5\n\tvpternlogq $0x96,%%zmm16,%%zmm17,%%zmm6\n\t"
			"vpternlogq $0x96,%%zmm16,%%zmm17,%%zmm7\n\tvpternlogq $0x96,%%zmm16,%%zmm17,%%zmm8"
			::: "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8");
	tter = (zmm_now_ns() - t0) / (8.0 * n);

	t0 = zmm_now_ns();
	for (long i = 0; i < n; i++)
		__asm__ volatile(
			"vpshufb %%zmm16,%%zmm1,%%zmm1\n\tvpshufb %%zmm16,%%zmm2,%%zmm2\n\t"
			"vpshufb %%zmm16,%%zmm3,%%zmm3\n\tvpshufb %%zmm16,%%zmm4,%%zmm4\n\t"
			"vpshufb %%zmm16,%%zmm5,%%zmm5\n\tvpshufb %%zmm16,%%zmm6,%%zmm6\n\t"
			"vpshufb %%zmm16,%%zmm7,%%zmm7\n\tvpshufb %%zmm16,%%zmm8,%%zmm8"
			::: "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8");
	tshuf = (zmm_now_ns() - t0) / (8.0 * n);
	__asm__ volatile("vzeroupper");

	printf("vpaddb zmm latency               %.2f cycles\n", tadd / tcyc);
	printf("vgf2p8affineqb zmm latency       %.2f cycles\n", tlat / tcyc);
	printf("vgf2p8affineqb zmm throughput    %.2f cycles/insn (%.2f/cycle)\n", tthr / tcyc, tcyc / tthr);
	printf("vpternlogq zmm throughput        %.2f cycles/insn (%.2f/cycle)\n", tter / tcyc, tcyc / tter);
	printf("vpshufb zmm throughput           %.2f cycles/insn (%.2f/cycle)\n", tshuf / tcyc, tcyc / tshuf);
}
