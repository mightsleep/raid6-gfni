/*
 * GFNI with ymm for CPUs with AVX2 + GFNI and no AVX-512 (Alder Lake and newer client
 * Intel, N100/N150). Compiled only with -mavx2 -mgfni, so the compiler has no way
 * to emit an EVEX instruction (checked with objdump, see NOTES.md).
 *
 * Without vpternlogq every XOR is a separate instruction. Horner over two disks:
 *   Q = 4*Q ^ (2*a ^ b),  P = P ^ (a ^ b)
 * so 2 affine + 4 vpxor per two disks and 32 B; the chain through Q is affine + 1 xor.
 */
#include <immintrin.h>
#include "raid6.h"

#define LD(ptr) _mm256_load_si256((const __m256i *)(ptr))
#define X(a, b) _mm256_xor_si256((a), (b))
#define AFF(x, m) _mm256_gf2p8affine_epi64_epi8((x), (m), 0)
#define BCAST(c) _mm256_set1_epi64x((long long)GFMAT(c))
#define ST(ptr, v) _mm256_store_si256((__m256i *)(ptr), (v))
#define STNT(ptr, v) _mm256_stream_si256((__m256i *)(ptr), (v))

/*
 * N ymm per iteration (32*N B). N is a constant, the loops over i fully unroll
 * after inlining and the arrays p[], q[] end up in registers (checked: no spill
 * in objdump). For N = 4: 8 accumulators + 2 matrices + temporaries = 16 ymm.
 */
static inline __attribute__((always_inline)) void gen_body(int disks, size_t bytes,
							   void **ptrs, const int N)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[z0 + 1], *q = d[z0 + 2];
	const __m256i m2 = BCAST(2), m4 = BCAST(4);

	for (size_t o = 0; o < bytes; o += 32 * N) {
		__m256i pv[4], qv[4];
#pragma GCC unroll 4
		for (int i = 0; i < N; i++)
			pv[i] = qv[i] = LD(d[z0] + o + 32 * i);
		int z = z0 - 1;
		for (; z >= 1; z -= 2) {
			const u8 *sa = d[z] + o, *sb = d[z - 1] + o;
#pragma GCC unroll 4
			for (int i = 0; i < N; i++) {
				__m256i a = LD(sa + 32 * i), b = LD(sb + 32 * i);
				pv[i] = X(pv[i], X(a, b));
				qv[i] = X(AFF(qv[i], m4), X(AFF(a, m2), b));
			}
		}
		if (z == 0) {
#pragma GCC unroll 4
			for (int i = 0; i < N; i++) {
				__m256i a = LD(d[0] + o + 32 * i);
				pv[i] = X(pv[i], a);
				qv[i] = X(AFF(qv[i], m2), a);
			}
		}
#pragma GCC unroll 4
		for (int i = 0; i < N; i++) {
			STNT(p + o + 32 * i, pv[i]);
			STNT(q + o + 32 * i, qv[i]);
		}
	}
	_mm_sfence();
}

/* as zmm: right side by Horner, left side (disks below start) one affine by 2^start */
static inline __attribute__((always_inline)) void xor_body(int disks, int start, int stop,
							   size_t bytes, void **ptrs,
							   const int N)
{
	u8 **d = (u8 **)ptrs;
	u8 *p = d[disks - 2], *q = d[disks - 1];
	const __m256i m2 = BCAST(2), m4 = BCAST(4);
	const __m256i ms = BCAST(raid6_gfexp[start]);

	for (size_t o = 0; o < bytes; o += 32 * N) {
		__m256i pv[4], qv[4];
#pragma GCC unroll 4
		for (int i = 0; i < N; i++) {
			qv[i] = LD(d[stop] + o + 32 * i);
			pv[i] = X(qv[i], LD(p + o + 32 * i));
		}
		int z = stop - 1;
		for (; z >= start + 1; z -= 2) {
			const u8 *sa = d[z] + o, *sb = d[z - 1] + o;
#pragma GCC unroll 4
			for (int i = 0; i < N; i++) {
				__m256i a = LD(sa + 32 * i), b = LD(sb + 32 * i);
				pv[i] = X(pv[i], X(a, b));
				qv[i] = X(AFF(qv[i], m4), X(AFF(a, m2), b));
			}
		}
		if (z == start) {
#pragma GCC unroll 4
			for (int i = 0; i < N; i++) {
				__m256i a = LD(d[z] + o + 32 * i);
				pv[i] = X(pv[i], a);
				qv[i] = X(AFF(qv[i], m2), a);
			}
		}
#pragma GCC unroll 4
		for (int i = 0; i < N; i++) {
			if (start)
				qv[i] = AFF(qv[i], ms);
			ST(q + o + 32 * i, X(qv[i], LD(q + o + 32 * i)));
			ST(p + o + 32 * i, pv[i]);
		}
	}
}

__attribute__((noinline)) void gfni256x2_gen_syndrome(int disks, size_t bytes, void **ptrs)
{
	gen_body(disks, bytes, ptrs, 2);
}

__attribute__((noinline)) void gfni256x4_gen_syndrome(int disks, size_t bytes, void **ptrs)
{
	gen_body(disks, bytes, ptrs, 4);
}

__attribute__((noinline)) void gfni256x2_xor_syndrome(int disks, int start, int stop,
						      size_t bytes, void **ptrs)
{
	xor_body(disks, start, stop, bytes, ptrs, 2);
}

__attribute__((noinline)) void gfni256x4_xor_syndrome(int disks, int start, int stop,
						      size_t bytes, void **ptrs)
{
	xor_body(disks, start, stop, bytes, ptrs, 4);
}

/*
 * Single-pass recovery of two data disks (like gfni_recov_2data_fused):
 * syndrome without the failed disks in registers, then
 *   Db = A*(P ^ P') ^ B*(Q ^ Q'),  Da = Db ^ (P ^ P').
 * 64 B per iteration: 4 accumulators + 4 matrices.
 */
__attribute__((noinline)) void gfni256_recov_2data(int disks, size_t bytes, int faila,
						   int failb, void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[disks - 2], *q = d[disks - 1];
	u8 *da = d[faila], *db = d[failb];
	const u8 *src[MAX_DISKS];
	const __m256i m2 = BCAST(2), m4 = BCAST(4);
	const __m256i mpb = BCAST(raid6_gfexi[failb - faila]);
	const __m256i mq = BCAST(raid6_gfinv[raid6_gfexp[faila] ^ raid6_gfexp[failb]]);

	for (int z = 0; z <= z0; z++)
		src[z] = (z == faila || z == failb) ? zero_page_buf : d[z];

	for (size_t o = 0; o < bytes; o += 64) {
		__m256i q0 = LD(src[z0] + o), q1 = LD(src[z0] + o + 32);
		__m256i p0 = X(LD(p + o), q0), p1 = X(LD(p + o + 32), q1);
		int z = z0 - 1;
		for (; z >= 1; z -= 2) {
			__m256i a0 = LD(src[z] + o), a1 = LD(src[z] + o + 32);
			__m256i b0 = LD(src[z - 1] + o), b1 = LD(src[z - 1] + o + 32);
			p0 = X(p0, X(a0, b0));
			p1 = X(p1, X(a1, b1));
			q0 = X(AFF(q0, m4), X(AFF(a0, m2), b0));
			q1 = X(AFF(q1, m4), X(AFF(a1, m2), b1));
		}
		if (z == 0) {
			__m256i a0 = LD(src[0] + o), a1 = LD(src[0] + o + 32);
			p0 = X(p0, a0);
			p1 = X(p1, a1);
			q0 = X(AFF(q0, m2), a0);
			q1 = X(AFF(q1, m2), a1);
		}
		q0 = X(q0, LD(q + o));
		q1 = X(q1, LD(q + o + 32));
		__m256i b0 = X(AFF(p0, mpb), AFF(q0, mq));
		__m256i b1 = X(AFF(p1, mpb), AFF(q1, mq));
		ST(db + o, b0);
		ST(db + o + 32, b1);
		ST(da + o, X(b0, p0));
		ST(da + o + 32, X(b1, p1));
	}
}

/*
 * Single-pass recovery of a data disk and P: P' and Q' from the surviving disks,
 * Da = g^-a * (Q ^ Q'), P = P' ^ Da.
 */
__attribute__((noinline)) void gfni256_recov_datap(int disks, size_t bytes, int faila,
						   void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[disks - 2], *q = d[disks - 1];
	u8 *da = d[faila];
	const u8 *src[MAX_DISKS];
	const __m256i m2 = BCAST(2), m4 = BCAST(4);
	const __m256i mq = BCAST(raid6_gfinv[raid6_gfexp[faila]]);

	for (int z = 0; z <= z0; z++)
		src[z] = z == faila ? zero_page_buf : d[z];

	for (size_t o = 0; o < bytes; o += 64) {
		__m256i p0 = LD(src[z0] + o), p1 = LD(src[z0] + o + 32);
		__m256i q0 = p0, q1 = p1;
		int z = z0 - 1;
		for (; z >= 1; z -= 2) {
			__m256i a0 = LD(src[z] + o), a1 = LD(src[z] + o + 32);
			__m256i b0 = LD(src[z - 1] + o), b1 = LD(src[z - 1] + o + 32);
			p0 = X(p0, X(a0, b0));
			p1 = X(p1, X(a1, b1));
			q0 = X(AFF(q0, m4), X(AFF(a0, m2), b0));
			q1 = X(AFF(q1, m4), X(AFF(a1, m2), b1));
		}
		if (z == 0) {
			__m256i a0 = LD(src[0] + o), a1 = LD(src[0] + o + 32);
			p0 = X(p0, a0);
			p1 = X(p1, a1);
			q0 = X(AFF(q0, m2), a0);
			q1 = X(AFF(q1, m2), a1);
		}
		__m256i x0 = AFF(X(q0, LD(q + o)), mq);
		__m256i x1 = AFF(X(q1, LD(q + o + 32)), mq);
		ST(da + o, x0);
		ST(da + o + 32, x1);
		ST(p + o, X(p0, x0));
		ST(p + o + 32, X(p1, x1));
	}
}

/* VEX vgf2p8affineqb ymm: all 256 matrices against scalar gfmul */
int ymm_affine_errors(void)
{
	int bad = 0;
	for (int c = 0; c < 256; c++) {
		__m256i m = BCAST(c);
		for (int x0 = 0; x0 < 256; x0 += 32) {
			u8 in[32] __aligned(32), out[32] __aligned(32);
			for (int i = 0; i < 32; i++)
				in[i] = x0 + i;
			ST(out, AFF(LD(in), m));
			for (int i = 0; i < 32; i++)
				bad += out[i] != gfmul(c, in[i]);
		}
	}
	return bad;
}

/* VEX vgf2p8mulb xmm */
u8 xmm_gf2p8mulb(u8 a, u8 b)
{
	__m128i r = _mm_gf2p8mul_epi8(_mm_set1_epi8((char)a), _mm_set1_epi8((char)b));
	return (u8)_mm_extract_epi8(r, 0);
}
