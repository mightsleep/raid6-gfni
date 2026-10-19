/*
 * RAID6 P+Q over GF(2^8)/0x11d: harness without SIMD flags, reference, ports
 * of the kernel code (avx2.c, recov_avx2.c, avx512.c, recov_avx512.c compiled
 * unchanged through kshim.h) and GFNI versions (zmm in raid6_zmm.c, ymm/VEX in
 * raid6_ymm.c). Implementations are picked at run time; those the CPU lacks are skipped.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "raid6.h"

u8 raid6_gfmul[256][256] __aligned(256);
u8 raid6_vgfmul[256][32] __aligned(256);
u8 raid6_gfexp[256], raid6_gflog[256], raid6_gfinv[256], raid6_gfexi[256];
u64 raid6_gfmat[256];

u8 zero_page_buf[MAX_BYTES] __aligned(4096);

extern const struct raid6_calls raid6_avx2x1, raid6_avx2x2, raid6_avx2x4;
extern const struct raid6_calls raid6_avx512x1, raid6_avx512x2, raid6_avx512x4;
extern const struct raid6_recov_calls raid6_recov_avx2, raid6_recov_avx512;

/* ---------- tables (same as mktables.c) ---------- */

u8 gfmul(u8 a, u8 b)
{
	u8 v = 0;
	while (b) {
		if (b & 1)
			v ^= a;
		a = (a << 1) ^ (a & 0x80 ? 0x1d : 0);
		b >>= 1;
	}
	return v;
}

static u8 gfpow(u8 a, int b)
{
	u8 v = 1;
	b %= 255;
	if (b < 0)
		b += 255;
	while (b) {
		if (b & 1)
			v = gfmul(v, a);
		a = gfmul(a, a);
		b >>= 1;
	}
	return v;
}

static void init_tables(void)
{
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++)
			raid6_gfmul[i][j] = gfmul(i, j);
		for (int j = 0; j < 16; j++) {
			raid6_vgfmul[i][j] = gfmul(i, j);
			raid6_vgfmul[i][16 + j] = gfmul(i, j << 4);
		}
	}
	u8 v = 1;
	for (int i = 0; i < 256; i++) {
		raid6_gfexp[i] = v;
		v = gfmul(v, 2);
		if (v == 1)
			v = 0;
	}
	for (int i = 0; i < 256; i++) {
		raid6_gflog[i] = 255;
		for (int k = 0; k < 256; k++)
			if (raid6_gfexp[k] == i) {
				raid6_gflog[i] = k;
				break;
			}
		raid6_gfinv[i] = gfpow(i, 254);
	}
	for (int i = 0; i < 256; i++)
		raid6_gfexi[i] = raid6_gfinv[raid6_gfexp[i] ^ 1];
}

/*
 * Matrices for vgf2p8affineqb: output bit i = parity(A.byte[7-i] & x).
 * For multiplication by a constant c, row i is the mask of k for which bit i of c*x^k is 1.
 */
static u64 gf_matrix(u8 c)
{
	u64 m = 0;
	for (int i = 0; i < 8; i++) {
		u8 row = 0;
		for (int k = 0; k < 8; k++)
			if (gfmul(c, 1 << k) >> i & 1)
				row |= 1 << k;
		m |= (u64)row << (8 * (7 - i));
	}
	return m;
}

/* ---------- (a) scalar reference after int.uc / recov.c ---------- */

static void ref_gen_syndrome(int disks, size_t bytes, void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	int z0 = disks - 3;
	u8 *p = d[z0 + 1], *q = d[z0 + 2];

	for (size_t i = 0; i < bytes; i++) {
		u8 wp = d[z0][i], wq = wp;
		for (int z = z0 - 1; z >= 0; z--) {
			wp ^= d[z][i];
			wq = gfmul(wq, 2) ^ d[z][i];
		}
		p[i] = wp;
		q[i] = wq;
	}
}

static void ref_xor_syndrome(int disks, int start, int stop, size_t bytes,
			     void **ptrs)
{
	u8 **d = (u8 **)ptrs;
	u8 *p = d[disks - 2], *q = d[disks - 1];

	for (size_t i = 0; i < bytes; i++) {
		u8 wp = d[stop][i], wq = wp;
		for (int z = stop - 1; z >= start; z--) {
			wp ^= d[z][i];
			wq = gfmul(wq, 2) ^ d[z][i];
		}
		for (int z = start - 1; z >= 0; z--)
			wq = gfmul(wq, 2);
		p[i] ^= wp;
		q[i] ^= wq;
	}
}

static void ref_recov_2data(int disks, size_t bytes, int faila, int failb,
			    void **ptrs)
{
	u8 *p = ptrs[disks - 2], *q = ptrs[disks - 1];
	u8 *dp = ptrs[faila], *dq = ptrs[failb];

	ptrs[faila] = zero_page_buf;
	ptrs[disks - 2] = dp;
	ptrs[failb] = zero_page_buf;
	ptrs[disks - 1] = dq;
	ref_gen_syndrome(disks, bytes, ptrs);
	ptrs[faila] = dp;
	ptrs[failb] = dq;
	ptrs[disks - 2] = p;
	ptrs[disks - 1] = q;

	const u8 *pbmul = raid6_gfmul[raid6_gfexi[failb - faila]];
	const u8 *qmul = raid6_gfmul[raid6_gfinv[raid6_gfexp[faila] ^
						 raid6_gfexp[failb]]];
	for (size_t i = 0; i < bytes; i++) {
		u8 px = p[i] ^ dp[i];
		u8 qx = qmul[q[i] ^ dq[i]];
		u8 db = pbmul[px] ^ qx;
		dq[i] = db;
		dp[i] = db ^ px;
	}
}

static void ref_recov_datap(int disks, size_t bytes, int faila, void **ptrs)
{
	u8 *p = ptrs[disks - 2], *q = ptrs[disks - 1];
	u8 *dq = ptrs[faila];

	ptrs[faila] = zero_page_buf;
	ptrs[disks - 1] = dq;
	ref_gen_syndrome(disks, bytes, ptrs);
	ptrs[faila] = dq;
	ptrs[disks - 1] = q;

	const u8 *qmul = raid6_gfmul[raid6_gfinv[raid6_gfexp[faila]]];
	for (size_t i = 0; i < bytes; i++) {
		dq[i] = qmul[q[i] ^ dq[i]];
		p[i] ^= dq[i];
	}
}


/* ---------- selection by CPU ---------- */

enum { F_AVX2 = 1, F_AVX512 = 2, F_GFNI = 4 };
static unsigned cpu_feat;

static void detect_cpu(unsigned mask_off)
{
	__builtin_cpu_init();
	if (__builtin_cpu_supports("avx2"))
		cpu_feat |= F_AVX2;
	/* kernel avx512.c needs F+BW, our zmm code also VL and DQ */
	if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
	    __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq"))
		cpu_feat |= F_AVX512;
	/* VEX gf2p8affineqb also needs AVX; without AVX2 we do not use it */
	if (__builtin_cpu_supports("gfni"))
		cpu_feat |= F_GFNI;
	cpu_feat &= ~mask_off;
}

static int have(unsigned need) { return (cpu_feat & need) == need; }

static const char *need_str(unsigned need)
{
	static char buf[64];
	buf[0] = 0;
	if (need & F_AVX2)
		strcat(buf, " avx2");
	if (need & F_AVX512)
		strcat(buf, " avx512");
	if (need & F_GFNI)
		strcat(buf, " gfni");
	return buf;
}

static char cpu_model[128] = "?";
static char cpu_flags[128];

static void cpu_describe(void)
{
	FILE *f = fopen("/proc/cpuinfo", "r");
	char line[4096];
	if (f) {
		while (fgets(line, sizeof(line), f))
			if (!strncmp(line, "model name", 10)) {
				char *s = strchr(line, ':');
				if (s) {
					s += 1 + (s[1] == ' ');
					s[strcspn(s, "\n")] = 0;
					snprintf(cpu_model, sizeof(cpu_model), "%s", s);
				}
				break;
			}
		fclose(f);
	}
	static const char *fl[] = { "avx2", "avx512f", "avx512bw", "avx512vl", "gfni",
				    "vaes", "vpclmulqdq" };
	for (unsigned i = 0; i < sizeof(fl) / sizeof(fl[0]); i++) {
		int on;
		/* __builtin_cpu_supports wants a string literal */
		switch (i) {
		case 0: on = __builtin_cpu_supports("avx2"); break;
		case 1: on = __builtin_cpu_supports("avx512f"); break;
		case 2: on = __builtin_cpu_supports("avx512bw"); break;
		case 3: on = __builtin_cpu_supports("avx512vl"); break;
		case 4: on = __builtin_cpu_supports("gfni"); break;
		case 5: on = __builtin_cpu_supports("vaes"); break;
		default: on = __builtin_cpu_supports("vpclmulqdq"); break;
		}
		if (on) {
			strcat(cpu_flags, cpu_flags[0] ? " " : "");
			strcat(cpu_flags, fl[i]);
		}
	}
}

/* ---------- interface for tests and bench ---------- */

typedef void (*gen_fn)(int, size_t, void **);
typedef void (*xor_fn)(int, int, int, size_t, void **);
typedef void (*rec2_fn)(int, size_t, int, int, void **);
typedef void (*recp_fn)(int, size_t, int, void **);

/* gen used inside recov (a static_call in the kernel) */
static gen_fn cur_gen = ref_gen_syndrome;
void raid6_gen_syndrome(int disks, size_t bytes, void **ptrs)
{
	cur_gen(disks, bytes, ptrs);
}

struct gen_impl {
	const char *name;
	unsigned need;
	gen_fn gen;
	xor_fn xor;
	const struct raid6_calls *k; /* kernel implementation; gen and xor are filled in at run time */
};

static struct gen_impl gens[] = {
	{ "ref", 0, ref_gen_syndrome, ref_xor_syndrome },
	{ "avx2x1", F_AVX2, .k = &raid6_avx2x1 },
	{ "avx2x2", F_AVX2, .k = &raid6_avx2x2 },
	{ "avx2x4", F_AVX2, .k = &raid6_avx2x4 },
	{ "avx512x1", F_AVX512, .k = &raid6_avx512x1 },
	{ "avx512x2", F_AVX512, .k = &raid6_avx512x2 },
	{ "avx512x4", F_AVX512, .k = &raid6_avx512x4 },
	{ "bw-tuned", F_AVX512, bw_gen_syndrome, NULL },
	{ "gfni", F_AVX512 | F_GFNI, gfni_gen_syndrome, gfni_xor_syndrome },
	{ "gfni-direct", F_AVX512 | F_GFNI, gfni_direct_gen_syndrome, NULL },
	{ "gfni-h1 (ISA-L style)", F_AVX512 | F_GFNI, gfni_h1_gen_syndrome, NULL },
	{ "gfni256x2 (ymm)", F_AVX2 | F_GFNI, gfni256x2_gen_syndrome, gfni256x2_xor_syndrome },
	{ "gfni256x4 (ymm)", F_AVX2 | F_GFNI, gfni256x4_gen_syndrome, gfni256x4_xor_syndrome },
};
#define NGENS (sizeof(gens) / sizeof(gens[0]))

struct rec_impl {
	const char *name;
	unsigned need;
	gen_fn gen; /* gen under recov */
	rec2_fn r2;
	recp_fn rp;
	const struct raid6_calls *kg;
	const struct raid6_recov_calls *kr;
};

static struct rec_impl recs[] = {
	{ "ref", 0, ref_gen_syndrome, ref_recov_2data, ref_recov_datap },
	{ "kernel-avx2 (gen avx2x2)", F_AVX2, .kg = &raid6_avx2x2, .kr = &raid6_recov_avx2 },
	{ "kernel-avx2 (gen avx2x4)", F_AVX2, .kg = &raid6_avx2x4, .kr = &raid6_recov_avx2 },
	{ "kernel-avx512 (gen avx512x2)", F_AVX512, .kg = &raid6_avx512x2, .kr = &raid6_recov_avx512 },
	{ "kernel-avx512 (gen avx512x4)", F_AVX512, .kg = &raid6_avx512x4, .kr = &raid6_recov_avx512 },
	{ "gfni", F_AVX512 | F_GFNI, gfni_gen_syndrome, gfni_recov_2data, gfni_recov_datap },
	{ "gfni-fused", F_AVX512 | F_GFNI, gfni_gen_syndrome, gfni_recov_2data_fused, gfni_recov_datap },
	{ "gfni (gen without NT)", F_AVX512 | F_GFNI, gfni_gen_syndrome_t, gfni_recov_2data, gfni_recov_datap },
	{ "kernel-avx512 (gen gfni without NT)", F_AVX512 | F_GFNI, gfni_gen_syndrome_t, .kr = &raid6_recov_avx512 },
	/* both fused, the gen below them is not called */
	{ "gfni256-fused (ymm)", F_AVX2 | F_GFNI, gfni256x2_gen_syndrome, gfni256_recov_2data, gfni256_recov_datap },
};
#define NRECS (sizeof(recs) / sizeof(recs[0]))

static void bind_kernel(void)
{
	for (unsigned g = 0; g < NGENS; g++)
		if (gens[g].k) {
			gens[g].gen = gens[g].k->gen_syndrome;
			gens[g].xor = gens[g].k->xor_syndrome;
		}
	for (unsigned r = 0; r < NRECS; r++) {
		if (recs[r].kg)
			recs[r].gen = recs[r].kg->gen_syndrome;
		if (recs[r].kr) {
			recs[r].r2 = recs[r].kr->data2;
			recs[r].rp = recs[r].kr->datap;
		}
	}
}

/* the kernel avx2/avx512 code has no vzeroupper (the patch is not in master yet) */
static inline void clean_upper(void)
{
	if (cpu_feat & (F_AVX2 | F_AVX512))
		__asm__ volatile("vzeroupper");
}

static void report_skipped(void)
{
	for (unsigned g = 0; g < NGENS; g++)
		if (!have(gens[g].need))
			printf("skipped (CPU lacks%s): gen %s\n", need_str(gens[g].need), gens[g].name);
	for (unsigned r = 0; r < NRECS; r++)
		if (!have(recs[r].need))
			printf("skipped (CPU lacks%s): recov %s\n", need_str(recs[r].need), recs[r].name);
}

/* ---------- tests ---------- */

static u8 *pool;
static u64 rng = 0x9e3779b97f4a7c15ULL;
static u64 rnd(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return rng;
}
static void fill(u8 *b, size_t n)
{
	for (size_t i = 0; i < n; i += 8) {
		u64 r = rnd();
		memcpy(b + i, &r, 8);
	}
}

static u8 *disk(int i) { return pool + (size_t)i * MAX_BYTES; }
/* second set of buffers for P/Q from the reference */
static u8 *refbuf(int i) { return pool + (size_t)(MAX_DISKS + i) * MAX_BYTES; }
static u8 *save(int i) { return pool + (size_t)(MAX_DISKS + 2 + i) * MAX_BYTES; }

static long failures, checks;
static void expect(int ok, const char *what, const char *impl, int disks, int a, int b, size_t bytes)
{
	checks++;
	if (!ok) {
		failures++;
		if (failures < 20)
			fprintf(stderr, "FAIL %s %s disks=%d a=%d b=%d bytes=%zu\n",
				what, impl, disks, a, b, bytes);
	}
}

static void test_gf_basics(void)
{
	if (!have(F_AVX2 | F_GFNI)) {
		printf("GFNI with VEX not available, matrix test skipped\n");
		return;
	}
	/* gf2p8mulb uses 0x11b (AES), RAID6 0x11d: unusable */
	u8 r = xmm_gf2p8mulb(0x80, 2);
	printf("gf2p8mulb(0x80, 2) = 0x%02x (0x11d would give 0x1d, 0x11b gives 0x1b)\n", r);
	expect(r == 0x1b, "gf2p8mulb is 0x11b", "hw", 0, 0, 0, 0);

	/* affine matrices: full check of all 256x256 products, VEX ymm and EVEX zmm */
	int bad = ymm_affine_errors();
	clean_upper();
	printf("affine matrix ymm (VEX): %d errors in 65536 products\n", bad);
	expect(bad == 0, "affine matrix ymm", "hw", 0, 0, 0, 0);
	if (have(F_AVX512 | F_GFNI)) {
		bad = zmm_affine_errors();
		clean_upper();
		printf("affine matrix zmm (EVEX): %d errors in 65536 products\n", bad);
		expect(bad == 0, "affine matrix zmm", "hw", 0, 0, 0, 0);
	}
	printf("matrix for multiplication by 2 (0x11d): 0x%016llx\n", (unsigned long long)GFMAT(2));
}

static void test_gen(void)
{
	static const size_t sizes[] = { 512, 4096, 65536 };
	void *ptrs[MAX_DISKS];
	for (int disks = 4; disks <= MAX_DISKS; disks++)
		for (int si = 0; si < 3; si++) {
			size_t bytes = sizes[si];
			for (int i = 0; i < disks - 2; i++)
				fill(disk(i), bytes);
			for (int i = 0; i < disks - 2; i++)
				ptrs[i] = disk(i);
			ptrs[disks - 2] = refbuf(0);
			ptrs[disks - 1] = refbuf(1);
			ref_gen_syndrome(disks, bytes, ptrs);
			for (unsigned g = 1; g < NGENS; g++) {
				if (!have(gens[g].need))
					continue;
				ptrs[disks - 2] = disk(disks - 2);
				ptrs[disks - 1] = disk(disks - 1);
				memset(disk(disks - 2), 0xa5, bytes);
				memset(disk(disks - 1), 0x5a, bytes);
				gens[g].gen(disks, bytes, ptrs);
				clean_upper();
				expect(!memcmp(disk(disks - 2), refbuf(0), bytes), "gen P", gens[g].name, disks, 0, 0, bytes);
				expect(!memcmp(disk(disks - 1), refbuf(1), bytes), "gen Q", gens[g].name, disks, 0, 0, bytes);
			}
		}
}

static void test_xor(void)
{
	void *ptrs[MAX_DISKS];
	const size_t bytes = 1024;
	for (int disks = 4; disks <= MAX_DISKS; disks++)
		for (int start = 0; start <= disks - 3; start++)
			for (int stop = start; stop <= disks - 3; stop++) {
				for (int i = 0; i < disks; i++) {
					fill(disk(i), bytes);
					ptrs[i] = disk(i);
				}
				memcpy(refbuf(0), disk(disks - 2), bytes);
				memcpy(refbuf(1), disk(disks - 1), bytes);
				memcpy(save(0), disk(disks - 2), bytes);
				memcpy(save(1), disk(disks - 1), bytes);
				ptrs[disks - 2] = refbuf(0);
				ptrs[disks - 1] = refbuf(1);
				ref_xor_syndrome(disks, start, stop, bytes, ptrs);
				ptrs[disks - 2] = disk(disks - 2);
				ptrs[disks - 1] = disk(disks - 1);
				for (unsigned g = 1; g < NGENS; g++) {
					if (!gens[g].xor || !have(gens[g].need))
						continue;
					memcpy(disk(disks - 2), save(0), bytes);
					memcpy(disk(disks - 1), save(1), bytes);
					gens[g].xor(disks, start, stop, bytes, ptrs);
					clean_upper();
					expect(!memcmp(disk(disks - 2), refbuf(0), bytes), "xor P", gens[g].name, disks, start, stop, bytes);
					expect(!memcmp(disk(disks - 1), refbuf(1), bytes), "xor Q", gens[g].name, disks, start, stop, bytes);
				}
			}
}

static void test_recov(void)
{
	void *ptrs[MAX_DISKS];
	const size_t bytes = 4096;
	for (int disks = 4; disks <= MAX_DISKS; disks++) {
		for (int i = 0; i < disks - 2; i++) {
			fill(disk(i), bytes);
			ptrs[i] = disk(i);
		}
		ptrs[disks - 2] = disk(disks - 2);
		ptrs[disks - 1] = disk(disks - 1);
		ref_gen_syndrome(disks, bytes, ptrs);
		for (int i = 0; i < disks; i++)
			memcpy(save(i), disk(i), bytes);

		for (unsigned r = 0; r < NRECS; r++) {
			if (!have(recs[r].need))
				continue;
			cur_gen = recs[r].gen;
			for (int a = 0; a < disks - 2; a++) {
				for (int b = a + 1; b < disks - 2; b++) {
					fill(disk(a), bytes);
					fill(disk(b), bytes);
					recs[r].r2(disks, bytes, a, b, ptrs);
					clean_upper();
					int ok = 1;
					for (int i = 0; i < disks; i++)
						ok &= !memcmp(disk(i), save(i), bytes);
					expect(ok, "recov 2data", recs[r].name, disks, a, b, bytes);
					for (int i = 0; i < disks; i++)
						memcpy(disk(i), save(i), bytes);
				}
				fill(disk(a), bytes);
				fill(disk(disks - 2), bytes);
				recs[r].rp(disks, bytes, a, ptrs);
				clean_upper();
				int ok = 1;
				for (int i = 0; i < disks; i++)
					ok &= !memcmp(disk(i), save(i), bytes);
				expect(ok, "recov datap", recs[r].name, disks, a, -1, bytes);
				for (int i = 0; i < disks; i++)
					memcpy(disk(i), save(i), bytes);
			}
		}
	}
}

/* ---------- bench (rough) ---------- */

static double now_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e9 + t.tv_nsec;
}

#define ROUNDS 5
#define BENCH_DATA 8
#define BENCH_DISKS (BENCH_DATA + 2)

static long reps_for(size_t bytes) { return 100L * 65536 / bytes; }

/* warm up the cores; with AVX-512 also the upper halves of zmm */
static void warmup(void)
{
	if (have(F_AVX512)) {
		zmm_warmup(300e6);
		return;
	}
	double t0 = now_ns();
	while (now_ns() - t0 < 300e6)
		for (int i = 0; i < 10000; i++) {
			if (have(F_AVX2))
				__asm__ volatile("vpaddb %%ymm0,%%ymm0,%%ymm0" ::: "xmm0");
			else
				__asm__ volatile("add $1,%%eax" ::: "eax");
		}
	clean_upper();
}

enum { K_GEN, K_XOR, K_REC2, K_RECP };

/* one measurement of one case, ns per call */
static double run_case(int kind, unsigned idx, size_t bytes, void **ptrs)
{
	const int disks = BENCH_DISKS;
	int start = disks / 2 - 1, stop = disks - 3; /* as in algos.c */
	long reps = reps_for(bytes);
	if (kind >= K_REC2)
		cur_gen = recs[idx].gen;
	double t = now_ns();
	for (long i = 0; i < reps; i++) {
		switch (kind) {
		case K_GEN: gens[idx].gen(disks, bytes, ptrs); break;
		case K_XOR: gens[idx].xor(disks, start, stop, bytes, ptrs); break;
		case K_REC2: recs[idx].r2(disks, bytes, 1, 5, ptrs); break;
		case K_RECP: recs[idx].rp(disks, bytes, 3, ptrs); break;
		}
	}
	t = (now_ns() - t) / reps;
	clean_upper();
	return t;
}

#define MAXIMPL 16

static void bench(double ns_per_cycle, FILE *csv)
{
	static const size_t sizes[] = { 4096, 16384, 65536 };
	static const char *kn[] = { "gen", "xor", "rec2", "recp" };
	void *ptrs[MAX_DISKS];
	const int disks = BENCH_DISKS;
	double best[4][MAXIMPL];

	printf("\n# bench %d+2 disks, ns per call (min of %d interleaved rounds), "
	       "GB/s = data bytes/ns, B/cycle from calibration\n", BENCH_DATA, ROUNDS);
	for (int i = 0; i < disks; i++) {
		fill(disk(i), MAX_BYTES);
		ptrs[i] = disk(i);
	}
	for (int si = 0; si < 3; si++) {
		size_t bytes = sizes[si];
		for (int k = 0; k < 4; k++)
			for (int j = 0; j < MAXIMPL; j++)
				best[k][j] = 1e30;
		for (int r = 0; r < ROUNDS; r++)
			for (int k = 0; k < 4; k++) {
				unsigned n = k < K_REC2 ? NGENS : NRECS;
				for (unsigned j = 1; j < n; j++) {
					if (k < K_REC2 ? !have(gens[j].need) : !have(recs[j].need))
						continue;
					if (k == K_XOR && !gens[j].xor)
						continue;
					double t = run_case(k, j, bytes, ptrs);
					if (t < best[k][j])
						best[k][j] = t;
				}
			}
		for (int k = 0; k < 4; k++) {
			unsigned n = k < K_REC2 ? NGENS : NRECS;
			/* the xor bench works on half of the disks only (start..stop) */
			int nd = k == K_XOR ? disks - 3 - (disks / 2 - 1) + 1 : BENCH_DATA;
			for (unsigned j = 1; j < n; j++) {
				if (best[k][j] > 1e29)
					continue;
				const char *name = k < K_REC2 ? gens[j].name : recs[j].name;
				double gbs = nd * bytes / best[k][j];
				double bpc = nd * bytes / (best[k][j] / ns_per_cycle);
				printf("%-5s %-32s %6zu B %9.1f ns %7.2f GB/s %6.1f B/cycle\n",
				       kn[k], name, bytes, best[k][j], gbs, bpc);
				if (csv)
					fprintf(csv, "\"%s\",\"%s\",%s,\"%s\",%zu,%.1f,%.3f,%.2f\n",
						cpu_model, cpu_flags, kn[k], name, bytes,
						best[k][j], gbs, bpc);
			}
		}
	}
}

/* clock from a chain of scalar add (latency 1 cycle everywhere) and ymm instructions */
static double bench_insn(void)
{
	const long n = 20000000;
	double t0;

	long acc = 0;
	t0 = now_ns();
	for (long i = 0; i < n; i++)
		__asm__ volatile("add $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\t"
				 "add $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0" : "+r"(acc));
	double tcyc = (now_ns() - t0) / (8.0 * n);
	printf("\n# instructions, cycles from a chain of scalar add (%.3f ns/cycle = %.2f GHz)\n", tcyc, 1 / tcyc);

	if (have(F_AVX2 | F_GFNI)) {
		/* only ymm0 to ymm9, so VEX; the matrix in ymm9 is not set, timing does not depend on it */
		t0 = now_ns();
		for (long i = 0; i < n; i++)
			__asm__ volatile("vgf2p8affineqb $0,%%ymm9,%%ymm0,%%ymm0\n\t"
					 "vgf2p8affineqb $0,%%ymm9,%%ymm0,%%ymm0\n\t"
					 "vgf2p8affineqb $0,%%ymm9,%%ymm0,%%ymm0\n\t"
					 "vgf2p8affineqb $0,%%ymm9,%%ymm0,%%ymm0" ::: "xmm0");
		double tlat = (now_ns() - t0) / (4.0 * n);
		t0 = now_ns();
		for (long i = 0; i < n; i++)
			__asm__ volatile(
				"vgf2p8affineqb $0,%%ymm9,%%ymm1,%%ymm1\n\tvgf2p8affineqb $0,%%ymm9,%%ymm2,%%ymm2\n\t"
				"vgf2p8affineqb $0,%%ymm9,%%ymm3,%%ymm3\n\tvgf2p8affineqb $0,%%ymm9,%%ymm4,%%ymm4\n\t"
				"vgf2p8affineqb $0,%%ymm9,%%ymm5,%%ymm5\n\tvgf2p8affineqb $0,%%ymm9,%%ymm6,%%ymm6\n\t"
				"vgf2p8affineqb $0,%%ymm9,%%ymm7,%%ymm7\n\tvgf2p8affineqb $0,%%ymm9,%%ymm8,%%ymm8"
				::: "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8");
		double tthr = (now_ns() - t0) / (8.0 * n);
		printf("vgf2p8affineqb ymm latency       %.2f cycles\n", tlat / tcyc);
		printf("vgf2p8affineqb ymm throughput    %.2f cycles/insn (%.2f/cycle)\n", tthr / tcyc, tcyc / tthr);
	}
	if (have(F_AVX2)) {
		t0 = now_ns();
		for (long i = 0; i < n; i++)
			__asm__ volatile(
				"vpxor %%ymm9,%%ymm1,%%ymm1\n\tvpxor %%ymm9,%%ymm2,%%ymm2\n\t"
				"vpxor %%ymm9,%%ymm3,%%ymm3\n\tvpxor %%ymm9,%%ymm4,%%ymm4\n\t"
				"vpxor %%ymm9,%%ymm5,%%ymm5\n\tvpxor %%ymm9,%%ymm6,%%ymm6\n\t"
				"vpxor %%ymm9,%%ymm7,%%ymm7\n\tvpxor %%ymm9,%%ymm8,%%ymm8"
				::: "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8");
		double txor = (now_ns() - t0) / (8.0 * n);
		printf("vpxor ymm throughput             %.2f cycles/insn (%.2f/cycle)\n", txor / tcyc, tcyc / txor);
		clean_upper();
	}
	if (have(F_AVX512 | F_GFNI))
		zmm_bench_insn(tcyc);
	return tcyc;
}

static void usage(const char *argv0)
{
	fprintf(stderr, "usage: %s [bench] [--csv FILE] [--no-avx512] [--no-gfni] [--no-avx2]\n"
		"  without 'bench' runs the tests, with 'bench' only the matrix test and the bench\n"
		"  --no-* disables implementations as on a CPU without that extension\n", argv0);
	exit(2);
}

int main(int argc, char **argv)
{
	int do_bench = 0;
	unsigned mask_off = 0;
	const char *csv_path = NULL;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "bench"))
			do_bench = 1;
		else if (!strcmp(argv[i], "--csv") && i + 1 < argc)
			csv_path = argv[++i];
		else if (!strcmp(argv[i], "--no-avx512"))
			mask_off |= F_AVX512;
		else if (!strcmp(argv[i], "--no-gfni"))
			mask_off |= F_GFNI;
		else if (!strcmp(argv[i], "--no-avx2"))
			mask_off |= F_AVX2;
		else
			usage(argv[0]);
	}

	detect_cpu(mask_off);
	cpu_describe();
	printf("# cpu: %s\n# flags: %s\n# used:%s%s\n", cpu_model, cpu_flags,
	       need_str(cpu_feat), mask_off ? " (some disabled on the command line)" : "");

	init_tables();
	for (int c = 0; c < 256; c++)
		raid6_gfmat[c] = gf_matrix(c);
	bind_kernel();
	pool = aligned_alloc(4096, (size_t)(2 * MAX_DISKS + 4) * MAX_BYTES);
	fill(pool, (size_t)(2 * MAX_DISKS + 4) * MAX_BYTES);

	report_skipped();
	test_gf_basics();
	if (!do_bench) {
		test_gen();
		test_xor();
		test_recov();
		printf("tests: %ld checks, %ld failures\n", checks, failures);
	} else {
		FILE *csv = NULL;
		if (csv_path) {
			csv = fopen(csv_path, "w");
			if (!csv) {
				perror(csv_path);
				return 2;
			}
			fprintf(csv, "cpu,flags,op,impl,bytes,ns,gbps,bytes_per_cycle\n");
		}
		warmup();
		double c = bench_insn();
		bench(c, csv);
		if (csv)
			fclose(csv);
	}
	return failures != 0;
}
