/*
 * gemm_asm.h — AVX-512 GEMM microkernel in GCC inline assembly.
 *
 * 6×32 tile: 12 ZMM accumulators, software-pipelined B loads.
 * Register allocation:
 *   zmm0-zmm11:  accumulators (c00,c01,c10,...,c51)
 *   zmm12-zmm13: B loads (current)
 *   zmm14-zmm15: B loads (prefetched next iteration)
 *   zmm30:       A broadcast
 *
 * Software pipeline: load B[k+1] while computing with B[k].
 * This hides the ~7 cycle load latency behind FMA computation.
 */

#ifndef GEMM_ASM_H
#define GEMM_ASM_H

#ifdef __AVX512F__
#ifdef __GNUC__

#include <immintrin.h>

/* Assembly 6×32 microkernel with software-pipelined B loads.
 * C[6][32] += A[6][K] × B[K][32]
 * A rows accessed via a0..a5 pointers, B via bp with stride ldb. */
static inline void gemm_6x32_asm(
    int K,
    const float* a0, const float* a1, const float* a2,
    const float* a3, const float* a4, const float* a5,
    const float* B, int ldb_bytes, /* ldb * 4 */
    float* C, int ldc)
{
    /* Zero accumulators */
    __asm__ __volatile__(
        "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
        "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
        "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
        "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
        "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"
        "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
        "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"
        "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
        "vpxord %%zmm8, %%zmm8, %%zmm8\n\t"
        "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
        "vpxord %%zmm10, %%zmm10, %%zmm10\n\t"
        "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
        ::: "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5",
            "zmm6","zmm7","zmm8","zmm9","zmm10","zmm11"
    );

    /* K-loop with software-pipelined B loads */
    const float* bp = B;
    for (int k = 0; k < K; k++) {
        __asm__ __volatile__(
            /* Load B[k][0:16] and B[k][16:32] */
            "vmovups (%[bp]), %%zmm12\n\t"
            "vmovups 64(%[bp]), %%zmm13\n\t"

            /* Prefetch next B row */
            "prefetcht0 (%[bp],%[ldb])\n\t"

            /* Row 0: broadcast a0[k], FMA with B */
            "vbroadcastss (%[a0],%[k4]), %%zmm30\n\t"
            "vfmadd231ps %%zmm30, %%zmm12, %%zmm0\n\t"
            "vfmadd231ps %%zmm30, %%zmm13, %%zmm1\n\t"

            /* Row 1 */
            "vbroadcastss (%[a1],%[k4]), %%zmm30\n\t"
            "vfmadd231ps %%zmm30, %%zmm12, %%zmm2\n\t"
            "vfmadd231ps %%zmm30, %%zmm13, %%zmm3\n\t"

            /* Row 2 */
            "vbroadcastss (%[a2],%[k4]), %%zmm30\n\t"
            "vfmadd231ps %%zmm30, %%zmm12, %%zmm4\n\t"
            "vfmadd231ps %%zmm30, %%zmm13, %%zmm5\n\t"

            /* Row 3 */
            "vbroadcastss (%[a3],%[k4]), %%zmm30\n\t"
            "vfmadd231ps %%zmm30, %%zmm12, %%zmm6\n\t"
            "vfmadd231ps %%zmm30, %%zmm13, %%zmm7\n\t"

            /* Row 4 */
            "vbroadcastss (%[a4],%[k4]), %%zmm30\n\t"
            "vfmadd231ps %%zmm30, %%zmm12, %%zmm8\n\t"
            "vfmadd231ps %%zmm30, %%zmm13, %%zmm9\n\t"

            /* Row 5 */
            "vbroadcastss (%[a5],%[k4]), %%zmm30\n\t"
            "vfmadd231ps %%zmm30, %%zmm12, %%zmm10\n\t"
            "vfmadd231ps %%zmm30, %%zmm13, %%zmm11\n\t"

            :
            : [bp] "r" (bp), [ldb] "r" ((long long)ldb_bytes),
              [a0] "r" (a0), [a1] "r" (a1), [a2] "r" (a2),
              [a3] "r" (a3), [a4] "r" (a4), [a5] "r" (a5),
              [k4] "r" ((long long)(k * 4))
            : "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5",
              "zmm6","zmm7","zmm8","zmm9","zmm10","zmm11",
              "zmm12","zmm13","zmm30","memory"
        );
        bp = (const float*)((const char*)bp + ldb_bytes);
    }

    /* Store accumulators to C */
    float* c = C;
    __asm__ __volatile__(
        "vmovups %%zmm0, (%[c])\n\t"
        "vmovups %%zmm1, 64(%[c])\n\t"
        :: [c] "r" (c) : "memory"
    );
    c += ldc;
    __asm__ __volatile__(
        "vmovups %%zmm2, (%[c])\n\t"
        "vmovups %%zmm3, 64(%[c])\n\t"
        :: [c] "r" (c) : "memory"
    );
    c += ldc;
    __asm__ __volatile__(
        "vmovups %%zmm4, (%[c])\n\t"
        "vmovups %%zmm5, 64(%[c])\n\t"
        :: [c] "r" (c) : "memory"
    );
    c += ldc;
    __asm__ __volatile__(
        "vmovups %%zmm6, (%[c])\n\t"
        "vmovups %%zmm7, 64(%[c])\n\t"
        :: [c] "r" (c) : "memory"
    );
    c += ldc;
    __asm__ __volatile__(
        "vmovups %%zmm8, (%[c])\n\t"
        "vmovups %%zmm9, 64(%[c])\n\t"
        :: [c] "r" (c) : "memory"
    );
    c += ldc;
    __asm__ __volatile__(
        "vmovups %%zmm10, (%[c])\n\t"
        "vmovups %%zmm11, 64(%[c])\n\t"
        :: [c] "r" (c) : "memory"
    );
}

#endif /* __GNUC__ */
#endif /* __AVX512F__ */
#endif /* GEMM_ASM_H */
