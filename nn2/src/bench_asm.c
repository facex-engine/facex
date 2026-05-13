/*
 * bench_asm.c — Compare assembly GEMM kernel vs GCC intrinsics.
 * Uses rdtsc for cycle-accurate measurement, volatile sink to prevent DCE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ---- rdtsc ---- */
static inline uint64_t rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* ---- Intrinsics 6x32 microkernel (same as gemm.c) ---- */
static void ukernel_6x32_intrinsics(
    int K,
    const float* a0, const float* a1, const float* a2,
    const float* a3, const float* a4, const float* a5,
    const float* B, int ldb,
    float* C, int ldc)
{
    __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
    __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
    __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
    __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
    __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
    __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();

    for (int k = 0; k < K; k++) {
        const float* bp = B + k * ldb;
        __m512 b0 = _mm512_loadu_ps(bp);
        __m512 b1 = _mm512_loadu_ps(bp + 16);
        __m512 va;
        va=_mm512_set1_ps(a0[k]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01);
        va=_mm512_set1_ps(a1[k]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11);
        va=_mm512_set1_ps(a2[k]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21);
        va=_mm512_set1_ps(a3[k]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31);
        va=_mm512_set1_ps(a4[k]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41);
        va=_mm512_set1_ps(a5[k]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51);
    }
    _mm512_storeu_ps(C+0*ldc,c00);_mm512_storeu_ps(C+0*ldc+16,c01);
    _mm512_storeu_ps(C+1*ldc,c10);_mm512_storeu_ps(C+1*ldc+16,c11);
    _mm512_storeu_ps(C+2*ldc,c20);_mm512_storeu_ps(C+2*ldc+16,c21);
    _mm512_storeu_ps(C+3*ldc,c30);_mm512_storeu_ps(C+3*ldc+16,c31);
    _mm512_storeu_ps(C+4*ldc,c40);_mm512_storeu_ps(C+4*ldc+16,c41);
    _mm512_storeu_ps(C+5*ldc,c50);_mm512_storeu_ps(C+5*ldc+16,c51);
}

/* ---- Assembly kernel (from gemm_avx512.S) ---- */
typedef struct {
    int K;
    int _pad;
    const float* a[6];
    const float* B;
    int ldb_bytes;
    int _pad2;
    float* C;
    int ldc_bytes;
} GemmKernArgs;

extern void gemm_6x32_kern(const GemmKernArgs* args);

/* Volatile sink to prevent dead code elimination */
static volatile float g_sink;

int main(void)
{
    /* Test sizes representative of YOLO layers */
    int test_K[] = {16, 32, 48, 64, 128, 256};
    int ntests = sizeof(test_K) / sizeof(test_K[0]);

    printf("=== Assembly vs Intrinsics GEMM 6x32 Benchmark ===\n");
    printf("%-8s %12s %12s %8s\n", "K", "Intrinsics", "Assembly", "Ratio");
    printf("%-8s %12s %12s %8s\n", "", "(cycles)", "(cycles)", "(asm/int)");
    printf("-----------------------------------------------\n");

    for (int ti = 0; ti < ntests; ti++) {
        int K = test_K[ti];
        int ldb = 32;  /* NR=32 */
        int ldc = 32;

        /* Allocate aligned buffers */
        float* A = (float*)_mm_malloc(6 * K * sizeof(float), 64);
        float* B = (float*)_mm_malloc(K * 32 * sizeof(float), 64);
        float* C1 = (float*)_mm_malloc(6 * 32 * sizeof(float), 64);
        float* C2 = (float*)_mm_malloc(6 * 32 * sizeof(float), 64);

        /* Fill with small values */
        for (int i = 0; i < 6 * K; i++) A[i] = 0.01f * (i % 17);
        for (int i = 0; i < K * 32; i++) B[i] = 0.01f * (i % 13);

        int ITERS = 10000;
        uint64_t best_intr = UINT64_MAX;
        uint64_t best_asm = UINT64_MAX;

        /* Warmup */
        for (int i = 0; i < 100; i++) {
            memset(C1, 0, 6 * 32 * sizeof(float));
            ukernel_6x32_intrinsics(K,
                A, A+K, A+2*K, A+3*K, A+4*K, A+5*K,
                B, ldb, C1, ldc);
            g_sink = C1[0];
        }

        /* Benchmark intrinsics */
        for (int i = 0; i < ITERS; i++) {
            memset(C1, 0, 6 * 32 * sizeof(float));
            _mm_mfence();
            uint64_t t0 = rdtsc();
            ukernel_6x32_intrinsics(K,
                A, A+K, A+2*K, A+3*K, A+4*K, A+5*K,
                B, ldb, C1, ldc);
            uint64_t t1 = rdtsc();
            g_sink = C1[0] + C1[95] + C1[191]; /* use result */
            uint64_t dt = t1 - t0;
            if (dt < best_intr) best_intr = dt;
        }

        /* Warmup assembly */
        GemmKernArgs args;
        args.K = K;
        args.a[0] = A; args.a[1] = A+K; args.a[2] = A+2*K;
        args.a[3] = A+3*K; args.a[4] = A+4*K; args.a[5] = A+5*K;
        args.B = B;
        args.ldb_bytes = ldb * 4;
        args.C = C2;
        args.ldc_bytes = ldc * 4;

        for (int i = 0; i < 100; i++) {
            memset(C2, 0, 6 * 32 * sizeof(float));
            gemm_6x32_kern(&args);
            g_sink = C2[0];
        }

        /* Benchmark assembly */
        for (int i = 0; i < ITERS; i++) {
            memset(C2, 0, 6 * 32 * sizeof(float));
            _mm_mfence();
            uint64_t t0 = rdtsc();
            gemm_6x32_kern(&args);
            uint64_t t1 = rdtsc();
            g_sink = C2[0] + C2[95] + C2[191]; /* use result */
            uint64_t dt = t1 - t0;
            if (dt < best_asm) best_asm = dt;
        }

        /* Verify correctness */
        float max_err = 0;
        for (int i = 0; i < 6 * 32; i++) {
            float err = fabsf(C1[i] - C2[i]);
            if (err > max_err) max_err = err;
        }

        double ratio = (double)best_asm / (double)best_intr;
        double flops = 2.0 * 6 * 32 * K;
        printf("K=%-5d  %10llu    %10llu    %.3f   (err=%.2e, %.0f FLOPs)\n",
               K, (unsigned long long)best_intr, (unsigned long long)best_asm,
               ratio, max_err, flops);

        _mm_free(A); _mm_free(B); _mm_free(C1); _mm_free(C2);
    }

    printf("\nRatio < 1.0 = assembly is faster\n");
    return 0;
}
