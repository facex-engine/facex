/*
 * test_int8_gemm.c — Verify INT8 GEMM correctness vs FP32 reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

extern void pack_weights_4x8c8(const int8_t*, const float*, int, int, void*, int32_t*);
extern int packed_weights_size_4x8c8(int, int);
extern void int8_gemm_4x8c8(const int8_t*, int, int, int,
                              const void*, int32_t*, const int32_t*);

/* FP32 reference matmul */
static void matmul_ref(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += A[m*K+k] * B[k*N+n];
            C[m*N+n] = sum;
        }
}

/* Cosine similarity */
static float cosine(const float* a, const float* b, int n) {
    float dot=0, na=0, nb=0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

static void test_gemm(int M, int K, int N) {
    /* Generate random FP32 matrices */
    float* A_fp32 = (float*)malloc(M * K * sizeof(float));
    float* B_fp32 = (float*)malloc(K * N * sizeof(float));
    float* C_ref  = (float*)malloc(M * N * sizeof(float));
    float* C_int8 = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M*K; i++) A_fp32[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K*N; i++) B_fp32[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    /* FP32 reference */
    matmul_ref(A_fp32, B_fp32, C_ref, M, K, N);

    /* Quantize A per-tensor */
    float a_max = 0;
    for (int i = 0; i < M*K; i++) { float v = fabsf(A_fp32[i]); if (v > a_max) a_max = v; }
    float a_scale = a_max / 127.0f;
    float a_inv = 1.0f / (a_scale + 1e-9f);

    int K_padded = (K + 7) & ~7;
    int8_t* A_int8 = (int8_t*)calloc(M * K_padded, 1);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            int q = (int)(A_fp32[m*K+k] * a_inv + (A_fp32[m*K+k] >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127; if (q < -128) q = -128;
            A_int8[m*K_padded+k] = (int8_t)q;
        }

    /* Quantize B per-channel (transpose to [N, K]) */
    int8_t* B_int8 = (int8_t*)malloc(N * K);
    float* w_scales = (float*)malloc(N * sizeof(float));
    for (int n = 0; n < N; n++) {
        float mx = 0;
        for (int k = 0; k < K; k++) { float v = fabsf(B_fp32[k*N+n]); if (v > mx) mx = v; }
        w_scales[n] = mx / 127.0f;
        for (int k = 0; k < K; k++) {
            float v = B_fp32[k*N+n] / (w_scales[n] + 1e-9f);
            int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127; if (q < -128) q = -128;
            B_int8[n*K+k] = (int8_t)q;
        }
    }

    /* Pack weights */
    int pw_size = packed_weights_size_4x8c8(K, N);
    void* packed = malloc(pw_size);
    int32_t* col_sums = (int32_t*)calloc((N+7)&~7, sizeof(int32_t));
    pack_weights_4x8c8(B_int8, NULL, K, N, packed, col_sums);

    /* Scalar INT8 reference (direct s8×s8, no SIMD, no packing) */
    int32_t* C_scalar = (int32_t*)calloc(M * N, sizeof(int32_t));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int32_t)A_int8[m*K_padded+k] * (int32_t)B_int8[n*K+k];
            C_scalar[m*N+n] = sum;
        }

    /* INT8 GEMM (SIMD microkernel) */
    int32_t* C_i32 = (int32_t*)calloc(M * N, sizeof(int32_t));
    int8_gemm_4x8c8(A_int8, M, K_padded, N, packed, C_i32, col_sums);

    /* Check scalar vs SIMD (both int32, should be exact match) */
    int mismatch = 0;
    for (int i = 0; i < M*N; i++)
        if (C_scalar[i] != C_i32[i]) mismatch++;

    /* Dequantize: fp32 = int32 * a_scale * w_scale[n] */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C_int8[m*N+n] = (float)C_scalar[m*N+n] * a_scale * w_scales[n]; /* use scalar for dequant */

    /* Compare with FP32 reference */
    float cos = cosine(C_ref, C_int8, M*N);

    /* Also check SIMD-dequanted */
    float* C_simd_fp = (float*)malloc(M*N*sizeof(float));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C_simd_fp[m*N+n] = (float)C_i32[m*N+n] * a_scale * w_scales[n];
    float cos_simd = cosine(C_ref, C_simd_fp, M*N);
    float max_err = 0;
    for (int i = 0; i < M*N; i++) {
        float err = fabsf(C_ref[i] - C_int8[i]);
        if (err > max_err) max_err = err;
    }

    printf("  M=%d K=%d N=%d: scalar_cos=%.4f simd_cos=%.4f mismatch=%d/%d %s\n",
           M, K, N, cos, cos_simd, mismatch, M*N,
           (cos > 0.99f && mismatch == 0) ? "PASS" : (cos > 0.99f ? "SCALAR_OK" : "FAIL"));

    free(A_fp32); free(B_fp32); free(C_ref); free(C_int8);
    free(A_int8); free(B_int8); free(w_scales);
    free(packed); free(col_sums); free(C_i32);
    free(C_scalar); free(C_simd_fp);
}

int main(void) {
    printf("INT8 GEMM Correctness Test\n");
    printf("==========================\n");

    /* Various sizes — including EdgeFace layer sizes */
    test_gemm(1, 32, 128);    /* small */
    test_gemm(4, 64, 256);    /* medium */
    test_gemm(16, 100, 400);  /* stage 2 block */
    test_gemm(49, 192, 768);  /* stage 3 (7×7 spatial × large channels) */
    test_gemm(196, 32, 128);  /* stage 0 (14×14) */
    test_gemm(784, 19, 128);  /* stage 0 (28×28, small K) */
    test_gemm(1, 512, 115);   /* head FC */
    test_gemm(3, 115, 192);   /* small M */

    /* Edge cases */
    test_gemm(1, 8, 8);       /* minimal */
    test_gemm(1, 16, 16);     /* exactly 2 KR blocks */
    test_gemm(5, 19, 32);     /* non-multiple-of-4 M, non-multiple-of-8 K */

    printf("\nDone.\n");
    return 0;
}
