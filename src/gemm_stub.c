/*
 * gemm_stub.c — Stub for INT8 GEMM when AVX2 is not available (WASM build).
 * The engine falls back to FP32 matmul when packed weights are NULL.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void pack_weights_4x8c8(const int8_t* w, const float* scales, int K, int N,
                         void* packed, int32_t* col_sums) {
    (void)w; (void)scales; (void)K; (void)N; (void)packed; (void)col_sums;
}

int packed_weights_size_4x8c8(int K, int N) {
    (void)K; (void)N;
    return 64; /* minimum allocation */
}

void int8_gemm_4x8c8(int M, int N, int K,
                      const uint8_t* A, int lda,
                      const void* B_packed,
                      const int32_t* col_sums,
                      float* C, int ldc,
                      const float* a_scales,
                      const float* b_scales,
                      float a_zero) {
    (void)M; (void)N; (void)K; (void)A; (void)lda; (void)B_packed;
    (void)col_sums; (void)C; (void)ldc; (void)a_scales; (void)b_scales; (void)a_zero;
    /* Never called in WASM — engine uses FP32 matmul path */
}
