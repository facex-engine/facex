/*
 * gemm_sparse_int8.c — Sparse INT8 GEMM with AVX-512 VNNI.
 *
 * Block-sparse: weights stored in CSR-like format with 1×4 blocks.
 * Zero blocks are skipped entirely → 75% sparsity = 4x fewer ops.
 * VNNI VPDPBUSD: 4 INT8 multiplies per 32-bit lane per cycle.
 *
 * Combined: sparse × INT8 = up to 16x over dense FP32 GEMM.
 *
 * Weight format (sparse packed):
 *   For each output channel group (NR=16 channels):
 *     [uint16 nnz_blocks]                    - number of non-zero K/4 blocks
 *     [nnz_blocks × uint16 block_indices]    - which K/4 blocks are non-zero
 *     [nnz_blocks × 64 bytes]               - packed weights (16 cols × 4 k-values)
 *
 * Execution: iterate only over non-zero blocks per output group.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

/* ============ Sparse weight packing ============ */

typedef struct {
    void*    data;      /* packed sparse data */
    size_t   data_size;
    int      K, N;      /* original dimensions */
    int      K4;        /* K padded to 4 */
    float    sparsity;  /* fraction of zero blocks */
} SparseINT8Weights;

SparseINT8Weights* nn2_sparse_int8_pack(
    const int8_t* weights, /* [N, K] dense weights */
    int K, int N,
    int32_t* col_sums)     /* [N] output: 128 * sum(w) per channel */
{
    int K4 = (K + 3) & ~3;
    int n_blocks = K4 / 4;

    /* First pass: count non-zero blocks per group to size buffer */
    size_t total_size = 0;
    int total_nnz = 0, total_blocks = 0;

    for (int co = 0; co < N; co += 16) {
        int nr = (co + 16 <= N) ? 16 : N - co;
        int nnz = 0;
        for (int b = 0; b < n_blocks; b++) {
            /* Check if any of the 16 channels × 4 k-values is non-zero */
            int has_nonzero = 0;
            for (int j = 0; j < nr && !has_nonzero; j++) {
                for (int kk = 0; kk < 4 && !has_nonzero; kk++) {
                    int k = b * 4 + kk;
                    if (k < K && weights[(co + j) * K + k] != 0)
                        has_nonzero = 1;
                }
            }
            if (has_nonzero) nnz++;
        }
        total_size += 2 + nnz * 2 + nnz * 64; /* header + indices + data */
        total_nnz += nnz;
        total_blocks += n_blocks;
    }

    SparseINT8Weights* sw = (SparseINT8Weights*)malloc(sizeof(SparseINT8Weights));
    sw->data = malloc(total_size);
    sw->data_size = total_size;
    sw->K = K;
    sw->N = N;
    sw->K4 = K4;
    sw->sparsity = 1.0f - (float)total_nnz / total_blocks;

    /* Second pass: pack */
    uint8_t* out = (uint8_t*)sw->data;

    for (int co = 0; co < N; co += 16) {
        int nr = (co + 16 <= N) ? 16 : N - co;

        /* Col sums for compensation */
        for (int j = 0; j < nr; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int32_t)weights[(co + j) * K + k];
            col_sums[co + j] = 128 * sum;
        }

        /* Find non-zero blocks */
        uint16_t nnz = 0;
        uint16_t indices[4096]; /* max K4/4 blocks */
        for (int b = 0; b < n_blocks; b++) {
            int has_nz = 0;
            for (int j = 0; j < nr && !has_nz; j++)
                for (int kk = 0; kk < 4 && !has_nz; kk++) {
                    int k = b * 4 + kk;
                    if (k < K && weights[(co + j) * K + k] != 0) has_nz = 1;
                }
            if (has_nz)
                indices[nnz++] = (uint16_t)b;
        }

        /* Write header */
        memcpy(out, &nnz, 2); out += 2;
        memcpy(out, indices, nnz * 2); out += nnz * 2;

        /* Write non-zero block data: [16 cols × 4 k-values = 64 bytes] per block */
        for (int bi = 0; bi < nnz; bi++) {
            int b = indices[bi];
            for (int j = 0; j < 16; j++) {
                for (int kk = 0; kk < 4; kk++) {
                    int k = b * 4 + kk;
                    if (j < nr && k < K)
                        *out++ = (uint8_t)weights[(co + j) * K + k];
                    else
                        *out++ = 0;
                }
            }
        }
    }

    return sw;
}

void nn2_sparse_int8_free(SparseINT8Weights* sw)
{
    if (sw) { free(sw->data); free(sw); }
}

/* ============ Sparse INT8 GEMM with VNNI ============ */

#ifdef __AVX512VNNI__

void nn2_sparse_int8_gemm(
    const int8_t* A, int M, int K,  /* A[M, K] activations (will be quantized) */
    const SparseINT8Weights* sw,
    const int32_t* col_sums,
    const float* w_scales,          /* [N] per-channel */
    const float* bias,              /* [N] or NULL */
    float act_scale,                /* per-tensor activation scale */
    int act_type,                   /* 0=none, 1=SiLU, 2=ReLU */
    float* output)                  /* [M, N] float32 output */
{
    int N = sw->N;
    int K4 = sw->K4;

    /* Pad A to K4 if needed */
    const int8_t* A_eff = A;
    int8_t* A_pad = NULL;
    if (K != K4) {
        A_pad = (int8_t*)calloc(M * K4, 1);
        for (int m = 0; m < M; m++)
            memcpy(A_pad + m * K4, A + m * K, K);
        A_eff = A_pad;
    }

    const uint32_t xor32 = 0x80808080u;
    const uint8_t* wptr = (const uint8_t*)sw->data;

    for (int co = 0; co < N; co += 16) {
        /* Read sparse header */
        uint16_t nnz;
        memcpy(&nnz, wptr, 2); wptr += 2;
        const uint16_t* block_indices = (const uint16_t*)wptr;
        wptr += nnz * 2;
        const uint8_t* block_data = wptr;
        wptr += nnz * 64;

        /* Process 4 activation rows at a time */
        for (int m = 0; m < M; m += 4) {
            int mr = (m + 4 <= M) ? 4 : M - m;

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            const int8_t* a0 = A_eff + m * K4;
            const int8_t* a1 = (mr > 1) ? a0 + K4 : a0;
            const int8_t* a2 = (mr > 2) ? a0 + 2*K4 : a0;
            const int8_t* a3 = (mr > 3) ? a0 + 3*K4 : a0;

            /* Only iterate over non-zero blocks! */
            for (int bi = 0; bi < nnz; bi++) {
                int b = block_indices[bi];
                int k_off = b * 4;

                __m512i vb = _mm512_loadu_si512((const __m512i*)(block_data + bi * 64));

                /* Load and convert 4 activation bytes per row */
                uint32_t a0v, a1v, a2v, a3v;
                memcpy(&a0v, a0 + k_off, 4); a0v ^= xor32;
                memcpy(&a1v, a1 + k_off, 4); a1v ^= xor32;
                memcpy(&a2v, a2 + k_off, 4); a2v ^= xor32;
                memcpy(&a3v, a3 + k_off, 4); a3v ^= xor32;

                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(a0v), vb);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(a1v), vb);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(a2v), vb);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(a3v), vb);
            }

            /* Fused epilogue: dequant + bias + activation */
            int nr = (co + 16 <= N) ? 16 : N - co;
            int32_t acc_buf[4][16];
            _mm512_storeu_si512(acc_buf[0], acc0);
            _mm512_storeu_si512(acc_buf[1], acc1);
            _mm512_storeu_si512(acc_buf[2], acc2);
            _mm512_storeu_si512(acc_buf[3], acc3);

            for (int i = 0; i < mr; i++) {
                float* orow = output + (m + i) * N + co;
                for (int j = 0; j < nr; j++) {
                    float val = (float)(acc_buf[i][j] - col_sums[co + j])
                                * act_scale * w_scales[co + j];
                    if (bias) val += bias[co + j];
                    if (act_type == 1) {
                        float s = 1.0f / (1.0f + expf(-val));
                        val *= s;
                    } else if (act_type == 2 && val < 0) {
                        val = 0;
                    }
                    orow[j] = val;
                }
            }
        }
    }

    free(A_pad);
}

#endif /* __AVX512VNNI__ */

/* ============ Benchmark helper ============ */

void nn2_sparse_int8_bench(int M, int K, int N, float target_sparsity)
{
    /* Create weights with target sparsity */
    int8_t* w = (int8_t*)malloc(N * K);
    for (int i = 0; i < N * K; i++)
        w[i] = (rand() % 100 < (int)(target_sparsity * 100)) ? 0 : (int8_t)(rand() % 20 - 10);

    int32_t* csums = (int32_t*)calloc(N, sizeof(int32_t));
    SparseINT8Weights* sw = nn2_sparse_int8_pack(w, K, N, csums);

    float* wscales = (float*)malloc(N * sizeof(float));
    float* bias = (float*)calloc(N, sizeof(float));
    for (int i = 0; i < N; i++) wscales[i] = 0.01f;

    int8_t* a = (int8_t*)calloc(M * K, 1);
    float* out = (float*)malloc(M * N * sizeof(float));

    printf("  Sparse INT8 [%d,%d,%d] sparsity=%.0f%%: packed %.1f KB (%.0f%% of dense)\n",
           M, K, N, sw->sparsity * 100,
           sw->data_size / 1024.0,
           sw->data_size * 100.0 / (N * K));

#ifdef __AVX512VNNI__
    /* Warmup */
    nn2_sparse_int8_gemm(a, M, K, sw, csums, wscales, bias, 0.05f, 0, out);

    /* Bench */
    /* (timing done externally) */
#endif

    free(w); free(csums); free(wscales); free(bias); free(a); free(out);
    nn2_sparse_int8_free(sw);
}
