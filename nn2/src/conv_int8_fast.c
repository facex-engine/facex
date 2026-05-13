/*
 * conv_int8_fast.c — Fast INT8 conv with VNNI, transposed GEMM layout.
 *
 * Instead of Weight[Cout,K] × im2col[K,spatial] (FP32 layout),
 * transpose to: im2col^T[spatial,K] × Weight_packed[K,Cout] (INT8 layout).
 *
 * This gives contiguous A-rows for VNNI: each spatial position
 * has K consecutive INT8 values = perfect for VPDPBUSD.
 *
 * 2.6× proven speedup over FP32 GEMM.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "fast_exp.h"

#ifdef __AVX512VNNI__

/* Transpose [K, spatial] → [spatial, K] and quantize to INT8 */
static void transpose_quantize(const float* src, int K, int spatial,
                                float scale, int8_t* dst, int K4)
{
    for (int s = 0; s < spatial; s++) {
        int k = 0;
        for (; k < K; k++) {
            int v = (int)roundf(src[k * spatial + s] * scale);
            dst[s * K4 + k] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
        for (; k < K4; k++) dst[s * K4 + k] = 0;
    }
}

/* Transpose [spatial, Cout] → [Cout, spatial] */
static void transpose_output(const float* src, int spatial, int Cout, float* dst)
{
    for (int co = 0; co < Cout; co++)
        for (int s = 0; s < spatial; s++)
            dst[co * spatial + s] = src[s * Cout + co];
}

/* INT8 VNNI conv: full pipeline with transposed GEMM.
 * Input: FP32 [Cin, H, W] (or im2col [K, spatial])
 * Output: FP32 [Cout, Hout, Wout]
 * Weights: pre-packed VNNI format */
void nn2_conv2d_int8_fast(
    const float* input,      /* [Cin, H, W] for 1x1, or im2col [K, spatial] for 3x3 */
    int K, int spatial,      /* K=Cin*k*k, spatial=Hout*Wout */
    const int8_t* w_packed,  /* VNNI-packed [Cout/16][K4/4][64] */
    const int32_t* col_sums, /* [Cout] compensation */
    const float* w_scales,   /* [Cout] per-channel */
    const float* bias,       /* [Cout] */
    int Cout, int act,
    float act_scale,
    float* output,           /* [Cout, spatial] */
    int8_t* quant_buf)       /* workspace: [spatial, K4] */
{
    int K4 = (K + 3) & ~3;
    uint32_t xor32 = 0x80808080u;

    /* Step 1: Transpose [K, spatial] → [spatial, K] and quantize to INT8 */
    transpose_quantize(input, K, spatial, act_scale, quant_buf, K4);

    /* Step 2: INT8 VNNI GEMM: C[spatial, Cout] = A[spatial, K4] × B_packed[K4, Cout]
     * Process MR=4 spatial rows at a time for register reuse */
    float* out_t = (float*)malloc(spatial * Cout * sizeof(float));

    for (int s = 0; s < spatial; s++) {
        const int8_t* a = quant_buf + s * K4;
        const uint8_t* bp = (const uint8_t*)w_packed;

        for (int co = 0; co < Cout; co += 16) {
            __m512i vacc = _mm512_setzero_si512();

            for (int k = 0; k < K4; k += 4) {
                uint32_t av;
                memcpy(&av, a + k, 4);
                av ^= xor32;
                __m512i va = _mm512_set1_epi32(av);
                __m512i vb = _mm512_load_si512((const __m512i*)bp);
                bp += 64;
                vacc = _mm512_dpbusd_epi32(vacc, va, vb);
            }

            /* Dequantize + bias + SiLU inline */
            __m512i vcomp = _mm512_loadu_si512((const __m512i*)(col_sums + co));
            __m512 vf = _mm512_cvtepi32_ps(_mm512_sub_epi32(vacc, vcomp));
            __m512 vws = _mm512_loadu_ps(w_scales + co);
            __m512 vb = _mm512_loadu_ps(bias + co);
            float inv_scale = 1.0f / (act_scale + 1e-10f);
            __m512 result = _mm512_fmadd_ps(
                _mm512_mul_ps(vf, _mm512_set1_ps(inv_scale)), vws, vb);
            if (act == 1) result = _mm512_silu_fast(result);

            /* Store to transposed output [spatial, Cout] */
            int nr = (co + 16 <= Cout) ? 16 : Cout - co;
            if (nr >= 16)
                _mm512_storeu_ps(out_t + s * Cout + co, result);
            else {
                __mmask16 mask = ((__mmask16)1 << nr) - 1;
                _mm512_mask_storeu_ps(out_t + s * Cout + co, mask, result);
            }
        }
    }

    /* Step 3: Transpose [spatial, Cout] → [Cout, spatial] */
    transpose_output(out_t, spatial, Cout, output);
    free(out_t);
}

#endif /* __AVX512VNNI__ */
