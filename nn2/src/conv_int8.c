/*
 * conv_int8.c — INT8 quantized convolution using AVX-512 VNNI.
 *
 * Pipeline per conv layer:
 *   1. Quantize FP32 input to INT8 (per-tensor scale)
 *   2. INT8 GEMM with VNNI: VPDPBUSD (4×u8×s8→s32)
 *   3. Dequantize INT32 accumulators to FP32
 *   4. Add bias + apply SiLU activation
 *   5. Output FP32 for next layer
 *
 * Throughput: 4× over FP32 for compute-bound layers.
 * Memory: 4× less weight data → better cache utilization.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "fast_exp.h"

#ifdef __AVX512VNNI__

/* Quantize FP32 tensor to INT8 with per-tensor scale */
static void quantize_f32_to_s8(const float* in, int n, float scale, int8_t* out)
{
    __m512 vscale = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(in + i), vscale);
        __m512i vi = _mm512_cvtps_epi32(v);
        /* Clamp to [-127, 127] and pack to int8 */
        vi = _mm512_max_epi32(vi, _mm512_set1_epi32(-127));
        vi = _mm512_min_epi32(vi, _mm512_set1_epi32(127));
        /* Pack 16 int32 → 16 int8 (lower bytes) */
        __m128i packed = _mm512_cvtsepi32_epi8(vi);
        _mm_storeu_si128((__m128i*)(out + i), packed);
    }
    for (; i < n; i++) {
        int v = (int)roundf(in[i] * scale);
        out[i] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
    }
}

/* INT8 GEMM: C_s32[M,N] = A_u8[M,K] × B_s8_packed[K,N]
 * A is converted from s8 to u8 via +128 offset (XOR trick).
 * B is pre-packed in VNNI format: [N/16 groups][K/4 blocks][16×4 bytes].
 * Compensation: col_sums[n] = 128 * sum(B[:,n]) per output channel. */
static void int8_gemm_16x16(
    const int8_t* A, int M, int K, int N,
    const int8_t* B_packed,  /* [N/16][K/4][64 bytes] */
    int32_t* C)
{
    int K4 = (K + 3) & ~3;
    const uint32_t xor32 = 0x80808080u;

    for (int m = 0; m < M; m++) {
        const int8_t* a_row = A + m * K;
        const uint8_t* bp = (const uint8_t*)B_packed;

        for (int n = 0; n < N; n += 16) {
            __m512i acc = _mm512_setzero_si512();

            for (int k = 0; k < K4; k += 4) {
                /* Load 4 A bytes, convert s8→u8, broadcast to all lanes */
                uint32_t a_val;
                if (k + 4 <= K)
                    memcpy(&a_val, a_row + k, 4);
                else {
                    a_val = 0;
                    memcpy(&a_val, a_row + k, K - k);
                }
                a_val ^= xor32;
                __m512i va = _mm512_set1_epi32(a_val);

                /* Load B panel: 16 channels × 4 k-values = 64 bytes */
                __m512i vb = _mm512_loadu_si512((const __m512i*)bp);
                bp += 64;

                /* VPDPBUSD: acc += u8(A) dot s8(B) per 32-bit lane */
                acc = _mm512_dpbusd_epi32(acc, va, vb);
            }

            _mm512_storeu_si512((__m512i*)(C + m * N + n), acc);
        }
    }
}

/* Dequantize INT32 accumulators to FP32 and apply bias + activation.
 * result = (acc - compensation) * (input_scale * weight_scale) + bias
 * then apply SiLU or LeakyReLU. */
static void dequant_bias_act(
    const int32_t* acc, int M, int N,
    const int32_t* col_sums, /* [N] */
    float input_scale,       /* 1/act_scale (to convert back) */
    const float* w_scales,   /* [N] per-channel */
    const float* bias,       /* [N] */
    int act,
    float* output)
{
    float inv_input = 1.0f / (input_scale + 1e-10f);
    for (int m = 0; m < M; m++) {
        int i = 0;
#ifdef __AVX512F__
        for (; i + 16 <= N; i += 16) {
            __m512i vacc = _mm512_loadu_si512((const __m512i*)(acc + m*N + i));
            __m512i vcomp = _mm512_loadu_si512((const __m512i*)(col_sums + i));
            __m512 vfacc = _mm512_cvtepi32_ps(_mm512_sub_epi32(vacc, vcomp));
            __m512 vws = _mm512_loadu_ps(w_scales + i);
            __m512 vb = _mm512_loadu_ps(bias + i);
            /* result = acc * inv_input * w_scale + bias */
            __m512 result = _mm512_fmadd_ps(
                _mm512_mul_ps(vfacc, _mm512_set1_ps(inv_input)), vws, vb);
            if (act == 1) result = _mm512_silu_fast(result);
            _mm512_storeu_ps(output + m*N + i, result);
        }
#endif
        for (; i < N; i++) {
            float val = (float)(acc[m*N+i] - col_sums[i]) * inv_input * w_scales[i] + bias[i];
            if (act == 1) { float s = 1.0f/(1.0f+expf(-val)); val *= s; }
            else if (act == 3 && val < 0) val *= 0.1f;
            output[m*N+i] = val;
        }
    }
}

/* Pack INT8 weights into VNNI format: [N/16][K/4][64 bytes]
 * Also compute col_sums for unsigned offset compensation. */
void nn2_int8_pack_vnni(const int8_t* weights, int N, int K,
                         int8_t* packed, int32_t* col_sums)
{
    int K4 = (K + 3) & ~3;
    for (int n = 0; n < N; n += 16) {
        int nr = (n + 16 <= N) ? 16 : N - n;
        /* Col sums */
        for (int j = 0; j < nr; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int32_t)weights[(n+j)*K + k];
            col_sums[n+j] = 128 * sum;
        }
        /* Pack: [K4/4 blocks][16 × 4 bytes] */
        for (int k = 0; k < K4; k += 4) {
            for (int j = 0; j < 16; j++) {
                for (int kk = 0; kk < 4; kk++) {
                    int idx = (n/16) * (K4/4) * 64 + (k/4) * 64 + j * 4 + kk;
                    if (j < nr && (k+kk) < K)
                        packed[idx] = weights[(n+j)*K + (k+kk)];
                    else
                        packed[idx] = 0;
                }
            }
        }
    }
}

/* Full INT8 conv: quantize input → INT8 GEMM → dequant+bias+act → FP32 output */
void nn2_conv2d_int8(
    const float* input, int Cin, int H, int W,
    const int8_t* w_packed,  /* VNNI-packed INT8 weights */
    const int32_t* col_sums,
    const float* w_scales,   /* per-channel weight scales */
    const float* bias,
    int Cout, int k, int stride, int pad, int act,
    float act_scale,         /* per-tensor activation scale */
    float* output,
    float* col_buf,          /* im2col scratch */
    int8_t* quant_buf)       /* quantized activation scratch */
{
    int Hout = (H + 2*pad - k) / stride + 1;
    int Wout = (W + 2*pad - k) / stride + 1;
    int spatial = Hout * Wout;
    int K = Cin * k * k;
    int K4 = (K + 3) & ~3;

    /* Step 1: im2col (still FP32) */
    if (k > 1)
        nn2_im2col(input, Cin, H, W, k, k, stride, pad, col_buf);
    else
        col_buf = (float*)input; /* 1x1: no im2col */

    /* Step 2: Quantize activations to INT8 */
    int act_size = K * spatial;
    quantize_f32_to_s8(col_buf, act_size, act_scale, quant_buf);

    /* Step 3: INT8 GEMM with VNNI */
    /* Note: GEMM expects A[M,K] × B_packed → C[M,N]
     * For conv: M=spatial, K=Cin*k*k, N=Cout */
    /* But our B_packed is [Cout, K] packed per-Cout-group.
     * We need transposed: C^T[spatial,Cout] = A^T[spatial,K] × B_packed
     * Actually: output[Cout,spatial] = Weight[Cout,K] × im2col[K,spatial]
     * INT8 GEMM: iterate output channels in groups of 16 */
    int32_t* acc = (int32_t*)malloc(Cout * spatial * sizeof(int32_t));

    /* Simple INT8 GEMM: for each spatial position, accumulate across K */
    int N16 = (Cout + 15) & ~15;
    memset(acc, 0, N16 * spatial * sizeof(int32_t));

    /* Transpose approach: process per-spatial-position */
    for (int s = 0; s < spatial; s++) {
        const int8_t* a = quant_buf + s; /* stride = spatial (column of im2col) */
        /* Need contiguous A for VNNI: copy column to contiguous buffer */
        int8_t a_col[2048]; /* max K */
        for (int kk = 0; kk < K; kk++)
            a_col[kk] = quant_buf[kk * spatial + s];
        /* Zero-pad to K4 */
        for (int kk = K; kk < K4; kk++) a_col[kk] = 0;

        /* VNNI: accumulate for all Cout channels */
        const uint8_t* bp = (const uint8_t*)w_packed;
        uint32_t xor32 = 0x80808080u;

        for (int co = 0; co < Cout; co += 16) {
            __m512i vacc = _mm512_setzero_si512();
            for (int kk = 0; kk < K4; kk += 4) {
                uint32_t a_val;
                memcpy(&a_val, a_col + kk, 4);
                a_val ^= xor32;
                __m512i va = _mm512_set1_epi32(a_val);
                __m512i vb = _mm512_loadu_si512((const __m512i*)bp);
                bp += 64;
                vacc = _mm512_dpbusd_epi32(vacc, va, vb);
            }
            /* Store accumulator */
            for (int j = 0; j < 16 && (co+j) < Cout; j++)
                acc[(co+j) * spatial + s] = ((int32_t*)&vacc)[j];
        }
    }

    /* Step 4: Dequantize + bias + activation */
    for (int co = 0; co < Cout; co++) {
        float ws = w_scales[co];
        float b = bias[co];
        float inv_as = 1.0f / (act_scale + 1e-10f);
        int i = 0;
#ifdef __AVX512F__
        __m512 v_inv = _mm512_set1_ps(inv_as * ws);
        __m512 v_b = _mm512_set1_ps(b);
        __m512i v_comp = _mm512_set1_epi32(col_sums[co]);
        for (; i + 16 <= spatial; i += 16) {
            __m512i vacc = _mm512_loadu_si512((const __m512i*)(acc + co*spatial + i));
            __m512 vf = _mm512_cvtepi32_ps(_mm512_sub_epi32(vacc, v_comp));
            __m512 result = _mm512_fmadd_ps(vf, v_inv, v_b);
            if (act == 1) result = _mm512_silu_fast(result);
            _mm512_storeu_ps(output + co*spatial + i, result);
        }
#endif
        for (; i < spatial; i++) {
            float val = (float)(acc[co*spatial+i] - col_sums[co]) * inv_as * ws + b;
            if (act == 1) { float s = 1.0f/(1.0f+expf(-val)); val *= s; }
            else if (act == 3 && val < 0) val *= 0.1f;
            output[co*spatial+i] = val;
        }
    }

    free(acc);
}

#endif /* __AVX512VNNI__ */
