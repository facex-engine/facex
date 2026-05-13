/*
 * gemm_vnni.c — INT8 VNNI GEMM for NHWC pipeline.
 *
 * C[M,N] = A_u8[M,K4] × B_s8_packed[K4,N] via VPDPBUSD
 * Then: dequant + bias + act + requant → INT8 output.
 *
 * A is s8 input converted to u8 via XOR 0x80 (add 128).
 * B is pre-packed VNNI: [N/16][K4/4][16×4 bytes].
 * Compensation: col_sums[n] = 128 * sum(B[:,n]).
 *
 * Threading: parallel across N-tiles (16 channels per tile).
 */

#include "nn2_int8.h"
#include "../nn2_internal.h"
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifdef __AVX512VNNI__

#include "../fast_exp.h"

/* ============ VNNI GEMM worker ============ */

typedef struct {
    const int8_t* A;
    int M, K4, N;
    const int8_t* B_packed;
    const int32_t* col_sums;
    const float* w_scales;
    const float* bias;
    float inv_act_scale;   /* 1 / act_scale */
    float out_inv_scale;   /* 1 / out_scale (for requant) */
    int act;
    int8_t* output;
    int out_stride;
} VnniCtx;

/* MR=4: process 4 spatial rows sharing same B panel loads */
#define VNNI_MR 4

static void vnni_worker(void* arg, int start, int end)
{
    VnniCtx* g = (VnniCtx*)arg;
    const uint32_t xor32 = 0x80808080u;

    for (int ni = start; ni < end; ni++) {
        int n = ni * 16;
        int nr = (n + 16 <= g->N) ? 16 : g->N - n;
        const uint8_t* bp_base = (const uint8_t*)g->B_packed
                                  + (size_t)(ni) * (g->K4 / 4) * 64;

        /* Preload dequant constants (reused across all M rows) */
        __m512i vcomp = _mm512_loadu_si512((const __m512i*)(g->col_sums + n));
        __m512 vws = _mm512_loadu_ps(g->w_scales + n);
        __m512 vbias = _mm512_loadu_ps(g->bias + n);
        __m512 vinv = _mm512_set1_ps(g->inv_act_scale);
        __m512 vout_s = _mm512_set1_ps(g->out_inv_scale);
        __m512i vmin = _mm512_set1_epi32(-127);
        __m512i vmax = _mm512_set1_epi32(127);

        int m = 0;
        /* Process MR=4 rows at a time — share B panel loads */
        for (; m + VNNI_MR <= g->M; m += VNNI_MR) {
            const int8_t* a0 = g->A + (m+0) * g->K4;
            const int8_t* a1 = g->A + (m+1) * g->K4;
            const int8_t* a2 = g->A + (m+2) * g->K4;
            const int8_t* a3 = g->A + (m+3) * g->K4;
            const uint8_t* bp = bp_base;

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            for (int k = 0; k < g->K4; k += 4) {
                __m512i vb = _mm512_load_si512((const __m512i*)bp);
                bp += 64;
                uint32_t av;
                memcpy(&av, a0+k, 4); av ^= xor32;
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(av), vb);
                memcpy(&av, a1+k, 4); av ^= xor32;
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(av), vb);
                memcpy(&av, a2+k, 4); av ^= xor32;
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(av), vb);
                memcpy(&av, a3+k, 4); av ^= xor32;
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(av), vb);
            }

            /* Dequant + bias + act + requant for all 4 rows */
            #define VNNI_STORE(i, acc_i) { \
                __m512 vf = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc_i, vcomp)); \
                __m512 res = _mm512_fmadd_ps(_mm512_mul_ps(vf, vinv), vws, vbias); \
                if (g->act == 1) res = _mm512_silu_fast(res); \
                __m512i vi = _mm512_cvtps_epi32(_mm512_mul_ps(res, vout_s)); \
                vi = _mm512_max_epi32(vi, vmin); vi = _mm512_min_epi32(vi, vmax); \
                __m128i pk = _mm512_cvtsepi32_epi8(vi); \
                _mm_storeu_si128((__m128i*)(g->output + (m+(i)) * g->out_stride + n), pk); \
            }
            VNNI_STORE(0, acc0);
            VNNI_STORE(1, acc1);
            VNNI_STORE(2, acc2);
            VNNI_STORE(3, acc3);
            #undef VNNI_STORE
        }

        /* Tail: remaining rows */
        for (; m < g->M; m++) {
            const int8_t* a_row = g->A + m * g->K4;
            const uint8_t* bp = bp_base;
            __m512i vacc = _mm512_setzero_si512();
            for (int k = 0; k < g->K4; k += 4) {
                uint32_t av; memcpy(&av, a_row+k, 4); av ^= xor32;
                vacc = _mm512_dpbusd_epi32(vacc, _mm512_set1_epi32(av),
                       _mm512_load_si512((const __m512i*)bp));
                bp += 64;
            }
            __m512 vf = _mm512_cvtepi32_ps(_mm512_sub_epi32(vacc, vcomp));
            __m512 res = _mm512_fmadd_ps(_mm512_mul_ps(vf, vinv), vws, vbias);
            if (g->act == 1) res = _mm512_silu_fast(res);
            __m512i vi = _mm512_cvtps_epi32(_mm512_mul_ps(res, vout_s));
            vi = _mm512_max_epi32(vi, vmin); vi = _mm512_min_epi32(vi, vmax);
            _mm_storeu_si128((__m128i*)(g->output + m * g->out_stride + n),
                             _mm512_cvtsepi32_epi8(vi));
        }
    }
}

void i8_gemm_bias_act(
    const int8_t* A, int M, int K4, int N,
    const int8_t* B_packed,
    const int32_t* col_sums,
    const float* w_scales,
    const float* bias,
    float act_scale,
    float out_scale,
    int act,
    int8_t* output,
    int out_stride)
{
    float inv_act = 1.0f / (act_scale + 1e-10f);
    float out_inv = 1.0f / (out_scale + 1e-10f);

    VnniCtx ctx = {A, M, K4, N, B_packed, col_sums, w_scales, bias,
                    inv_act, out_inv, act, output, out_stride};

    int n_tiles = (N + 15) / 16;
    int grain = (n_tiles > 32) ? n_tiles / (nn2_tp_num_threads() * 2) : 1;
    if (grain < 1) grain = 1;
    if (n_tiles >= 2)
        nn2_tp_parallel_for(vnni_worker, &ctx, n_tiles, grain);
    else
        vnni_worker(&ctx, 0, n_tiles);
}

/* ============ Weight packing ============ */

void i8_pack_weights(const float* weights, int Cout, int K,
                     int8_t* packed, int32_t* col_sums,
                     float* w_scales)
{
    int K4 = K_PAD4(K);
    int Cout16 = C_PAD16(Cout);

    /* Step 1: Find per-channel scale (absmax) */
    for (int co = 0; co < Cout; co++) {
        float amax = 0;
        for (int k = 0; k < K; k++) {
            float v = fabsf(weights[co * K + k]);
            if (v > amax) amax = v;
        }
        w_scales[co] = amax / 127.0f;
        if (w_scales[co] < 1e-10f) w_scales[co] = 1e-10f;
    }

    /* Step 2: Quantize + pack VNNI format [Cout/16][K4/4][16×4] */
    memset(packed, 0, (size_t)(Cout16 / 16) * (K4 / 4) * 64);

    for (int n = 0; n < Cout; n += 16) {
        int nr = (n + 16 <= Cout) ? 16 : Cout - n;
        int block_base = (n / 16) * (K4 / 4) * 64;

        /* Column sums for compensation */
        for (int j = 0; j < nr; j++) {
            int32_t sum = 0;
            float inv_s = 1.0f / w_scales[n + j];
            for (int k = 0; k < K; k++) {
                int8_t q = (int8_t)roundf(weights[(n + j) * K + k] * inv_s);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                sum += (int32_t)q;
            }
            col_sums[n + j] = 128 * sum;
        }

        /* Pack: [K4/4 blocks][16 channels × 4 k-values] */
        for (int k = 0; k < K4; k += 4) {
            for (int j = 0; j < 16; j++) {
                for (int kk = 0; kk < 4; kk++) {
                    int idx = block_base + (k / 4) * 64 + j * 4 + kk;
                    if (j < nr && (k + kk) < K) {
                        float inv_s = 1.0f / w_scales[n + j];
                        int v = (int)roundf(weights[(n + j) * K + (k + kk)] * inv_s);
                        packed[idx] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
                    }
                }
            }
        }
    }
}

#else /* no VNNI */

void i8_gemm_bias_act(const int8_t* A, int M, int K4, int N,
    const int8_t* B_packed, const int32_t* col_sums,
    const float* w_scales, const float* bias,
    float act_scale, float out_scale, int act,
    int8_t* output, int out_stride)
{
    (void)A;(void)M;(void)K4;(void)N;(void)B_packed;(void)col_sums;
    (void)w_scales;(void)bias;(void)act_scale;(void)out_scale;(void)act;
    (void)output;(void)out_stride;
}

void i8_pack_weights(const float* weights, int Cout, int K,
                     int8_t* packed, int32_t* col_sums, float* w_scales)
{
    (void)weights;(void)Cout;(void)K;(void)packed;(void)col_sums;(void)w_scales;
}

#endif /* __AVX512VNNI__ */
