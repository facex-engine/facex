/*
 * transformer_ops.c — FP32 ops for EdgeFace-XS Transformer blocks.
 *
 * LayerNorm, GELU, Softmax, L2Normalize, MatMul (FP32).
 * These run in FP32 (not INT8) since Transformer attention
 * involves dynamic×dynamic MatMul which doesn't benefit from INT8.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(FACEX_HAVE_SME) || defined(FACEX_HAVE_ACCELERATE)
#include <stdatomic.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#ifdef __wasm_simd128__
#include "../include/wasm_compat.h"
#endif
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define FACEX_HAVE_NEON 1
#endif

/* ============ LayerNorm ============ */
/* out[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i] */
void layer_norm_fp32(const float* x, int N, int C,
                     const float* gamma, const float* beta,
                     float eps, float* out) {
    for (int n = 0; n < N; n++) {
        const float* row = x + (size_t)n * C;
        float* orow = out + (size_t)n * C;
        int c = 0;

#ifdef __AVX512F__
        /* AVX-512 vectorized mean */
        __m512 vsum512 = _mm512_setzero_ps();
        for (c = 0; c + 16 <= C; c += 16)
            vsum512 = _mm512_add_ps(vsum512, _mm512_loadu_ps(row + c));
        float sum = _mm512_reduce_add_ps(vsum512);
        for (; c < C; c++) sum += row[c];
        float mean = sum / C;

        /* AVX-512 vectorized variance */
        __m512 vmean512 = _mm512_set1_ps(mean);
        __m512 vvar512 = _mm512_setzero_ps();
        for (c = 0; c + 16 <= C; c += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(row + c), vmean512);
            vvar512 = _mm512_fmadd_ps(d, d, vvar512);
        }
        float var_sum = _mm512_reduce_add_ps(vvar512);
        for (; c < C; c++) { float d = row[c] - mean; var_sum += d * d; }
        float inv_std = 1.0f / sqrtf(var_sum / C + eps);

        /* AVX-512 vectorized normalize + affine */
        __m512 vis512 = _mm512_set1_ps(inv_std);
        for (c = 0; c + 16 <= C; c += 16) {
            __m512 v = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(row + c), vmean512), vis512);
            if (gamma) v = _mm512_mul_ps(v, _mm512_loadu_ps(gamma + c));
            if (beta) v = _mm512_add_ps(v, _mm512_loadu_ps(beta + c));
            _mm512_storeu_ps(orow + c, v);
        }
        for (; c < C; c++) {
            float v = (row[c] - mean) * inv_std;
            if (gamma) v *= gamma[c];
            if (beta) v += beta[c];
            orow[c] = v;
        }
#elif defined(__AVX2__)
        c = 0;
        /* Vectorized mean */
        __m256 vsum = _mm256_setzero_ps();
        for (c = 0; c + 8 <= C; c += 8)
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(row + c));
        /* Horizontal sum */
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
        float sum = _mm_cvtss_f32(lo);
        for (; c < C; c++) sum += row[c];
        float mean = sum / C;

        /* Vectorized variance */
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vvar = _mm256_setzero_ps();
        for (c = 0; c + 8 <= C; c += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + c), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
        }
        lo = _mm256_castps256_ps128(vvar);
        hi = _mm256_extractf128_ps(vvar, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
        float var_sum = _mm_cvtss_f32(lo);
        for (; c < C; c++) { float d = row[c] - mean; var_sum += d * d; }
        float inv_std = 1.0f / sqrtf(var_sum / C + eps);

        /* Vectorized normalize + affine */
        __m256 vis = _mm256_set1_ps(inv_std);
        for (c = 0; c + 8 <= C; c += 8) {
            __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + c), vmean), vis);
            if (gamma) v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + c));
            if (beta) v = _mm256_add_ps(v, _mm256_loadu_ps(beta + c));
            _mm256_storeu_ps(orow + c, v);
        }
        for (; c < C; c++) {
            float v = (row[c] - mean) * inv_std;
            if (gamma) v *= gamma[c];
            if (beta) v += beta[c];
            orow[c] = v;
        }
#else
        float sum = 0;
        for (c = 0; c < C; c++) sum += row[c];
        float mean = sum / C;
        float var_sum = 0;
        for (c = 0; c < C; c++) { float d = row[c] - mean; var_sum += d * d; }
        float inv_std = 1.0f / sqrtf(var_sum / C + eps);
        for (c = 0; c < C; c++) {
            orow[c] = (row[c] - mean) * inv_std;
            if (gamma) orow[c] *= gamma[c];
            if (beta) orow[c] += beta[c];
        }
#endif
    }
}

/* ============ GELU (fast approximation) ============ */
/* gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Using fast tanh: tanh(x) ≈ x*(27+x^2)/(27+9*x^2) for |x|<3, clamp otherwise */
static inline float fast_tanh(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* AVX2 vectorized exp(x) for x in [-88, 0] (negative range for erf computation).
 * Uses range reduction: exp(x) = 2^n * exp(r) where n=round(x/ln2), r=x-n*ln2.
 * Polynomial approximation for exp(r) on [-ln2/2, ln2/2]. Max error ~1e-7. */
#ifdef __AVX2__
__m256 _mm256_exp_ps(__m256 x) {
    const __m256 v_ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 v_inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    /* Polynomial coefficients for exp(r) on [-ln2/2, ln2/2] */
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666666f);
    const __m256 c4 = _mm256_set1_ps(0.041666666f);
    const __m256 c5 = _mm256_set1_ps(0.008333333f);

    /* n = round(x / ln2) */
    __m256 t = _mm256_fmadd_ps(x, v_inv_ln2, v_half);
    __m256 n = _mm256_floor_ps(t);
    /* r = x - n * ln2 */
    __m256 r = _mm256_fnmadd_ps(n, v_ln2, x);
    /* exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 (Horner) */
    __m256 p = _mm256_fmadd_ps(c5, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);
    /* Scale by 2^n: add n to the exponent bits */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_slli_epi32(ni, 23); /* shift to exponent position */
    __m256i pi = _mm256_castps_si256(p);
    p = _mm256_castsi256_ps(_mm256_add_epi32(pi, ni));
    /* Clamp: for very negative x, return 0 */
    p = _mm256_max_ps(p, _mm256_setzero_ps());
    return p;
}
#endif

void gelu_fp32(float* x, int n) {
    /* Fast tanh-based GELU with AVX-512 when available */
    int i = 0;
#ifdef __AVX512F__
    const __m512 v_c1 = _mm512_set1_ps(0.7978845608f);
    const __m512 v_c2 = _mm512_set1_ps(0.044715f);
    const __m512 v_half = _mm512_set1_ps(0.5f);
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_neg_one = _mm512_set1_ps(-1.0f);
    const __m512 v_27 = _mm512_set1_ps(27.0f);
    const __m512 v_9 = _mm512_set1_ps(9.0f);
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vx3 = _mm512_mul_ps(_mm512_mul_ps(vx, vx), vx);
        __m512 inner = _mm512_mul_ps(v_c1, _mm512_fmadd_ps(v_c2, vx3, vx));
        __m512 inner2 = _mm512_mul_ps(inner, inner);
        __m512 num = _mm512_mul_ps(inner, _mm512_add_ps(v_27, inner2));
        __m512 den = _mm512_fmadd_ps(v_9, inner2, v_27);
        __m512 t = _mm512_div_ps(num, den);
        t = _mm512_min_ps(t, v_one);
        t = _mm512_max_ps(t, v_neg_one);
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_mul_ps(v_half, vx), _mm512_add_ps(v_one, t)));
    }
#elif defined(__AVX2__)
    const __m256 v_c1 = _mm256_set1_ps(0.7978845608f);
    const __m256 v_c2 = _mm256_set1_ps(0.044715f);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_neg_one = _mm256_set1_ps(-1.0f);
    const __m256 v_27 = _mm256_set1_ps(27.0f);
    const __m256 v_9 = _mm256_set1_ps(9.0f);

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        /* inner = sqrt(2/pi) * (x + 0.044715 * x^3) */
        __m256 vx3 = _mm256_mul_ps(_mm256_mul_ps(vx, vx), vx);
        __m256 inner = _mm256_mul_ps(v_c1, _mm256_fmadd_ps(v_c2, vx3, vx));
        /* tanh(inner) ≈ inner*(27+inner²)/(27+9*inner²) */
        __m256 inner2 = _mm256_mul_ps(inner, inner);
        __m256 num = _mm256_mul_ps(inner, _mm256_add_ps(v_27, inner2));
        __m256 den = _mm256_fmadd_ps(v_9, inner2, v_27);
        __m256 t = _mm256_div_ps(num, den);
        t = _mm256_min_ps(t, v_one);
        t = _mm256_max_ps(t, v_neg_one);

        /* gelu = 0.5 * x * (1 + tanh) */
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(v_half, vx), _mm256_add_ps(v_one, t));
        _mm256_storeu_ps(x + i, result);
    }
#endif
    for (; i < n; i++) {
        float v = x[i];
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        float t2 = inner * (27.0f + inner*inner) / (27.0f + 9.0f * inner*inner);
        if (t2 > 1.0f) t2 = 1.0f; if (t2 < -1.0f) t2 = -1.0f;
        x[i] = 0.5f * v * (1.0f + t2);
    }
}

/* ============ Softmax ============ */
/* softmax along last dim: out[i] = exp(x[i] - max) / sum(exp(x - max)) */
void softmax_fp32(float* x, int N, int C) {
    for (int n = 0; n < N; n++) {
        float* row = x + (size_t)n * C;
        float mx = row[0];
        for (int c = 1; c < C; c++) if (row[c] > mx) mx = row[c];
        float sum = 0;
        for (int c = 0; c < C; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }
        float inv = 1.0f / sum;
        for (int c = 0; c < C; c++) row[c] *= inv;
    }
}

/* ============ Dynamic INT8 MatMul ============ */
/* C[M,N] = A_fp32[M,K] @ W_int8_packed[K,N], with dynamic A quantization.
 * W is pre-packed in c8 format. A is quantized per-tensor at runtime. */
void matmul_dynamic_int8(const float* A_fp32, int M, int K, int N,
                         const void* W_packed, const int32_t* col_sums,
                         const float* w_scales, float* C_fp32)
{
    int K_padded = (K + 7) & ~7;
    int n_elem = M * K;

    /* Static workspace (avoid malloc per call) */
    static int8_t* s_A_int8 = NULL;
    static int32_t* s_C_int32 = NULL;
    static size_t s_A_cap = 0, s_C_cap = 0;
    size_t a_need = (size_t)M * K_padded;
    size_t c_need = (size_t)M * N;
    if (a_need > s_A_cap) { free(s_A_int8); s_A_int8 = (int8_t*)malloc(a_need); s_A_cap = a_need; }
    if (c_need > s_C_cap) { free(s_C_int32); s_C_int32 = (int32_t*)malloc(c_need * 4); s_C_cap = c_need; }
    int8_t* A_int8 = s_A_int8;
    int32_t* C_int32 = s_C_int32;

    /* Dynamic per-tensor quantization: find max|A| then quantize.
     * AVX-512: scan 16 floats per iteration. */
    float a_max = 0;
#ifdef __AVX512F__
    {
        __m512 vmax = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n_elem; i += 16)
            vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(A_fp32 + i)));
        a_max = _mm512_reduce_max_ps(vmax);
        for (; i < n_elem; i++) {
            float v = A_fp32[i] < 0 ? -A_fp32[i] : A_fp32[i];
            if (v > a_max) a_max = v;
        }
    }
#elif defined(__AVX2__)
    {
        __m256 vmax = _mm256_setzero_ps();
        __m256 vmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        int i = 0;
        for (; i + 8 <= n_elem; i += 8)
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(A_fp32 + i), vmask));
        __m128 lo = _mm256_castps256_ps128(vmax), hi = _mm256_extractf128_ps(vmax, 1);
        lo = _mm_max_ps(lo, hi); lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
        a_max = _mm_cvtss_f32(lo);
        for (; i < n_elem; i++) {
            float v = A_fp32[i] < 0 ? -A_fp32[i] : A_fp32[i];
            if (v > a_max) a_max = v;
        }
    }
#else
    for (int i = 0; i < n_elem; i++) {
        float v = A_fp32[i] < 0 ? -A_fp32[i] : A_fp32[i];
        if (v > a_max) a_max = v;
    }
#endif
    float a_scale = a_max / 127.0f;
    if (a_scale < 1e-8f) a_scale = 1e-8f;
    float a_inv = 1.0f / a_scale;

    /* Quantize A: AVX-512 processes 16 floats → 16 int8 per iteration */
#ifdef __AVX512F__
    {
        __m512 vinv = _mm512_set1_ps(a_inv);
        for (int m = 0; m < M; m++) {
            const float* arow = A_fp32 + (size_t)m * K;
            int8_t* orow = A_int8 + (size_t)m * K_padded;
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                __m512 vf = _mm512_loadu_ps(arow + k);
                __m512i vi = _mm512_cvtps_epi32(_mm512_mul_ps(vf, vinv));
                vi = _mm512_max_epi32(vi, _mm512_set1_epi32(-128));
                vi = _mm512_min_epi32(vi, _mm512_set1_epi32(127));
                /* Pack 16 × int32 → 16 × int8 via two-stage packs */
                __m256i lo = _mm512_castsi512_si256(vi);
                __m256i hi = _mm512_extracti64x4_epi64(vi, 1);
                __m256i i16 = _mm256_packs_epi32(lo, hi);
                i16 = _mm256_permute4x64_epi64(i16, 0xD8);
                __m128i i16_lo = _mm256_castsi256_si128(i16);
                __m128i i16_hi = _mm256_extracti128_si256(i16, 1);
                __m128i i8 = _mm_packs_epi16(i16_lo, i16_hi);
                _mm_storeu_si128((__m128i*)(orow + k), i8);
            }
            for (; k + 8 <= K; k += 8) {
                __m256 vf = _mm256_loadu_ps(arow + k);
                __m256i vi = _mm256_cvtps_epi32(_mm256_mul_ps(vf, _mm256_set1_ps(a_inv)));
                vi = _mm256_max_epi32(vi, _mm256_set1_epi32(-128));
                vi = _mm256_min_epi32(vi, _mm256_set1_epi32(127));
                __m128i lo2 = _mm256_castsi256_si128(vi);
                __m128i hi2 = _mm256_extracti128_si256(vi, 1);
                __m128i i16 = _mm_packs_epi32(lo2, hi2);
                __m128i i8 = _mm_packs_epi16(i16, i16);
                *(int64_t*)(orow + k) = _mm_extract_epi64(i8, 0);
            }
            for (; k < K; k++) {
                int q = (int)(arow[k] * a_inv + (arow[k] >= 0 ? 0.5f : -0.5f));
                if (q > 127) q = 127; if (q < -128) q = -128;
                orow[k] = (int8_t)q;
            }
            for (int k2 = K; k2 < K_padded; k2++) orow[k2] = 0;
        }
    }
#elif defined(__AVX2__)
    {
        __m256 v_inv = _mm256_set1_ps(a_inv);
        for (int m = 0; m < M; m++) {
            const float* arow = A_fp32 + (size_t)m * K;
            int8_t* orow = A_int8 + (size_t)m * K_padded;
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 vf = _mm256_loadu_ps(arow + k);
                __m256i vi = _mm256_cvtps_epi32(_mm256_mul_ps(vf, v_inv));
                vi = _mm256_max_epi32(vi, _mm256_set1_epi32(-128));
                vi = _mm256_min_epi32(vi, _mm256_set1_epi32(127));
                __m128i lo = _mm256_castsi256_si128(vi);
                __m128i hi = _mm256_extracti128_si256(vi, 1);
                __m128i i16 = _mm_packs_epi32(lo, hi);
                __m128i i8 = _mm_packs_epi16(i16, i16);
                *(int64_t*)(orow + k) = _mm_extract_epi64(i8, 0);
            }
            for (; k < K; k++) {
                int q = (int)(arow[k] * a_inv + (arow[k] >= 0 ? 0.5f : -0.5f));
                if (q > 127) q = 127; if (q < -128) q = -128;
                orow[k] = (int8_t)q;
            }
            for (int k2 = K; k2 < K_padded; k2++) orow[k2] = 0;
        }
    }
#else
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int q = (int)(A_fp32[m*K+k] * a_inv + (A_fp32[m*K+k] >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127; if (q < -128) q = -128;
            A_int8[m*K_padded+k] = (int8_t)q;
        }
        for (int k = K; k < K_padded; k++) A_int8[m*K_padded+k] = 0;
    }
#endif

    /* 3. INT8 GEMM */
    extern void int8_gemm_4x8c8(const int8_t*, int, int, int,
                                  const void*, int32_t*, const int32_t*);
    int8_gemm_4x8c8(A_int8, M, K_padded, N, W_packed, C_int32, col_sums);

    /* 4. Dequant: AVX-512 with FMA for combined multiply */
#ifdef __AVX512F__
    {
        __m512 v_ascale = _mm512_set1_ps(a_scale);
        for (int m = 0; m < M; m++) {
            int n = 0;
            for (; n + 16 <= N; n += 16) {
                __m512i vi32 = _mm512_loadu_si512((__m512i*)(C_int32 + m*N + n));
                __m512 vf = _mm512_cvtepi32_ps(vi32);
                __m512 vws = _mm512_loadu_ps(w_scales + n);
                vf = _mm512_mul_ps(_mm512_mul_ps(vf, v_ascale), vws);
                _mm512_storeu_ps(C_fp32 + m*N + n, vf);
            }
            for (; n + 8 <= N; n += 8) {
                __m256i vi32 = _mm256_loadu_si256((__m256i*)(C_int32 + m*N + n));
                __m256 vf = _mm256_cvtepi32_ps(vi32);
                __m256 vws = _mm256_loadu_ps(w_scales + n);
                vf = _mm256_mul_ps(_mm256_mul_ps(vf, _mm256_set1_ps(a_scale)), vws);
                _mm256_storeu_ps(C_fp32 + m*N + n, vf);
            }
            for (; n < N; n++)
                C_fp32[m*N+n] = (float)C_int32[m*N+n] * a_scale * w_scales[n];
        }
    }
#elif defined(__AVX2__)
    {
        __m256 v_ascale = _mm256_set1_ps(a_scale);
        for (int m = 0; m < M; m++) {
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                __m256i vi32 = _mm256_loadu_si256((__m256i*)(C_int32 + m*N + n));
                __m256 vf = _mm256_cvtepi32_ps(vi32);
                __m256 vws = _mm256_loadu_ps(w_scales + n);
                vf = _mm256_mul_ps(_mm256_mul_ps(vf, v_ascale), vws);
                _mm256_storeu_ps(C_fp32 + m*N + n, vf);
            }
            for (; n < N; n++)
                C_fp32[m*N+n] = (float)C_int32[m*N+n] * a_scale * w_scales[n];
        }
    }
#else
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C_fp32[m*N+n] = (float)C_int32[m*N+n] * a_scale * w_scales[n];
#endif
}

/* matmul_dynamic_int8_gelu not needed — GELU already vectorized and fast */

/* ============ MatMul FP32 (AVX2 register-tiled, auto-INT8 for large K) ============ */
/* C[M,N] = A[M,K] @ B[K,N]
 * Microkernel: MR=4 rows × NR=16 columns (4×2 YMM accumulators = 8 regs)
 * For K >= 32 and N >= 8: automatically uses dynamic INT8 quantization
 * for 2-3x throughput improvement via vpmaddubsw. */
void matmul_fp32(const float* A, const float* B, float* C,
                 int M, int K, int N) {
#ifdef __AVX2__
    int m = 0;
    /* Main loop: MR=4 rows at a time */
    for (; m + 4 <= M; m += 4) {
        int n = 0;
        for (; n + 16 <= N; n += 16) {
            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 b0 = _mm256_loadu_ps(B + (size_t)k * N + n);
                __m256 b1 = _mm256_loadu_ps(B + (size_t)k * N + n + 8);
                __m256 a0 = _mm256_broadcast_ss(&A[(m+0)*K+k]);
                __m256 a1 = _mm256_broadcast_ss(&A[(m+1)*K+k]);
                __m256 a2 = _mm256_broadcast_ss(&A[(m+2)*K+k]);
                __m256 a3 = _mm256_broadcast_ss(&A[(m+3)*K+k]);
                c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
                c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
                c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
                c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
            }
            _mm256_storeu_ps(C+(m+0)*N+n, c00); _mm256_storeu_ps(C+(m+0)*N+n+8, c01);
            _mm256_storeu_ps(C+(m+1)*N+n, c10); _mm256_storeu_ps(C+(m+1)*N+n+8, c11);
            _mm256_storeu_ps(C+(m+2)*N+n, c20); _mm256_storeu_ps(C+(m+2)*N+n+8, c21);
            _mm256_storeu_ps(C+(m+3)*N+n, c30); _mm256_storeu_ps(C+(m+3)*N+n+8, c31);
        }
        for (; n + 8 <= N; n += 8) {
            __m256 c0=_mm256_setzero_ps(), c1=_mm256_setzero_ps();
            __m256 c2=_mm256_setzero_ps(), c3=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 b0 = _mm256_loadu_ps(B + (size_t)k * N + n);
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+0)*K+k]), b0, c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+1)*K+k]), b0, c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+2)*K+k]), b0, c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+3)*K+k]), b0, c3);
            }
            _mm256_storeu_ps(C+(m+0)*N+n, c0); _mm256_storeu_ps(C+(m+1)*N+n, c1);
            _mm256_storeu_ps(C+(m+2)*N+n, c2); _mm256_storeu_ps(C+(m+3)*N+n, c3);
        }
        for (; n < N; n++) {
            float s0=0, s1=0, s2=0, s3=0;
            for (int k = 0; k < K; k++) {
                float bv = B[(size_t)k*N+n];
                s0 += A[(m+0)*K+k] * bv; s1 += A[(m+1)*K+k] * bv;
                s2 += A[(m+2)*K+k] * bv; s3 += A[(m+3)*K+k] * bv;
            }
            C[(m+0)*N+n]=s0; C[(m+1)*N+n]=s1; C[(m+2)*N+n]=s2; C[(m+3)*N+n]=s3;
        }
    }
    /* M remainder: 1 row at a time with register accumulators (no memset needed) */
    for (; m < M; m++) {
        const float* a_row = A + (size_t)m * K;
        float* c_row = C + (size_t)m * N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 vc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++)
                vc = _mm256_fmadd_ps(_mm256_broadcast_ss(&a_row[k]),
                                     _mm256_loadu_ps(B + (size_t)k * N + n), vc);
            _mm256_storeu_ps(c_row + n, vc);
        }
        for (; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += a_row[k] * B[(size_t)k*N+n];
            c_row[n] = sum;
        }
    }
#else
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float a = A[(size_t)m * K + k];
            for (int n = 0; n < N; n++)
                C[(size_t)m * N + n] += a * B[(size_t)k * N + n];
        }
#endif
}

/* ============ FP32 B-packing: [K,N] → [ceil(N/8), K, 8] panel format ============ */
/* Pack weight matrix B[K,N] into column-panel format for cache-friendly GEMM.
 * Each panel of 8 columns is stored contiguously along K dimension.
 * This gives sequential 32-byte (1 cache line) loads in the inner K loop. */
void pack_b_fp32(const float* B, int K, int N, float* packed) {
#if defined(__AVX512F__)
    int NR = 16;
#elif defined(__AVX2__)
    int NR = 8;
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
    int n_panels = (N + NR - 1) / NR;
    for (int p = 0; p < n_panels; p++) {
        int n_start = p * NR;
        int nr = (n_start + NR <= N) ? NR : (N - n_start);
        float* dst = packed + (size_t)p * K * NR;
        for (int k = 0; k < K; k++) {
            const float* src = B + (size_t)k * N + n_start;
            int j = 0;
            for (; j < nr; j++) dst[k * NR + j] = src[j];
            for (; j < NR; j++) dst[k * NR + j] = 0; /* zero-pad last panel */
        }
    }
#else
    /* Scalar: just copy — matmul_fp32 expects standard [K,N] layout */
    memcpy(packed, B, (size_t)K * N * sizeof(float));
#endif
}

int packed_b_fp32_size(int K, int N) {
#ifdef __AVX512F__
    return ((N + 15) / 16) * K * 16;
#elif defined(__AVX2__)
    return ((N + 7) / 8) * K * 8;
#else
    return K * N;
#endif
}

/* ============ Threaded tile worker for packed GEMM ============ */
#include "threadpool.h"
typedef struct { const float* A; const float* B; float* C; int K,N,n_panels; } PGemmCtx;
#ifdef __AVX2__
static void _pgemm_worker(void* ctx_, int start, int end) {
    PGemmCtx* g = (PGemmCtx*)ctx_;
    int K_=g->K, N_=g->N, NR_=8;
    for (int mb = start; mb < end; mb += 4) {
        if (mb + 4 > end) break;
        int p = 0;
        for (; p + 2 <= g->n_panels; p += 2) {
            int n = p * NR_;
            const float* bp0 = g->B + (size_t)p * K_ * NR_;
            const float* bp1 = bp0 + (size_t)K_ * NR_;
            __m256 c00=_mm256_setzero_ps(),c01=_mm256_setzero_ps();
            __m256 c10=_mm256_setzero_ps(),c11=_mm256_setzero_ps();
            __m256 c20=_mm256_setzero_ps(),c21=_mm256_setzero_ps();
            __m256 c30=_mm256_setzero_ps(),c31=_mm256_setzero_ps();
            for (int k=0;k<K_;k++){
                __m256 b0=_mm256_load_ps(bp0+k*NR_),b1=_mm256_load_ps(bp1+k*NR_);
                __m256 a0=_mm256_broadcast_ss(&g->A[(mb+0)*K_+k]);
                __m256 a1=_mm256_broadcast_ss(&g->A[(mb+1)*K_+k]);
                __m256 a2=_mm256_broadcast_ss(&g->A[(mb+2)*K_+k]);
                __m256 a3=_mm256_broadcast_ss(&g->A[(mb+3)*K_+k]);
                c00=_mm256_fmadd_ps(a0,b0,c00);c01=_mm256_fmadd_ps(a0,b1,c01);
                c10=_mm256_fmadd_ps(a1,b0,c10);c11=_mm256_fmadd_ps(a1,b1,c11);
                c20=_mm256_fmadd_ps(a2,b0,c20);c21=_mm256_fmadd_ps(a2,b1,c21);
                c30=_mm256_fmadd_ps(a3,b0,c30);c31=_mm256_fmadd_ps(a3,b1,c31);
            }
            if(n+16<=N_){
                _mm256_storeu_ps(g->C+(mb+0)*N_+n,c00);_mm256_storeu_ps(g->C+(mb+0)*N_+n+8,c01);
                _mm256_storeu_ps(g->C+(mb+1)*N_+n,c10);_mm256_storeu_ps(g->C+(mb+1)*N_+n+8,c11);
                _mm256_storeu_ps(g->C+(mb+2)*N_+n,c20);_mm256_storeu_ps(g->C+(mb+2)*N_+n+8,c21);
                _mm256_storeu_ps(g->C+(mb+3)*N_+n,c30);_mm256_storeu_ps(g->C+(mb+3)*N_+n+8,c31);
            } else {
                _mm256_storeu_ps(g->C+(mb+0)*N_+n,c00);_mm256_storeu_ps(g->C+(mb+1)*N_+n,c10);
                _mm256_storeu_ps(g->C+(mb+2)*N_+n,c20);_mm256_storeu_ps(g->C+(mb+3)*N_+n,c30);
                int nr=N_-n-8;if(nr>0){float t[4][8];
                _mm256_storeu_ps(t[0],c01);_mm256_storeu_ps(t[1],c11);
                _mm256_storeu_ps(t[2],c21);_mm256_storeu_ps(t[3],c31);
                for(int i=0;i<4;i++)for(int j=0;j<nr;j++)g->C[(mb+i)*N_+n+8+j]=t[i][j];}
            }
        }
        for(;p<g->n_panels;p++){
            int n=p*NR_;
            const float* bp=g->B+(size_t)p*K_*NR_;
            __m256 c0=_mm256_setzero_ps(),c1=_mm256_setzero_ps();
            __m256 c2=_mm256_setzero_ps(),c3=_mm256_setzero_ps();
            for(int k=0;k<K_;k++){
                __m256 b=_mm256_load_ps(bp+k*NR_);
                c0=_mm256_fmadd_ps(_mm256_broadcast_ss(&g->A[(mb+0)*K_+k]),b,c0);
                c1=_mm256_fmadd_ps(_mm256_broadcast_ss(&g->A[(mb+1)*K_+k]),b,c1);
                c2=_mm256_fmadd_ps(_mm256_broadcast_ss(&g->A[(mb+2)*K_+k]),b,c2);
                c3=_mm256_fmadd_ps(_mm256_broadcast_ss(&g->A[(mb+3)*K_+k]),b,c3);
            }
            if(n+NR_<=N_){
                _mm256_storeu_ps(g->C+(mb+0)*N_+n,c0);_mm256_storeu_ps(g->C+(mb+1)*N_+n,c1);
                _mm256_storeu_ps(g->C+(mb+2)*N_+n,c2);_mm256_storeu_ps(g->C+(mb+3)*N_+n,c3);
            } else {float t[4][8];
                _mm256_storeu_ps(t[0],c0);_mm256_storeu_ps(t[1],c1);
                _mm256_storeu_ps(t[2],c2);_mm256_storeu_ps(t[3],c3);
                int nr=N_-n;for(int i=0;i<4;i++)for(int j=0;j<nr;j++)g->C[(mb+i)*N_+n+j]=t[i][j];}
        }
    }
}
#endif

/* ============ MatMul with pre-packed B ============ */
/* AVX-512 path: NR=16 (one ZMM = 16 floats), masked stores for remainder.
 * AVX2 fallback: NR=8 (one YMM = 8 floats). */
void matmul_fp32_packed(const float* A, const float* B_packed, float* C,
                        int M, int K, int N) {
#ifdef FACEX_HAVE_ACCELERATE
    /* Apple Accelerate.framework cblas_sgemm path. Runs on AMX, typically
     * 2-3× our NEON throughput at sizes that matter. The wrapper unpacks
     * the column-panel B into row-major and dispatches; for tiny M*K*N it
     * returns -1 so we fall through to the in-tree NEON kernel. */
    {
        extern int matmul_fp32_packed_accelerate(const float*, const float*,
                                                  float*, int, int, int);
        extern int facex_accelerate_validate(void);
        extern int facex_accelerate_enabled(void);
        static _Atomic int acc_state = 0; /* 0 = unchecked, 1 = ok, -1 = bad */
        int s = atomic_load_explicit(&acc_state, memory_order_acquire);
        if (s == 0) {
            s = (facex_accelerate_validate() == 0) ? 1 : -1;
            atomic_store_explicit(&acc_state, s, memory_order_release);
        }
        if (s == 1 && facex_accelerate_enabled() &&
            matmul_fp32_packed_accelerate(A, B_packed, C, M, K, N) == 0)
            return;
    }
#endif
#ifdef FACEX_HAVE_SME
    /* SME dispatch (Apple M4+ / future ARMv9 with FEAT_SME).
     * On first call we run a tiny SME-vs-scalar self-check; if SME is
     * present and the check passes, every subsequent call uses it.
     * The kernel itself returns -1 for shapes it refuses (M too small,
     * K too large) so the existing arch path below acts as fallback. */
    {
        extern int facex_has_sme(void);
        extern void facex_disable_sme(void);
        extern int facex_sme_validate(void);
        extern int matmul_fp32_packed_sme(const float*, const float*,
                                          float*, int, int, int);
        /* States: 0 = unchecked, 1 = enabled, -1 = disabled */
        static _Atomic int sme_state = 0;
        int s = atomic_load_explicit(&sme_state, memory_order_acquire);
        if (s == 0) {
            int ok = facex_has_sme() && (facex_sme_validate() == 0);
            s = ok ? 1 : -1;
            if (!ok) facex_disable_sme();
            atomic_store_explicit(&sme_state, s, memory_order_release);
        }
        if (s == 1 && matmul_fp32_packed_sme(A, B_packed, C, M, K, N) == 0)
            return;
    }
#endif
#ifdef __AVX512F__
    /* AVX-512: NR=16, MR=4. 4 ZMM accumulators + 1 B + 4 A broadcasts = 9 regs (32 available). */
    int NR = 16;
    int n_panels = (N + NR - 1) / NR;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            __m512 c0=_mm512_setzero_ps(), c1=_mm512_setzero_ps();
            __m512 c2=_mm512_setzero_ps(), c3=_mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m512 b = _mm512_loadu_ps(bp + k * NR);
                c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[(m+0)*K+k]), b, c0);
                c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[(m+1)*K+k]), b, c1);
                c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[(m+2)*K+k]), b, c2);
                c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[(m+3)*K+k]), b, c3);
            }
            if (n + NR <= N) {
                _mm512_storeu_ps(C+(m+0)*N+n, c0); _mm512_storeu_ps(C+(m+1)*N+n, c1);
                _mm512_storeu_ps(C+(m+2)*N+n, c2); _mm512_storeu_ps(C+(m+3)*N+n, c3);
            } else {
                __mmask16 mask = ((uint32_t)1 << (N - n)) - 1;
                _mm512_mask_storeu_ps(C+(m+0)*N+n, mask, c0);
                _mm512_mask_storeu_ps(C+(m+1)*N+n, mask, c1);
                _mm512_mask_storeu_ps(C+(m+2)*N+n, mask, c2);
                _mm512_mask_storeu_ps(C+(m+3)*N+n, mask, c3);
            }
        }
    }
    for (; m < M; m++) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            __m512 c = _mm512_setzero_ps();
            for (int k = 0; k < K; k++)
                c = _mm512_fmadd_ps(_mm512_set1_ps(A[m*K+k]), _mm512_loadu_ps(bp+k*NR), c);
            if (n + NR <= N) _mm512_storeu_ps(C+m*N+n, c);
            else _mm512_mask_storeu_ps(C+m*N+n, ((uint32_t)1<<(N-n))-1, c);
        }
    }
    return;
#elif defined(__AVX2__)
    int NR = 8;
    int n_panels = (N + NR - 1) / NR;

    /* Threaded path for large M */
    if (0 && M >= 128 && tp_num_threads() > 1) { /* disabled — threading at block level instead */
        PGemmCtx gctx = {A, B_packed, C, K, N, n_panels};
        int m4 = (M / 4) * 4;
        tp_parallel_for(_pgemm_worker, &gctx, m4, 16); /* grain=16 rows (4 tiles) */
        /* Handle M remainder sequentially */
        if (m4 >= M) return;
        A += (size_t)m4 * K;
        C += (size_t)m4 * N;
        M -= m4;
    }

    int m = 0;

    for (; m + 4 <= M; m += 4) {
        int p = 0;
        /* Process 2 panels at a time (16 columns) when possible */
        for (; p + 2 <= n_panels; p += 2) {
            int n = p * NR;
            const float* bp0 = B_packed + (size_t)p * K * NR;
            const float* bp1 = bp0 + (size_t)K * NR;
            __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps();
            __m256 c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
            __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps();
            __m256 c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
            { int k = 0;
            for (; k + 2 <= K; k += 2) {
                __m256 b0a = _mm256_load_ps(bp0 + k * NR);
                __m256 b1a = _mm256_load_ps(bp1 + k * NR);
                __m256 b0b = _mm256_load_ps(bp0 + (k+1) * NR);
                __m256 b1b = _mm256_load_ps(bp1 + (k+1) * NR);
                __m256 a0a = _mm256_broadcast_ss(&A[(m+0)*K+k]);
                __m256 a1a = _mm256_broadcast_ss(&A[(m+1)*K+k]);
                __m256 a2a = _mm256_broadcast_ss(&A[(m+2)*K+k]);
                __m256 a3a = _mm256_broadcast_ss(&A[(m+3)*K+k]);
                c00 = _mm256_fmadd_ps(a0a,b0a,c00); c01 = _mm256_fmadd_ps(a0a,b1a,c01);
                c10 = _mm256_fmadd_ps(a1a,b0a,c10); c11 = _mm256_fmadd_ps(a1a,b1a,c11);
                c20 = _mm256_fmadd_ps(a2a,b0a,c20); c21 = _mm256_fmadd_ps(a2a,b1a,c21);
                c30 = _mm256_fmadd_ps(a3a,b0a,c30); c31 = _mm256_fmadd_ps(a3a,b1a,c31);
                __m256 a0b = _mm256_broadcast_ss(&A[(m+0)*K+k+1]);
                __m256 a1b = _mm256_broadcast_ss(&A[(m+1)*K+k+1]);
                __m256 a2b = _mm256_broadcast_ss(&A[(m+2)*K+k+1]);
                __m256 a3b = _mm256_broadcast_ss(&A[(m+3)*K+k+1]);
                c00 = _mm256_fmadd_ps(a0b,b0b,c00); c01 = _mm256_fmadd_ps(a0b,b1b,c01);
                c10 = _mm256_fmadd_ps(a1b,b0b,c10); c11 = _mm256_fmadd_ps(a1b,b1b,c11);
                c20 = _mm256_fmadd_ps(a2b,b0b,c20); c21 = _mm256_fmadd_ps(a2b,b1b,c21);
                c30 = _mm256_fmadd_ps(a3b,b0b,c30); c31 = _mm256_fmadd_ps(a3b,b1b,c31);
            }
            for (; k < K; k++) {
                __m256 b0 = _mm256_load_ps(bp0 + k * NR);
                __m256 b1 = _mm256_load_ps(bp1 + k * NR);
                __m256 a0 = _mm256_broadcast_ss(&A[(m+0)*K+k]);
                __m256 a1 = _mm256_broadcast_ss(&A[(m+1)*K+k]);
                __m256 a2 = _mm256_broadcast_ss(&A[(m+2)*K+k]);
                __m256 a3 = _mm256_broadcast_ss(&A[(m+3)*K+k]);
                c00 = _mm256_fmadd_ps(a0,b0,c00); c01 = _mm256_fmadd_ps(a0,b1,c01);
                c10 = _mm256_fmadd_ps(a1,b0,c10); c11 = _mm256_fmadd_ps(a1,b1,c11);
                c20 = _mm256_fmadd_ps(a2,b0,c20); c21 = _mm256_fmadd_ps(a2,b1,c21);
                c30 = _mm256_fmadd_ps(a3,b0,c30); c31 = _mm256_fmadd_ps(a3,b1,c31);
            } }
            if (n + 16 <= N) {
                _mm256_storeu_ps(C+(m+0)*N+n,c00); _mm256_storeu_ps(C+(m+0)*N+n+8,c01);
                _mm256_storeu_ps(C+(m+1)*N+n,c10); _mm256_storeu_ps(C+(m+1)*N+n+8,c11);
                _mm256_storeu_ps(C+(m+2)*N+n,c20); _mm256_storeu_ps(C+(m+2)*N+n+8,c21);
                _mm256_storeu_ps(C+(m+3)*N+n,c30); _mm256_storeu_ps(C+(m+3)*N+n+8,c31);
            } else {
                /* Handle boundary: first panel always full, second may be partial */
                _mm256_storeu_ps(C+(m+0)*N+n,c00); _mm256_storeu_ps(C+(m+1)*N+n,c10);
                _mm256_storeu_ps(C+(m+2)*N+n,c20); _mm256_storeu_ps(C+(m+3)*N+n,c30);
                int nr = N - n - 8;
                if (nr > 0) {
                    float t[4][8];
                    _mm256_storeu_ps(t[0],c01); _mm256_storeu_ps(t[1],c11);
                    _mm256_storeu_ps(t[2],c21); _mm256_storeu_ps(t[3],c31);
                    for (int i=0;i<4;i++) for(int j=0;j<nr;j++) C[(m+i)*N+n+8+j]=t[i][j];
                }
            }
        }
        /* Single panel remainder */
        for (; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            __m256 c0=_mm256_setzero_ps(),c1=_mm256_setzero_ps();
            __m256 c2=_mm256_setzero_ps(),c3=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 b = _mm256_load_ps(bp + k * NR);
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+0)*K+k]),b,c0);
                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+1)*K+k]),b,c1);
                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+2)*K+k]),b,c2);
                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+3)*K+k]),b,c3);
            }
            if (n + NR <= N) {
                _mm256_storeu_ps(C+(m+0)*N+n,c0); _mm256_storeu_ps(C+(m+1)*N+n,c1);
                _mm256_storeu_ps(C+(m+2)*N+n,c2); _mm256_storeu_ps(C+(m+3)*N+n,c3);
            } else {
                float t[4][8];
                _mm256_storeu_ps(t[0],c0); _mm256_storeu_ps(t[1],c1);
                _mm256_storeu_ps(t[2],c2); _mm256_storeu_ps(t[3],c3);
                int nr = N - n;
                for (int i=0;i<4;i++) for(int j=0;j<nr;j++) C[(m+i)*N+n+j]=t[i][j];
            }
        }
    }
    /* MR=1 remainder */
    for (; m < M; m++) {
        const float* a_row = A + (size_t)m * K;
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            __m256 c0 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++)
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&a_row[k]),
                                     _mm256_load_ps(bp + k * NR), c0);
            if (n + NR <= N) {
                _mm256_storeu_ps(C + (size_t)m*N+n, c0);
            } else {
                float t[8]; _mm256_storeu_ps(t, c0);
                for (int j = 0; j < N - n; j++) C[(size_t)m*N+n+j] = t[j];
            }
        }
    }
#elif defined(FACEX_HAVE_NEON)
    /* AArch64 NEON: NR=8 (= 2× float32x4_t), MR=4 row tile.
     * Mirrors the AVX2 layout. B is column-panel [ceil(N/8), K, 8]. */
    const int NR = 8;
    int n_panels = (N + NR - 1) / NR;
    int m = 0;

    for (; m + 4 <= M; m += 4) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            for (int k = 0; k < K; k++) {
                float32x4_t b0 = vld1q_f32(bp + k * NR);
                float32x4_t b1 = vld1q_f32(bp + k * NR + 4);
                float32x4_t a0 = vdupq_n_f32(A[(m + 0) * K + k]);
                float32x4_t a1 = vdupq_n_f32(A[(m + 1) * K + k]);
                float32x4_t a2 = vdupq_n_f32(A[(m + 2) * K + k]);
                float32x4_t a3 = vdupq_n_f32(A[(m + 3) * K + k]);
                c00 = vfmaq_f32(c00, a0, b0); c01 = vfmaq_f32(c01, a0, b1);
                c10 = vfmaq_f32(c10, a1, b0); c11 = vfmaq_f32(c11, a1, b1);
                c20 = vfmaq_f32(c20, a2, b0); c21 = vfmaq_f32(c21, a2, b1);
                c30 = vfmaq_f32(c30, a3, b0); c31 = vfmaq_f32(c31, a3, b1);
            }
            if (n + NR <= N) {
                vst1q_f32(C + (m+0)*N + n,     c00); vst1q_f32(C + (m+0)*N + n + 4, c01);
                vst1q_f32(C + (m+1)*N + n,     c10); vst1q_f32(C + (m+1)*N + n + 4, c11);
                vst1q_f32(C + (m+2)*N + n,     c20); vst1q_f32(C + (m+2)*N + n + 4, c21);
                vst1q_f32(C + (m+3)*N + n,     c30); vst1q_f32(C + (m+3)*N + n + 4, c31);
            } else {
                /* Partial last panel — scalarize the tail store. */
                float t[4][8];
                vst1q_f32(t[0],     c00); vst1q_f32(t[0] + 4, c01);
                vst1q_f32(t[1],     c10); vst1q_f32(t[1] + 4, c11);
                vst1q_f32(t[2],     c20); vst1q_f32(t[2] + 4, c21);
                vst1q_f32(t[3],     c30); vst1q_f32(t[3] + 4, c31);
                int nr = N - n;
                for (int r = 0; r < 4; r++)
                    for (int j = 0; j < nr; j++)
                        C[(m + r) * N + n + j] = t[r][j];
            }
        }
    }
    /* M tail — single-row NEON */
    for (; m < M; m++) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
            for (int k = 0; k < K; k++) {
                float32x4_t a = vdupq_n_f32(A[m * K + k]);
                c0 = vfmaq_f32(c0, a, vld1q_f32(bp + k * NR));
                c1 = vfmaq_f32(c1, a, vld1q_f32(bp + k * NR + 4));
            }
            if (n + NR <= N) {
                vst1q_f32(C + m * N + n,     c0);
                vst1q_f32(C + m * N + n + 4, c1);
            } else {
                float t[8];
                vst1q_f32(t,     c0); vst1q_f32(t + 4, c1);
                int nr = N - n;
                for (int j = 0; j < nr; j++) C[m * N + n + j] = t[j];
            }
        }
    }
#else
    /* Scalar fallback: B is in column-panel format [ceil(N/NR), K, NR], NR=8. */
    {
        const int NR = 8;
        int n_panels = (N + NR - 1) / NR;
        memset(C, 0, (size_t)M * N * sizeof(float));
        for (int m = 0; m < M; m++) {
            for (int p = 0; p < n_panels; p++) {
                int n_base = p * NR;
                int nr = (n_base + NR <= N) ? NR : (N - n_base);
                const float* bp = B_packed + (size_t)p * K * NR;
                for (int k = 0; k < K; k++) {
                    float a = A[(size_t)m * K + k];
                    for (int j = 0; j < nr; j++)
                        C[(size_t)m * N + n_base + j] += a * bp[k * NR + j];
                }
            }
        }
    }
#endif
}

/* ============ Packed MatMul + fused bias ============ */
/* Same as matmul_fp32_packed but adds bias[n] to each output element during store.
 * Eliminates separate bias addition pass. */
void matmul_fp32_packed_bias(const float* A, const float* B_packed, const float* bias,
                             float* C, int M, int K, int N) {
#ifdef __AVX2__
    int NR = 8;
    int n_panels = (N + NR - 1) / NR;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        int p = 0;
        for (; p + 2 <= n_panels; p += 2) {
            int n = p * NR;
            const float* bp0 = B_packed + (size_t)p * K * NR;
            const float* bp1 = bp0 + (size_t)K * NR;
            __m256 c00=_mm256_setzero_ps(),c01=_mm256_setzero_ps();
            __m256 c10=_mm256_setzero_ps(),c11=_mm256_setzero_ps();
            __m256 c20=_mm256_setzero_ps(),c21=_mm256_setzero_ps();
            __m256 c30=_mm256_setzero_ps(),c31=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 b0=_mm256_load_ps(bp0+k*NR), b1=_mm256_load_ps(bp1+k*NR);
                __m256 a0=_mm256_broadcast_ss(&A[(m+0)*K+k]);
                __m256 a1=_mm256_broadcast_ss(&A[(m+1)*K+k]);
                __m256 a2=_mm256_broadcast_ss(&A[(m+2)*K+k]);
                __m256 a3=_mm256_broadcast_ss(&A[(m+3)*K+k]);
                c00=_mm256_fmadd_ps(a0,b0,c00); c01=_mm256_fmadd_ps(a0,b1,c01);
                c10=_mm256_fmadd_ps(a1,b0,c10); c11=_mm256_fmadd_ps(a1,b1,c11);
                c20=_mm256_fmadd_ps(a2,b0,c20); c21=_mm256_fmadd_ps(a2,b1,c21);
                c30=_mm256_fmadd_ps(a3,b0,c30); c31=_mm256_fmadd_ps(a3,b1,c31);
            }
            /* Add bias during store */
            __m256 bb0 = _mm256_loadu_ps(bias+n), bb1 = _mm256_loadu_ps(bias+n+8);
            c00=_mm256_add_ps(c00,bb0); c01=_mm256_add_ps(c01,bb1);
            c10=_mm256_add_ps(c10,bb0); c11=_mm256_add_ps(c11,bb1);
            c20=_mm256_add_ps(c20,bb0); c21=_mm256_add_ps(c21,bb1);
            c30=_mm256_add_ps(c30,bb0); c31=_mm256_add_ps(c31,bb1);
            if (n+16<=N) {
                _mm256_storeu_ps(C+(m+0)*N+n,c00); _mm256_storeu_ps(C+(m+0)*N+n+8,c01);
                _mm256_storeu_ps(C+(m+1)*N+n,c10); _mm256_storeu_ps(C+(m+1)*N+n+8,c11);
                _mm256_storeu_ps(C+(m+2)*N+n,c20); _mm256_storeu_ps(C+(m+2)*N+n+8,c21);
                _mm256_storeu_ps(C+(m+3)*N+n,c30); _mm256_storeu_ps(C+(m+3)*N+n+8,c31);
            } else {
                _mm256_storeu_ps(C+(m+0)*N+n,c00);_mm256_storeu_ps(C+(m+1)*N+n,c10);
                _mm256_storeu_ps(C+(m+2)*N+n,c20);_mm256_storeu_ps(C+(m+3)*N+n,c30);
                int nr=N-n-8; if(nr>0){float t[4][8];
                _mm256_storeu_ps(t[0],c01);_mm256_storeu_ps(t[1],c11);
                _mm256_storeu_ps(t[2],c21);_mm256_storeu_ps(t[3],c31);
                for(int i=0;i<4;i++)for(int j=0;j<nr;j++)C[(m+i)*N+n+8+j]=t[i][j];}
            }
        }
        for (; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            __m256 c0=_mm256_setzero_ps(),c1=_mm256_setzero_ps();
            __m256 c2=_mm256_setzero_ps(),c3=_mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 b=_mm256_load_ps(bp+k*NR);
                c0=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+0)*K+k]),b,c0);
                c1=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+1)*K+k]),b,c1);
                c2=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+2)*K+k]),b,c2);
                c3=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+3)*K+k]),b,c3);
            }
            __m256 bb=_mm256_loadu_ps(bias+n);
            c0=_mm256_add_ps(c0,bb);c1=_mm256_add_ps(c1,bb);
            c2=_mm256_add_ps(c2,bb);c3=_mm256_add_ps(c3,bb);
            if(n+NR<=N){
                _mm256_storeu_ps(C+(m+0)*N+n,c0);_mm256_storeu_ps(C+(m+1)*N+n,c1);
                _mm256_storeu_ps(C+(m+2)*N+n,c2);_mm256_storeu_ps(C+(m+3)*N+n,c3);
            } else {float t[4][8];
                _mm256_storeu_ps(t[0],c0);_mm256_storeu_ps(t[1],c1);
                _mm256_storeu_ps(t[2],c2);_mm256_storeu_ps(t[3],c3);
                int nr=N-n;for(int i=0;i<4;i++)for(int j=0;j<nr;j++)C[(m+i)*N+n+j]=t[i][j];}
        }
    }
    for (; m < M; m++) {
        const float* a_row = A + (size_t)m * K;
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            __m256 c0 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++)
                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&a_row[k]),_mm256_load_ps(bp+k*NR),c0);
            c0=_mm256_add_ps(c0,_mm256_loadu_ps(bias+n));
            if(n+NR<=N){_mm256_storeu_ps(C+(size_t)m*N+n,c0);}
            else{float t[8];_mm256_storeu_ps(t,c0);for(int j=0;j<N-n;j++)C[(size_t)m*N+n+j]=t[j];}
        }
    }
#elif defined(FACEX_HAVE_NEON)
    /* AArch64 NEON: NR=8, MR=4, with bias added at store time. */
    const int NR = 8;
    int n_panels = (N + NR - 1) / NR;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            for (int k = 0; k < K; k++) {
                float32x4_t b0 = vld1q_f32(bp + k * NR);
                float32x4_t b1 = vld1q_f32(bp + k * NR + 4);
                float32x4_t a0 = vdupq_n_f32(A[(m + 0) * K + k]);
                float32x4_t a1 = vdupq_n_f32(A[(m + 1) * K + k]);
                float32x4_t a2 = vdupq_n_f32(A[(m + 2) * K + k]);
                float32x4_t a3 = vdupq_n_f32(A[(m + 3) * K + k]);
                c00 = vfmaq_f32(c00, a0, b0); c01 = vfmaq_f32(c01, a0, b1);
                c10 = vfmaq_f32(c10, a1, b0); c11 = vfmaq_f32(c11, a1, b1);
                c20 = vfmaq_f32(c20, a2, b0); c21 = vfmaq_f32(c21, a2, b1);
                c30 = vfmaq_f32(c30, a3, b0); c31 = vfmaq_f32(c31, a3, b1);
            }
            float32x4_t bb0 = bias ? vld1q_f32(bias + n)     : vdupq_n_f32(0);
            float32x4_t bb1 = bias ? vld1q_f32(bias + n + 4) : vdupq_n_f32(0);
            c00 = vaddq_f32(c00, bb0); c01 = vaddq_f32(c01, bb1);
            c10 = vaddq_f32(c10, bb0); c11 = vaddq_f32(c11, bb1);
            c20 = vaddq_f32(c20, bb0); c21 = vaddq_f32(c21, bb1);
            c30 = vaddq_f32(c30, bb0); c31 = vaddq_f32(c31, bb1);
            if (n + NR <= N) {
                vst1q_f32(C + (m+0)*N + n,     c00); vst1q_f32(C + (m+0)*N + n + 4, c01);
                vst1q_f32(C + (m+1)*N + n,     c10); vst1q_f32(C + (m+1)*N + n + 4, c11);
                vst1q_f32(C + (m+2)*N + n,     c20); vst1q_f32(C + (m+2)*N + n + 4, c21);
                vst1q_f32(C + (m+3)*N + n,     c30); vst1q_f32(C + (m+3)*N + n + 4, c31);
            } else {
                float t[4][8];
                vst1q_f32(t[0],     c00); vst1q_f32(t[0] + 4, c01);
                vst1q_f32(t[1],     c10); vst1q_f32(t[1] + 4, c11);
                vst1q_f32(t[2],     c20); vst1q_f32(t[2] + 4, c21);
                vst1q_f32(t[3],     c30); vst1q_f32(t[3] + 4, c31);
                int nr = N - n;
                for (int r = 0; r < 4; r++)
                    for (int j = 0; j < nr; j++)
                        C[(m + r) * N + n + j] = t[r][j];
            }
        }
    }
    for (; m < M; m++) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
            for (int k = 0; k < K; k++) {
                float32x4_t a = vdupq_n_f32(A[m * K + k]);
                c0 = vfmaq_f32(c0, a, vld1q_f32(bp + k * NR));
                c1 = vfmaq_f32(c1, a, vld1q_f32(bp + k * NR + 4));
            }
            if (bias) {
                c0 = vaddq_f32(c0, vld1q_f32(bias + n));
                c1 = vaddq_f32(c1, vld1q_f32(bias + n + 4));
            }
            if (n + NR <= N) {
                vst1q_f32(C + m * N + n,     c0);
                vst1q_f32(C + m * N + n + 4, c1);
            } else {
                float t[8];
                vst1q_f32(t,     c0); vst1q_f32(t + 4, c1);
                int nr = N - n;
                for (int j = 0; j < nr; j++) C[m * N + n + j] = t[j];
            }
        }
    }
#else
    /* Scalar fallback: B in column-panel format [ceil(N/NR), K, NR], NR=8. */
    {
        const int NR = 8;
        int n_panels = (N + NR - 1) / NR;
        for (int m = 0; m < M; m++) {
            for (int p = 0; p < n_panels; p++) {
                int n_base = p * NR;
                int nr = (n_base + NR <= N) ? NR : (N - n_base);
                const float* bp = B_packed + (size_t)p * K * NR;
                for (int j = 0; j < nr; j++) {
                    float s = bias ? bias[n_base + j] : 0.0f;
                    for (int k = 0; k < K; k++)
                        s += A[(size_t)m * K + k] * bp[k * NR + j];
                    C[(size_t)m * N + n_base + j] = s;
                }
            }
        }
    }
#endif
}

/* ============ Fused MatMul + Bias + GELU (single memory pass) ============ */
/* Computes: C[m,n] = GELU(A[M,K] × B_packed[K,N] + bias[N])
 * Bias and GELU applied in-register during store — saves 2 memory round-trips.
 * Uses A&S 7.1.26 erf approximation, same as gelu_fp32. */
#ifdef __AVX2__
static inline __m256 _gelu_ymm(__m256 v) {
    const __m256 v_inv_sqrt2 = _mm256_set1_ps(0.7071067811865476f);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);
    __m256 arg = _mm256_mul_ps(v, v_inv_sqrt2);
    __m256 ax = _mm256_andnot_ps(v_sign_mask, arg);
    __m256 sign = _mm256_and_ps(arg, v_sign_mask);
    __m256 t = _mm256_div_ps(v_one, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), ax, v_one));
    __m256 exp_val = _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(ax, ax)));
    __m256 poly = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429f), t, _mm256_set1_ps(-1.453152027f));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.421413741f));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-0.284496736f));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.254829592f));
    poly = _mm256_mul_ps(poly, t);
    __m256 erf_val = _mm256_xor_ps(_mm256_fnmadd_ps(poly, exp_val, v_one), sign);
    return _mm256_mul_ps(_mm256_mul_ps(v_half, v), _mm256_add_ps(v_one, erf_val));
}
#endif

void matmul_bias_gelu_packed(const float* A, const float* B_packed, const float* bias,
                              float* C, int M, int K, int N) {
#ifdef __AVX2__
    int NR = 8;
    int n_panels = (N + NR - 1) / NR;
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        int p = 0;
        for (; p + 2 <= n_panels; p += 2) {
            int n = p * NR;
            const float* bp0 = B_packed + (size_t)p * K * NR;
            const float* bp1 = bp0 + (size_t)K * NR;
            __m256 c00=_mm256_setzero_ps(),c01=_mm256_setzero_ps();
            __m256 c10=_mm256_setzero_ps(),c11=_mm256_setzero_ps();
            __m256 c20=_mm256_setzero_ps(),c21=_mm256_setzero_ps();
            __m256 c30=_mm256_setzero_ps(),c31=_mm256_setzero_ps();
            for (int k=0;k<K;k++){
                __m256 b0=_mm256_load_ps(bp0+k*NR),b1=_mm256_load_ps(bp1+k*NR);
                __m256 a0=_mm256_broadcast_ss(&A[(m+0)*K+k]);
                __m256 a1=_mm256_broadcast_ss(&A[(m+1)*K+k]);
                __m256 a2=_mm256_broadcast_ss(&A[(m+2)*K+k]);
                __m256 a3=_mm256_broadcast_ss(&A[(m+3)*K+k]);
                c00=_mm256_fmadd_ps(a0,b0,c00);c01=_mm256_fmadd_ps(a0,b1,c01);
                c10=_mm256_fmadd_ps(a1,b0,c10);c11=_mm256_fmadd_ps(a1,b1,c11);
                c20=_mm256_fmadd_ps(a2,b0,c20);c21=_mm256_fmadd_ps(a2,b1,c21);
                c30=_mm256_fmadd_ps(a3,b0,c30);c31=_mm256_fmadd_ps(a3,b1,c31);
            }
            /* Fused: add bias + GELU in registers before store */
            __m256 bb0=_mm256_loadu_ps(bias+n), bb1=_mm256_loadu_ps(bias+n+8);
            c00=_gelu_ymm(_mm256_add_ps(c00,bb0)); c01=_gelu_ymm(_mm256_add_ps(c01,bb1));
            c10=_gelu_ymm(_mm256_add_ps(c10,bb0)); c11=_gelu_ymm(_mm256_add_ps(c11,bb1));
            c20=_gelu_ymm(_mm256_add_ps(c20,bb0)); c21=_gelu_ymm(_mm256_add_ps(c21,bb1));
            c30=_gelu_ymm(_mm256_add_ps(c30,bb0)); c31=_gelu_ymm(_mm256_add_ps(c31,bb1));
            if(n+16<=N){
                _mm256_storeu_ps(C+(m+0)*N+n,c00);_mm256_storeu_ps(C+(m+0)*N+n+8,c01);
                _mm256_storeu_ps(C+(m+1)*N+n,c10);_mm256_storeu_ps(C+(m+1)*N+n+8,c11);
                _mm256_storeu_ps(C+(m+2)*N+n,c20);_mm256_storeu_ps(C+(m+2)*N+n+8,c21);
                _mm256_storeu_ps(C+(m+3)*N+n,c30);_mm256_storeu_ps(C+(m+3)*N+n+8,c31);
            } else {
                _mm256_storeu_ps(C+(m+0)*N+n,c00);_mm256_storeu_ps(C+(m+1)*N+n,c10);
                _mm256_storeu_ps(C+(m+2)*N+n,c20);_mm256_storeu_ps(C+(m+3)*N+n,c30);
                int nr=N-n-8;if(nr>0){float t[4][8];
                _mm256_storeu_ps(t[0],c01);_mm256_storeu_ps(t[1],c11);
                _mm256_storeu_ps(t[2],c21);_mm256_storeu_ps(t[3],c31);
                for(int i=0;i<4;i++)for(int j=0;j<nr;j++)C[(m+i)*N+n+8+j]=t[i][j];}
            }
        }
        for(;p<n_panels;p++){
            int n=p*NR;
            const float* bp=B_packed+(size_t)p*K*NR;
            __m256 c0=_mm256_setzero_ps(),c1=_mm256_setzero_ps();
            __m256 c2=_mm256_setzero_ps(),c3=_mm256_setzero_ps();
            for(int k=0;k<K;k++){
                __m256 b=_mm256_load_ps(bp+k*NR);
                c0=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+0)*K+k]),b,c0);
                c1=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+1)*K+k]),b,c1);
                c2=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+2)*K+k]),b,c2);
                c3=_mm256_fmadd_ps(_mm256_broadcast_ss(&A[(m+3)*K+k]),b,c3);
            }
            __m256 bb=_mm256_loadu_ps(bias+n);
            c0=_gelu_ymm(_mm256_add_ps(c0,bb));c1=_gelu_ymm(_mm256_add_ps(c1,bb));
            c2=_gelu_ymm(_mm256_add_ps(c2,bb));c3=_gelu_ymm(_mm256_add_ps(c3,bb));
            if(n+NR<=N){
                _mm256_storeu_ps(C+(m+0)*N+n,c0);_mm256_storeu_ps(C+(m+1)*N+n,c1);
                _mm256_storeu_ps(C+(m+2)*N+n,c2);_mm256_storeu_ps(C+(m+3)*N+n,c3);
            } else {float t[4][8];
                _mm256_storeu_ps(t[0],c0);_mm256_storeu_ps(t[1],c1);
                _mm256_storeu_ps(t[2],c2);_mm256_storeu_ps(t[3],c3);
                int nr=N-n;for(int i=0;i<4;i++)for(int j=0;j<nr;j++)C[(m+i)*N+n+j]=t[i][j];}
        }
    }
    for(;m<M;m++){
        const float* a_row=A+(size_t)m*K;
        for(int p=0;p<n_panels;p++){
            int n=p*NR;
            const float* bp=B_packed+(size_t)p*K*NR;
            __m256 c0=_mm256_setzero_ps();
            for(int k=0;k<K;k++)
                c0=_mm256_fmadd_ps(_mm256_broadcast_ss(&a_row[k]),_mm256_load_ps(bp+k*NR),c0);
            c0=_gelu_ymm(_mm256_add_ps(c0,_mm256_loadu_ps(bias+n)));
            if(n+NR<=N)_mm256_storeu_ps(C+(size_t)m*N+n,c0);
            else{float t[8];_mm256_storeu_ps(t,c0);for(int j=0;j<N-n;j++)C[(size_t)m*N+n+j]=t[j];}
        }
    }
#elif defined(FACEX_HAVE_NEON)
    /* AArch64 NEON: NR=8, MR=4 matmul; bias added with NEON; GELU applied
     * per element with erff after the GEMM completes (no vector erf on AArch64). */
    const int NR = 8;
    const float inv_sqrt2 = 0.7071067811865476f;
    int n_panels = (N + NR - 1) / NR;
    int m = 0;

    for (; m + 4 <= M; m += 4) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            for (int k = 0; k < K; k++) {
                float32x4_t b0 = vld1q_f32(bp + k * NR);
                float32x4_t b1 = vld1q_f32(bp + k * NR + 4);
                float32x4_t a0 = vdupq_n_f32(A[(m + 0) * K + k]);
                float32x4_t a1 = vdupq_n_f32(A[(m + 1) * K + k]);
                float32x4_t a2 = vdupq_n_f32(A[(m + 2) * K + k]);
                float32x4_t a3 = vdupq_n_f32(A[(m + 3) * K + k]);
                c00 = vfmaq_f32(c00, a0, b0); c01 = vfmaq_f32(c01, a0, b1);
                c10 = vfmaq_f32(c10, a1, b0); c11 = vfmaq_f32(c11, a1, b1);
                c20 = vfmaq_f32(c20, a2, b0); c21 = vfmaq_f32(c21, a2, b1);
                c30 = vfmaq_f32(c30, a3, b0); c31 = vfmaq_f32(c31, a3, b1);
            }
            float32x4_t bb0 = vld1q_f32(bias + n);
            float32x4_t bb1 = vld1q_f32(bias + n + 4);
            c00 = vaddq_f32(c00, bb0); c01 = vaddq_f32(c01, bb1);
            c10 = vaddq_f32(c10, bb0); c11 = vaddq_f32(c11, bb1);
            c20 = vaddq_f32(c20, bb0); c21 = vaddq_f32(c21, bb1);
            c30 = vaddq_f32(c30, bb0); c31 = vaddq_f32(c31, bb1);

            float t[4][8];
            vst1q_f32(t[0],     c00); vst1q_f32(t[0] + 4, c01);
            vst1q_f32(t[1],     c10); vst1q_f32(t[1] + 4, c11);
            vst1q_f32(t[2],     c20); vst1q_f32(t[2] + 4, c21);
            vst1q_f32(t[3],     c30); vst1q_f32(t[3] + 4, c31);
            int nr = (n + NR <= N) ? NR : (N - n);
            for (int r = 0; r < 4; r++)
                for (int j = 0; j < nr; j++) {
                    float v = t[r][j];
                    C[(m + r) * N + n + j] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
                }
        }
    }
    for (; m < M; m++) {
        for (int p = 0; p < n_panels; p++) {
            int n = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
            for (int k = 0; k < K; k++) {
                float32x4_t a = vdupq_n_f32(A[m * K + k]);
                c0 = vfmaq_f32(c0, a, vld1q_f32(bp + k * NR));
                c1 = vfmaq_f32(c1, a, vld1q_f32(bp + k * NR + 4));
            }
            c0 = vaddq_f32(c0, vld1q_f32(bias + n));
            c1 = vaddq_f32(c1, vld1q_f32(bias + n + 4));
            float t[8];
            vst1q_f32(t,     c0); vst1q_f32(t + 4, c1);
            int nr = (n + NR <= N) ? NR : (N - n);
            for (int j = 0; j < nr; j++) {
                float v = t[j];
                C[m * N + n + j] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
            }
        }
    }
#else
    /* Scalar fallback: B in column-panel format [ceil(N/NR), K, NR], NR=8. Bias + GELU. */
    {
        const int NR = 8;
        const float inv_sqrt2 = 0.7071067811865476f;
        int n_panels = (N + NR - 1) / NR;
        for (int m = 0; m < M; m++) {
            for (int p = 0; p < n_panels; p++) {
                int n_base = p * NR;
                int nr = (n_base + NR <= N) ? NR : (N - n_base);
                const float* bp = B_packed + (size_t)p * K * NR;
                for (int j = 0; j < nr; j++) {
                    float s = bias[n_base + j];
                    for (int k = 0; k < K; k++)
                        s += A[(size_t)m * K + k] * bp[k * NR + j];
                    C[(size_t)m * N + n_base + j] = 0.5f * s * (1.0f + erff(s * inv_sqrt2));
                }
            }
        }
    }
#endif
}

/* ============ MatMul + Add (bias) ============ */
void matmul_bias_fp32(const float* A, const float* B, const float* bias,
                      float* C, int M, int K, int N) {
    matmul_fp32(A, B, C, M, K, N);
    if (bias) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                C[(size_t)m * N + n] += bias[n];
    }
}

/* ============ L2 Normalize ============ */
/* Normalize each row to unit L2 norm, with clipping */
void l2_normalize_fp32(float* x, int N, int C, float min_norm) {
    for (int n = 0; n < N; n++) {
        float* row = x + (size_t)n * C;
        float norm_sq = 0;
        for (int c = 0; c < C; c++) norm_sq += row[c] * row[c];
        float norm = sqrtf(norm_sq);
        if (norm < min_norm) norm = min_norm;
        float inv = 1.0f / norm;
        for (int c = 0; c < C; c++) row[c] *= inv;
    }
}

/* ============ Adaptive Average Pool ============ */
/* Pool spatial dims H×W → 1×1, for [N, C, H, W] layout */
void adaptive_avg_pool_fp32(const float* x, int C, int H, int W, float* out) {
    int HW = H * W;
    float inv = 1.0f / HW;
    for (int c = 0; c < C; c++) {
        float sum = 0;
        for (int hw = 0; hw < HW; hw++)
            sum += x[(size_t)c * HW + hw];
        out[c] = sum * inv;
    }
}

/* ============ Depthwise Conv NxN (generalized) ============ */
/* Supports arbitrary kernel size (3,5,7,9), stride 1, pad=K/2 */
/* DW Conv NxN on HWC layout — AVX2 optimized for C multiple of 8 */
void depthwise_conv_nxn_hwc_fp32(
    const float* in, int H, int W, int C,
    const float* weights, /* [C, K, K] */
    const float* bias,
    int K,
    float* out)
{
    int pad = K / 2;

    /* Weights assumed pre-transposed to [K*K, C] layout */
    int KK = K * K;
    const float* w_t = weights;

    for (int oy = 0; oy < H; oy++) {
        for (int ox = 0; ox < W; ox++) {
            float* o = out + ((size_t)oy * W + ox) * C;

            /* Init with bias */
            int c = 0;
#ifdef __AVX512F__
            for (; c + 16 <= C; c += 16)
                _mm512_storeu_ps(o + c, bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps());
#endif
#ifdef __AVX2__
            for (; c + 8 <= C; c += 8)
                _mm256_storeu_ps(o + c, bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps());
#endif
            for (; c < C; c++) o[c] = bias ? bias[c] : 0;

            /* Accumulate kernel */
            for (int ky = 0; ky < K; ky++) {
                int iy = oy - pad + ky;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < K; kx++) {
                    int ix = ox - pad + kx;
                    if (ix < 0 || ix >= W) continue;
                    const float* inp = in + ((size_t)iy * W + ix) * C;
                    int ki = ky * K + kx;
                    c = 0;
#ifdef __AVX512F__
                    for (; c + 16 <= C; c += 16) {
                        __m512 vi = _mm512_loadu_ps(inp + c);
                        __m512 vw = _mm512_loadu_ps(w_t + ki * C + c);
                        _mm512_storeu_ps(o + c, _mm512_fmadd_ps(vi, vw, _mm512_loadu_ps(o + c)));
                    }
#endif
#ifdef __AVX2__
                    for (; c + 8 <= C; c += 8) {
                        __m256 vi = _mm256_loadu_ps(inp + c);
                        int ki2 = ky * K + kx;
                        __m256 vw = _mm256_loadu_ps(w_t + ki2 * C + c);
                        _mm256_storeu_ps(o + c, _mm256_fmadd_ps(vi, vw, _mm256_loadu_ps(o + c)));
                    }
#endif
                    for (; c < C; c++) {
                        o[c] += inp[c] * w_t[(ky * K + kx) * C + c];
                    }
                }
            }
        }
    }
    /* w_t is pre-transposed, no free needed */
}

void depthwise_conv_nxn_fp32(
    const float* in, int H, int W, int C,
    const float* weights, /* [C, 1, K, K] */
    const float* bias,
    int K, /* kernel size */
    float* out)
{
    int pad = K / 2;
    int OH = H, OW = W;

    /* Channel-first: each channel is independent spatial conv */
    for (int c = 0; c < C; c++) {
        const float* in_c = in + (size_t)c * H * W;
        const float* w_c = weights + (size_t)c * K * K;
        float* out_c = out + (size_t)c * OH * OW;
        float b = bias ? bias[c] : 0;

        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float sum = b;
                for (int ky = 0; ky < K; ky++) {
                    int iy = oy - pad + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < K; kx++) {
                        int ix = ox - pad + kx;
                        if (ix < 0 || ix >= W) continue;
                        sum += in_c[(size_t)iy * W + ix] * w_c[(size_t)ky * K + kx];
                    }
                }
                out_c[(size_t)oy * OW + ox] = sum;
            }
        }
    }
}
