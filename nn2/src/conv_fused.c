/*
 * conv_fused.c — Fused im2col+GEMM for 3x3 convolutions.
 *
 * Instead of materializing the full im2col matrix (15+ MB for early layers),
 * load input elements directly during GEMM computation.
 *
 * For stride-1: consecutive output positions → contiguous input loads.
 * For stride-2: consecutive output positions → stride-2 input loads (gather).
 *
 * Processes MR=6 output channels × 16 output positions per microkernel.
 */

#include "nn2_internal.h"
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>

/* Fast SiLU from conv.c */
static inline __m256 _fast_sigmoid(__m256 x)
{
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one  = _mm256_set1_ps(1.0f);
    __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    return _mm256_fmadd_ps(half, _mm256_div_ps(x, _mm256_add_ps(one, ax)), half);
}
static inline __m256 _fast_silu(__m256 x)
{
    return _mm256_mul_ps(x, _fast_sigmoid(x));
}

/* ============ Fused 3x3 stride-1 pad-1 ============ */

void nn2_conv3x3_s1_fused(const ConvLayer* L, const float* in, int H, int W,
                           float* out)
{
    int co = L->cout, ci = L->cin;
    /* Process MR=6 output channels at a time */
    for (int oc0 = 0; oc0 < co; oc0 += NN2_MR) {
        int mr = (oc0 + NN2_MR <= co) ? NN2_MR : co - oc0;

        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow += 16) {
                int nr = (ow + 16 <= W) ? 16 : W - ow;

                /* Accumulators: mr × 2 YMM (16 output cols) */
                __m256 c0_lo = _mm256_setzero_ps(), c0_hi = _mm256_setzero_ps();
                __m256 c1_lo = _mm256_setzero_ps(), c1_hi = _mm256_setzero_ps();
                __m256 c2_lo = _mm256_setzero_ps(), c2_hi = _mm256_setzero_ps();
                __m256 c3_lo = _mm256_setzero_ps(), c3_hi = _mm256_setzero_ps();
                __m256 c4_lo = _mm256_setzero_ps(), c4_hi = _mm256_setzero_ps();
                __m256 c5_lo = _mm256_setzero_ps(), c5_hi = _mm256_setzero_ps();

                /* Accumulate over input channels and kernel positions */
                for (int ic = 0; ic < ci; ic++) {
                    const float* src = in + ic * H * W;

                    for (int kh = 0; kh < 3; kh++) {
                        int ih = oh - 1 + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* srow = src + ih * W;

                        for (int kw = 0; kw < 3; kw++) {
                            int iw = ow - 1 + kw;

                            /* Load 16 input values.
                             * For interior positions (1 ≤ ow ≤ W-17, kw=1):
                             * just load contiguous. Handle edges with scalar. */
                            __m256 v_lo, v_hi;

                            if (iw >= 0 && iw + 16 <= W && nr == 16) {
                                /* Fast path: all 16 values in bounds */
                                v_lo = _mm256_loadu_ps(srow + iw);
                                v_hi = _mm256_loadu_ps(srow + iw + 8);
                            } else {
                                /* Edge: load with bounds checking */
                                float tmp[16] = {0};
                                for (int j = 0; j < nr; j++) {
                                    int iw_j = iw + j;
                                    if (iw_j >= 0 && iw_j < W)
                                        tmp[j] = srow[iw_j];
                                }
                                v_lo = _mm256_loadu_ps(tmp);
                                v_hi = _mm256_loadu_ps(tmp + 8);
                            }

                            /* Multiply-accumulate for each output channel */
                            int woff = ic * 9 + kh * 3 + kw;
                            #define FMA_OC(i, lo, hi) if (i < mr) { \
                                __m256 vw = _mm256_broadcast_ss(&L->w[(oc0+(i))*ci*9 + woff]); \
                                lo = _mm256_fmadd_ps(vw, v_lo, lo); \
                                hi = _mm256_fmadd_ps(vw, v_hi, hi); \
                            }
                            FMA_OC(0, c0_lo, c0_hi);
                            FMA_OC(1, c1_lo, c1_hi);
                            FMA_OC(2, c2_lo, c2_hi);
                            FMA_OC(3, c3_lo, c3_hi);
                            FMA_OC(4, c4_lo, c4_hi);
                            FMA_OC(5, c5_lo, c5_hi);
                            #undef FMA_OC
                        }
                    }
                }

                /* Add bias + activation + store */
                #define FINISH_OC(i, lo, hi) if (i < mr) { \
                    __m256 vb = _mm256_broadcast_ss(&L->b[oc0+(i)]); \
                    lo = _mm256_add_ps(lo, vb); \
                    hi = _mm256_add_ps(hi, vb); \
                    if (L->act == 1) { lo = _fast_silu(lo); hi = _fast_silu(hi); } \
                    float* dst = out + (oc0+(i)) * H * W + oh * W + ow; \
                    if (nr >= 16) { \
                        _mm256_storeu_ps(dst, lo); \
                        _mm256_storeu_ps(dst + 8, hi); \
                    } else { \
                        float tmp[16]; \
                        _mm256_storeu_ps(tmp, lo); _mm256_storeu_ps(tmp + 8, hi); \
                        memcpy(dst, tmp, nr * sizeof(float)); \
                    } \
                }
                FINISH_OC(0, c0_lo, c0_hi);
                FINISH_OC(1, c1_lo, c1_hi);
                FINISH_OC(2, c2_lo, c2_hi);
                FINISH_OC(3, c3_lo, c3_hi);
                FINISH_OC(4, c4_lo, c4_hi);
                FINISH_OC(5, c5_lo, c5_hi);
                #undef FINISH_OC
            }
        }
    }
}

/* ============ Fused 3x3 stride-2 pad-1 ============ */

void nn2_conv3x3_s2_fused(const ConvLayer* L, const float* in, int H, int W,
                           float* out)
{
    int Hout = (H + 1) / 2, Wout = (W + 1) / 2;
    int co = L->cout, ci = L->cin;
    __m256i vidx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);

    for (int oc0 = 0; oc0 < co; oc0 += NN2_MR) {
        int mr = (oc0 + NN2_MR <= co) ? NN2_MR : co - oc0;

        for (int oh = 0; oh < Hout; oh++) {
            for (int ow = 0; ow < Wout; ow += 16) {
                int nr = (ow + 16 <= Wout) ? 16 : Wout - ow;

                __m256 c0_lo = _mm256_setzero_ps(), c0_hi = _mm256_setzero_ps();
                __m256 c1_lo = _mm256_setzero_ps(), c1_hi = _mm256_setzero_ps();
                __m256 c2_lo = _mm256_setzero_ps(), c2_hi = _mm256_setzero_ps();
                __m256 c3_lo = _mm256_setzero_ps(), c3_hi = _mm256_setzero_ps();
                __m256 c4_lo = _mm256_setzero_ps(), c4_hi = _mm256_setzero_ps();
                __m256 c5_lo = _mm256_setzero_ps(), c5_hi = _mm256_setzero_ps();

                for (int ic = 0; ic < ci; ic++) {
                    const float* src = in + ic * H * W;

                    for (int kh = 0; kh < 3; kh++) {
                        int ih = oh * 2 - 1 + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* srow = src + ih * W;

                        for (int kw = 0; kw < 3; kw++) {
                            int iw_base = ow * 2 - 1 + kw;
                            __m256 v_lo, v_hi;

                            /* Gather stride-2 values using AVX2 gather */
                            if (iw_base >= 0 && iw_base + 30 < W && nr == 16) {
                                v_lo = _mm256_i32gather_ps(srow + iw_base, vidx, 4);
                                v_hi = _mm256_i32gather_ps(srow + iw_base + 16, vidx, 4);
                            } else {
                                float tmp[16] = {0};
                                for (int j = 0; j < nr; j++) {
                                    int iw = iw_base + j * 2;
                                    if (iw >= 0 && iw < W) tmp[j] = srow[iw];
                                }
                                v_lo = _mm256_loadu_ps(tmp);
                                v_hi = _mm256_loadu_ps(tmp + 8);
                            }

                            int woff = ic * 9 + kh * 3 + kw;
                            #define FMA_OC2(i, lo, hi) if (i < mr) { \
                                __m256 vw = _mm256_broadcast_ss(&L->w[(oc0+(i))*ci*9 + woff]); \
                                lo = _mm256_fmadd_ps(vw, v_lo, lo); \
                                hi = _mm256_fmadd_ps(vw, v_hi, hi); \
                            }
                            FMA_OC2(0, c0_lo, c0_hi);
                            FMA_OC2(1, c1_lo, c1_hi);
                            FMA_OC2(2, c2_lo, c2_hi);
                            FMA_OC2(3, c3_lo, c3_hi);
                            FMA_OC2(4, c4_lo, c4_hi);
                            FMA_OC2(5, c5_lo, c5_hi);
                            #undef FMA_OC2
                        }
                    }
                }

                #define FINISH_OC2(i, lo, hi) if (i < mr) { \
                    __m256 vb = _mm256_broadcast_ss(&L->b[oc0+(i)]); \
                    lo = _mm256_add_ps(lo, vb); hi = _mm256_add_ps(hi, vb); \
                    if (L->act == 1) { lo = _fast_silu(lo); hi = _fast_silu(hi); } \
                    float* dst = out + (oc0+(i)) * Hout * Wout + oh * Wout + ow; \
                    if (nr >= 16) { \
                        _mm256_storeu_ps(dst, lo); _mm256_storeu_ps(dst + 8, hi); \
                    } else { \
                        float tmp[16]; \
                        _mm256_storeu_ps(tmp, lo); _mm256_storeu_ps(tmp + 8, hi); \
                        memcpy(dst, tmp, nr * sizeof(float)); \
                    } \
                }
                FINISH_OC2(0, c0_lo, c0_hi);
                FINISH_OC2(1, c1_lo, c1_hi);
                FINISH_OC2(2, c2_lo, c2_hi);
                FINISH_OC2(3, c3_lo, c3_hi);
                FINISH_OC2(4, c4_lo, c4_hi);
                FINISH_OC2(5, c5_lo, c5_hi);
                #undef FINISH_OC2
            }
        }
    }
}

#endif /* __AVX2__ */
