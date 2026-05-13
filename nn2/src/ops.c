/*
 * ops.c — Utility operations: SiLU, sigmoid, upsample, concat, maxpool.
 */

#include "nn2_internal.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ============ SiLU (x * sigmoid(x)) in-place ============ */

#include "fast_exp.h"

void nn2_silu_inplace(float* x, int n)
{
    int i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(x + i, _mm512_silu_fast(v));
    }
#endif
#ifdef __AVX2__
    /* AVX2 fast sigmoid approximation: x/(1+|x|) * 0.5 + 0.5, then x*sigmoid */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        __m256 denom = _mm256_add_ps(one, abs_v);
        __m256 sig = _mm256_fmadd_ps(_mm256_div_ps(v, denom), half, half);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, sig));
    }
#endif
    for (; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ============ Sigmoid in-place ============ */

void nn2_sigmoid_inplace(float* x, int n)
{
    int i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(x + i, _mm512_sigmoid_fast(v));
    }
#endif
#ifdef __AVX2__
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        __m256 denom = _mm256_add_ps(one, abs_v);
        __m256 sig = _mm256_fmadd_ps(_mm256_div_ps(v, denom), half, half);
        _mm256_storeu_ps(x + i, sig);
    }
#endif
    for (; i < n; i++)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

/* ============ Nearest-neighbor upsample 2x ============ */

typedef struct { const float* in; float* out; int C, H, W; } UpsampleCtx;

static void upsample_worker(void* arg, int start, int end)
{
    UpsampleCtx* ctx = (UpsampleCtx*)arg;
    int H = ctx->H, W = ctx->W, Wout = W * 2;
    for (int c = start; c < end; c++) {
        const float* src = ctx->in + c * H * W;
        float* dst = ctx->out + c * H * 2 * Wout;
        for (int h = 0; h < H; h++) {
            const float* srow = src + h * W;
            float* drow0 = dst + (h * 2) * Wout;
            float* drow1 = drow0 + Wout;
            int w = 0;
#ifdef __AVX512F__
            /* Process 16 input pixels → 32 output pixels per row pair */
            for (; w + 16 <= W; w += 16) {
                __m512 v = _mm512_loadu_ps(srow + w);
                /* Interleave: [a,b,c,...] → [a,a,b,b,c,c,...] */
                /* Use permutex2var to duplicate each element */
                __m512i idx_lo = _mm512_setr_epi32(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7);
                __m512i idx_hi = _mm512_setr_epi32(8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15);
                __m512 lo = _mm512_permutexvar_ps(idx_lo, v);
                __m512 hi = _mm512_permutexvar_ps(idx_hi, v);
                _mm512_storeu_ps(drow0 + w * 2, lo);
                _mm512_storeu_ps(drow0 + w * 2 + 16, hi);
                _mm512_storeu_ps(drow1 + w * 2, lo);
                _mm512_storeu_ps(drow1 + w * 2 + 16, hi);
            }
#endif
            for (; w < W; w++) {
                float v = srow[w];
                drow0[w*2] = v; drow0[w*2+1] = v;
                drow1[w*2] = v; drow1[w*2+1] = v;
            }
        }
    }
}

void nn2_upsample_2x(const float* in, int C, int H, int W, float* out)
{
    UpsampleCtx ctx = {in, out, C, H, W};
    if (C >= 32)
        nn2_tp_parallel_for(upsample_worker, &ctx, C, 8);
    else
        upsample_worker(&ctx, 0, C);
}

/* ============ Channel concat ============ */

void nn2_concat(const float* a, int ca, const float* b, int cb,
                int H, int W, float* out)
{
    int spatial = H * W;
    memcpy(out, a, (size_t)ca * spatial * sizeof(float));
    memcpy(out + ca * spatial, b, (size_t)cb * spatial * sizeof(float));
}

/* ============ MaxPool2d ============ */

/* ============ Depthwise Conv2d (AVX-512 + threaded) ============ */

#include <immintrin.h>

typedef struct {
    const float* in; const float* w; const float* bias;
    int C, H, W, k, stride, pad, act;
    float* out;
    int Hout, Wout;
} DWConvCtx;

static void dwconv_worker(void* arg, int start, int end)
{
    DWConvCtx* ctx = (DWConvCtx*)arg;
    int H = ctx->H, W = ctx->W, k = ctx->k;
    int stride = ctx->stride, pad = ctx->pad;
    int Hout = ctx->Hout, Wout = ctx->Wout;
    int act = ctx->act;

    for (int c = start; c < end; c++) {
        const float* src = ctx->in + c * H * W;
        const float* wt = ctx->w + c * k * k;
        float b = ctx->bias ? ctx->bias[c] : 0.0f;
        float* dst = ctx->out + c * Hout * Wout;

        /* Optimized path for 3x3 stride-1 pad-1 (most common DW in ShuffleNet) */
        if (k == 3 && stride == 1 && pad == 1) {
            float w00=wt[0],w01=wt[1],w02=wt[2];
            float w10=wt[3],w11=wt[4],w12=wt[5];
            float w20=wt[6],w21=wt[7],w22=wt[8];

            for (int oh = 0; oh < Hout; oh++) {
                const float* r0 = (oh > 0) ? src + (oh-1)*W : NULL;
                const float* r1 = src + oh*W;
                const float* r2 = (oh < H-1) ? src + (oh+1)*W : NULL;
                int ow = 0;

#ifdef __AVX512F__
                __m512 vb = _mm512_set1_ps(b);
                __m512 vw00=_mm512_set1_ps(w00),vw01=_mm512_set1_ps(w01),vw02=_mm512_set1_ps(w02);
                __m512 vw10=_mm512_set1_ps(w10),vw11=_mm512_set1_ps(w11),vw12=_mm512_set1_ps(w12);
                __m512 vw20=_mm512_set1_ps(w20),vw21=_mm512_set1_ps(w21),vw22=_mm512_set1_ps(w22);
                __m512 vleak = _mm512_set1_ps(0.1f);
                __m512 vzero = _mm512_setzero_ps();

                for (; ow + 16 <= Wout; ow += 16) {
                    __m512 sum = vb;
                    /* Row 0 */
                    if (r0) {
                        __m512 left  = (ow>0) ? _mm512_loadu_ps(r0+ow-1) : _mm512_maskz_loadu_ps(0xFFFE, r0+ow-1);
                        __m512 center = _mm512_loadu_ps(r0+ow);
                        __m512 right = (ow+16<W) ? _mm512_loadu_ps(r0+ow+1) : _mm512_maskz_loadu_ps(0x7FFF, r0+ow);
                        if (ow+16 >= W) right = _mm512_loadu_ps(r0+ow+1); /* safe if W padded */
                        sum = _mm512_fmadd_ps(vw00, left, sum);
                        sum = _mm512_fmadd_ps(vw01, center, sum);
                        sum = _mm512_fmadd_ps(vw02, right, sum);
                    }
                    /* Row 1 */
                    {
                        __m512 left  = (ow>0) ? _mm512_loadu_ps(r1+ow-1) : _mm512_maskz_loadu_ps(0xFFFE, r1+ow-1);
                        __m512 center = _mm512_loadu_ps(r1+ow);
                        __m512 right = _mm512_loadu_ps(r1+ow+1);
                        sum = _mm512_fmadd_ps(vw10, left, sum);
                        sum = _mm512_fmadd_ps(vw11, center, sum);
                        sum = _mm512_fmadd_ps(vw12, right, sum);
                    }
                    /* Row 2 */
                    if (r2) {
                        __m512 left  = (ow>0) ? _mm512_loadu_ps(r2+ow-1) : _mm512_maskz_loadu_ps(0xFFFE, r2+ow-1);
                        __m512 center = _mm512_loadu_ps(r2+ow);
                        __m512 right = _mm512_loadu_ps(r2+ow+1);
                        sum = _mm512_fmadd_ps(vw20, left, sum);
                        sum = _mm512_fmadd_ps(vw21, center, sum);
                        sum = _mm512_fmadd_ps(vw22, right, sum);
                    }
                    /* LeakyReLU */
                    if (act == 3) {
                        __mmask16 neg = _mm512_cmplt_ps_mask(sum, vzero);
                        sum = _mm512_mask_mul_ps(sum, neg, sum, vleak);
                    }
                    _mm512_storeu_ps(dst + oh*Wout + ow, sum);
                }
#endif
                /* Scalar tail */
                for (; ow < Wout; ow++) {
                    float s = b;
                    if (r0) {
                        if (ow>0)   s += r0[ow-1]*w00;
                                    s += r0[ow]*w01;
                        if (ow<W-1) s += r0[ow+1]*w02;
                    }
                    if (ow>0)   s += r1[ow-1]*w10;
                                s += r1[ow]*w11;
                    if (ow<W-1) s += r1[ow+1]*w12;
                    if (r2) {
                        if (ow>0)   s += r2[ow-1]*w20;
                                    s += r2[ow]*w21;
                        if (ow<W-1) s += r2[ow+1]*w22;
                    }
                    if (act == 3 && s < 0) s *= 0.1f;
                    dst[oh*Wout + ow] = s;
                }
            }
            continue;
        }

        /* Optimized 5x5 stride-1 pad-2 (Ghost-PAN uses this heavily) */
        if (k == 5 && stride == 1 && pad == 2) {
#ifdef __AVX512F__
            __m512 vw[25], vb512 = _mm512_set1_ps(b);
            __m512 vleak = _mm512_set1_ps(0.1f), vzero = _mm512_setzero_ps();
            for (int i = 0; i < 25; i++) vw[i] = _mm512_set1_ps(wt[i]);

            for (int oh = 0; oh < Hout; oh++) {
                int ow = 0;
                for (; ow + 16 <= Wout; ow += 16) {
                    __m512 sum = vb512;
                    for (int kh = 0; kh < 5; kh++) {
                        int ih = oh - 2 + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* row = src + ih * W;
                        for (int kw = 0; kw < 5; kw++) {
                            int iw_base = ow - 2 + kw;
                            __m512 v;
                            if (iw_base >= 0 && iw_base + 16 <= W) {
                                v = _mm512_loadu_ps(row + iw_base);
                            } else {
                                /* Edge: masked load */
                                __mmask16 mask = 0;
                                for (int j = 0; j < 16; j++) {
                                    int iw = iw_base + j;
                                    if (iw >= 0 && iw < W) mask |= (1 << j);
                                }
                                v = _mm512_maskz_loadu_ps(mask, row + iw_base);
                            }
                            sum = _mm512_fmadd_ps(vw[kh*5+kw], v, sum);
                        }
                    }
                    if (act == 3) {
                        __mmask16 neg = _mm512_cmplt_ps_mask(sum, vzero);
                        sum = _mm512_mask_mul_ps(sum, neg, sum, vleak);
                    }
                    _mm512_storeu_ps(dst + oh * Wout + ow, sum);
                }
                /* Scalar tail */
                for (; ow < Wout; ow++) {
                    float s = b;
                    for (int kh = 0; kh < 5; kh++) {
                        int ih = oh - 2 + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < 5; kw++) {
                            int iw = ow - 2 + kw;
                            if (iw >= 0 && iw < W) s += src[ih*W+iw] * wt[kh*5+kw];
                        }
                    }
                    if (act == 3 && s < 0) s *= 0.1f;
                    dst[oh * Wout + ow] = s;
                }
            }
            continue;
#endif
        }

        /* Optimized 5x5 stride-2 pad-2 (downsample in Ghost-PAN) */
        if (k == 5 && stride == 2 && pad == 2) {
#ifdef __AVX512F__
            __m512 vw5[25], vb512 = _mm512_set1_ps(b);
            __m512 vleak = _mm512_set1_ps(0.1f), vzero = _mm512_setzero_ps();
            __m256i vidx8 = _mm256_setr_epi32(0,2,4,6,8,10,12,14);
            for (int i = 0; i < 25; i++) vw5[i] = _mm512_set1_ps(wt[i]);

            for (int oh = 0; oh < Hout; oh++) {
                int ow = 0;
                for (; ow + 8 <= Wout; ow += 8) {
                    __m256 sum8 = _mm256_set1_ps(b);
                    for (int kh = 0; kh < 5; kh++) {
                        int ih = oh * 2 - 2 + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* row = src + ih * W;
                        for (int kw = 0; kw < 5; kw++) {
                            int iw_base = ow * 2 - 2 + kw;
                            __m256 v;
                            if (iw_base >= 0 && iw_base + 14 < W) {
                                v = _mm256_i32gather_ps(row + iw_base, vidx8, 4);
                            } else {
                                float tmp[8] = {0};
                                for (int j = 0; j < 8; j++) {
                                    int iw = iw_base + j * 2;
                                    if (iw >= 0 && iw < W) tmp[j] = row[iw];
                                }
                                v = _mm256_loadu_ps(tmp);
                            }
                            __m256 wv = _mm256_set1_ps(wt[kh*5+kw]);
                            sum8 = _mm256_fmadd_ps(wv, v, sum8);
                        }
                    }
                    if (act == 3) {
                        __m256 neg = _mm256_cmp_ps(sum8, _mm256_setzero_ps(), _CMP_LT_OQ);
                        sum8 = _mm256_blendv_ps(sum8, _mm256_mul_ps(sum8, _mm256_set1_ps(0.1f)), neg);
                    }
                    _mm256_storeu_ps(dst + oh * Wout + ow, sum8);
                }
                for (; ow < Wout; ow++) {
                    float s = b;
                    for (int kh = 0; kh < 5; kh++) {
                        int ih = oh*2-2+kh;
                        if (ih<0||ih>=H) continue;
                        for (int kw = 0; kw < 5; kw++) {
                            int iw = ow*2-2+kw;
                            if (iw>=0&&iw<W) s += src[ih*W+iw]*wt[kh*5+kw];
                        }
                    }
                    if (act==3&&s<0) s*=0.1f;
                    dst[oh*Wout+ow] = s;
                }
            }
            continue;
#endif
        }

        /* Generic fallback */
        for (int oh = 0; oh < Hout; oh++) {
            for (int ow = 0; ow < Wout; ow++) {
                float sum = b;
                for (int kh = 0; kh < k; kh++) {
                    int ih = oh * stride - pad + kh;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < k; kw++) {
                        int iw = ow * stride - pad + kw;
                        if (iw < 0 || iw >= W) continue;
                        sum += src[ih * W + iw] * wt[kh * k + kw];
                    }
                }
                if (act == 1) { float s = 1.0f/(1.0f+expf(-sum)); sum *= s; }
                else if (act == 3 && sum < 0) sum *= 0.1f;
                dst[oh * Wout + ow] = sum;
            }
        }
    }
}

void nn2_dwconv2d(const float* in, const float* w, const float* bias,
                  int C, int H, int W, int k, int stride, int pad,
                  int act, float* out)
{
    int Hout = (H + 2 * pad - k) / stride + 1;
    int Wout = (W + 2 * pad - k) / stride + 1;
    DWConvCtx ctx = {in, w, bias, C, H, W, k, stride, pad, act, out, Hout, Wout};

    if (C >= 8)
        nn2_tp_parallel_for(dwconv_worker, &ctx, C, 4);
    else
        dwconv_worker(&ctx, 0, C);
}

/* ============ Channel shuffle (ShuffleNetV2) ============ */

void nn2_channel_shuffle(const float* in, int C, int H, int W,
                         int groups, float* out)
{
    int channels_per_group = C / groups;
    int spatial = H * W;
    for (int g = 0; g < groups; g++) {
        for (int c = 0; c < channels_per_group; c++) {
            int src_ch = g * channels_per_group + c;
            int dst_ch = c * groups + g;
            memcpy(out + dst_ch * spatial, in + src_ch * spatial,
                   spatial * sizeof(float));
        }
    }
}

/* ============ MaxPool2d ============ */

/* Parallel maxpool worker: process channels [start, end) */
typedef struct { const float* in; float* out; int C, H, W, k, stride, pad; } MaxPoolCtx;

static void maxpool_worker(void* arg, int start, int end)
{
    MaxPoolCtx* ctx = (MaxPoolCtx*)arg;
    int H = ctx->H, W = ctx->W, k = ctx->k, stride = ctx->stride, pad = ctx->pad;
    int Hout = (H + 2*pad - k) / stride + 1;
    int Wout = (W + 2*pad - k) / stride + 1;

    for (int c = start; c < end; c++) {
        const float* src = ctx->in + c * H * W;
        float* dst = ctx->out + c * Hout * Wout;

        /* Fast path: SPPF-style same-size maxpool (stride=1) with separable passes.
         * For k=5,s=1,p=2: do 2 sequential k=3,s=1,p=1 maxpools (5=3+3-1). */
        if (stride == 1 && k == 5 && pad == 2 && Hout == H && Wout == W) {
            /* Two-pass k=3 maxpool (horizontal then vertical, repeated).
             * But simpler: just process each output position with 5×5 window
             * using AVX-512 across the width dimension. */
            for (int oh = 0; oh < Hout; oh++) {
                int ow = 0;
#ifdef __AVX512F__
                for (; ow + 16 <= Wout; ow += 16) {
                    __m512 vmax = _mm512_set1_ps(-FLT_MAX);
                    for (int kh = 0; kh < 5; kh++) {
                        int ih = oh - 2 + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* srow = src + ih * W;
                        for (int kw = 0; kw < 5; kw++) {
                            int iw = ow - 2 + kw;
                            /* Need to handle boundaries per-element */
                            if (iw >= 0 && iw + 16 <= W) {
                                __m512 v = _mm512_loadu_ps(srow + iw);
                                vmax = _mm512_max_ps(vmax, v);
                            } else {
                                /* Boundary: load with mask */
                                __mmask16 mask = 0;
                                for (int j = 0; j < 16; j++)
                                    if (iw + j >= 0 && iw + j < W) mask |= (1 << j);
                                if (mask) {
                                    __m512 v = _mm512_maskz_loadu_ps(mask, srow + (iw >= 0 ? iw : 0));
                                    /* Shift loaded values if iw < 0 */
                                    if (iw < 0) {
                                        /* fallback to scalar for boundary */
                                        for (int j = 0; j < 16; j++) {
                                            if (iw+j >= 0 && iw+j < W) {
                                                float sv = srow[iw+j];
                                                float cv = ((float*)&vmax)[j];
                                                if (sv > cv) ((float*)&vmax)[j] = sv;
                                            }
                                        }
                                        continue;
                                    }
                                    vmax = _mm512_max_ps(vmax, v);
                                }
                            }
                        }
                    }
                    _mm512_storeu_ps(dst + oh * Wout + ow, vmax);
                }
#endif
                for (; ow < Wout; ow++) {
                    float maxv = -FLT_MAX;
                    for (int kh = 0; kh < 5; kh++) {
                        int ih = oh - 2 + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < 5; kw++) {
                            int iw = ow - 2 + kw;
                            if (iw < 0 || iw >= W) continue;
                            float v = src[ih * W + iw];
                            if (v > maxv) maxv = v;
                        }
                    }
                    dst[oh * Wout + ow] = maxv;
                }
            }
            continue;
        }

        /* Generic path */
        for (int oh = 0; oh < Hout; oh++) {
            for (int ow = 0; ow < Wout; ow++) {
                float maxv = -FLT_MAX;
                for (int kh = 0; kh < k; kh++) {
                    int ih = oh * stride - pad + kh;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < k; kw++) {
                        int iw = ow * stride - pad + kw;
                        if (iw < 0 || iw >= W) continue;
                        float v = src[ih * W + iw];
                        if (v > maxv) maxv = v;
                    }
                }
                dst[oh * Wout + ow] = maxv;
            }
        }
    }
}

void nn2_maxpool2d(const float* in, int C, int H, int W,
                   int k, int stride, int pad, float* out)
{
    MaxPoolCtx ctx = {in, out, C, H, W, k, stride, pad};
    if (C >= 16)
        nn2_tp_parallel_for(maxpool_worker, &ctx, C, 4);
    else
        maxpool_worker(&ctx, 0, C);
}
