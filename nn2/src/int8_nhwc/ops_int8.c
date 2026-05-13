/*
 * ops_int8.c — INT8 NHWC operations: quantize, dequant, upsample, concat, maxpool.
 *
 * NHWC makes many ops trivial:
 * - Upsample: duplicate [W,C_pad] rows → contiguous copy
 * - Concat: copy Ca bytes then Cb bytes per pixel → simple interleave
 * - MaxPool: compare int8 values directly (no float conversion!)
 */

#include "nn2_int8.h"
#include "../nn2_internal.h"
#include <string.h>
#include <math.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ============ FP32 CHW → INT8 NHWC ============ */

void i8_quantize_nhwc(const float* fp32_chw, int C, int H, int W,
                       float scale, int8_t* out_nhwc, int C_pad)
{
    float inv_scale = 1.0f / (scale + 1e-10f);
    int spatial = H * W;

    /* Transpose CHW → HWC and quantize */
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int8_t* dst = out_nhwc + (h * W + w) * C_pad;
            for (int c = 0; c < C; c++) {
                float v = fp32_chw[c * spatial + h * W + w];
                int q = (int)roundf(v * inv_scale);
                dst[c] = (int8_t)(q < -127 ? -127 : q > 127 ? 127 : q);
            }
            /* Pad remaining channels to 0 */
            for (int c = C; c < C_pad; c++) dst[c] = 0;
        }
    }
}

/* ============ INT8 NHWC → FP32 CHW ============ */

void i8_dequant_to_chw(const int8_t* int8_nhwc, int C, int H, int W,
                        int C_pad, float scale, float* fp32_chw)
{
    int spatial = H * W;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            const int8_t* src = int8_nhwc + (h * W + w) * C_pad;
            for (int c = 0; c < C; c++)
                fp32_chw[c * spatial + h * W + w] = (float)src[c] * scale;
        }
    }
}

/* ============ Upsample 2× in INT8 NHWC ============ */

void i8_upsample_2x(const int8_t* in, int C_pad, int H, int W, int8_t* out)
{
    int Wout = W * 2;
    /* NHWC upsample: each pixel [C_pad bytes] gets duplicated to 2×2 block.
     * Row by row: copy each pixel twice, then duplicate the row. */
    for (int h = 0; h < H; h++) {
        const int8_t* srow = in + h * W * C_pad;
        int8_t* drow0 = out + (h * 2) * Wout * C_pad;
        int8_t* drow1 = drow0 + Wout * C_pad;

        for (int w = 0; w < W; w++) {
            const int8_t* px = srow + w * C_pad;
            /* Write pixel to (2w, 2h) and (2w+1, 2h) */
            memcpy(drow0 + (w * 2) * C_pad, px, C_pad);
            memcpy(drow0 + (w * 2 + 1) * C_pad, px, C_pad);
        }
        /* Duplicate row: drow1 = drow0 */
        memcpy(drow1, drow0, Wout * C_pad);
    }
}

/* ============ Concat in INT8 NHWC ============ */

void i8_concat(const int8_t* a, int Ca_pad, const int8_t* b, int Cb_pad,
               int H, int W, int8_t* out, int Cout_pad,
               float scale_a, float scale_b, float scale_out)
{
    int spatial = H * W;
    int Ca = Ca_pad; /* actual padded sizes */
    int Cb = Cb_pad;

    /* If scales match, just copy bytes (no rescaling needed) */
    if (fabsf(scale_a - scale_out) < 1e-6f && fabsf(scale_b - scale_out) < 1e-6f) {
        for (int i = 0; i < spatial; i++) {
            memcpy(out + i * Cout_pad, a + i * Ca_pad, Ca);
            memcpy(out + i * Cout_pad + Ca, b + i * Cb_pad, Cb);
            /* Zero-pad if Cout_pad > Ca + Cb */
            if (Ca + Cb < Cout_pad)
                memset(out + i * Cout_pad + Ca + Cb, 0, Cout_pad - Ca - Cb);
        }
    } else {
        /* Different scales: requantize during concat */
        float sa = scale_a / scale_out;
        float sb = scale_b / scale_out;
        for (int i = 0; i < spatial; i++) {
            const int8_t* pa = a + i * Ca_pad;
            const int8_t* pb = b + i * Cb_pad;
            int8_t* po = out + i * Cout_pad;
            for (int c = 0; c < Ca; c++) {
                int v = (int)roundf((float)pa[c] * sa);
                po[c] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
            }
            for (int c = 0; c < Cb; c++) {
                int v = (int)roundf((float)pb[c] * sb);
                po[Ca + c] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
            }
            if (Ca + Cb < Cout_pad)
                memset(po + Ca + Cb, 0, Cout_pad - Ca - Cb);
        }
    }
}

/* ============ MaxPool in INT8 NHWC ============ */

void i8_maxpool(const int8_t* in, int C_pad, int H, int W,
                int k, int stride, int pad, int8_t* out)
{
    int Hout = (H + 2 * pad - k) / stride + 1;
    int Wout = (W + 2 * pad - k) / stride + 1;

    /* INT8 maxpool: compare bytes directly! Much faster than float. */
    for (int oh = 0; oh < Hout; oh++) {
        for (int ow = 0; ow < Wout; ow++) {
            int8_t* dst = out + (oh * Wout + ow) * C_pad;
            /* Init to minimum */
            memset(dst, 0x80, C_pad); /* -128 */

            for (int kh = 0; kh < k; kh++) {
                int ih = oh * stride - pad + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < k; kw++) {
                    int iw = ow * stride - pad + kw;
                    if (iw < 0 || iw >= W) continue;
                    const int8_t* src = in + (ih * W + iw) * C_pad;
                    /* Max per channel — can use SIMD! */
                    int c = 0;
#ifdef __AVX512BW__
                    for (; c + 64 <= C_pad; c += 64) {
                        __m512i vd = _mm512_loadu_si512((__m512i*)(dst + c));
                        __m512i vs = _mm512_loadu_si512((__m512i*)(src + c));
                        _mm512_storeu_si512((__m512i*)(dst + c), _mm512_max_epi8(vd, vs));
                    }
#endif
                    for (; c < C_pad; c++)
                        if (src[c] > dst[c]) dst[c] = src[c];
                }
            }
        }
    }
}

/* ============ Shortcut (residual) add in INT8 ============ */

void i8_residual_add(int8_t* a, const int8_t* b, int total_bytes,
                      float scale_a, float scale_b, float scale_out)
{
    /* If all scales are the same, add and clamp */
    if (fabsf(scale_a - scale_b) < 1e-6f && fabsf(scale_a - scale_out) < 1e-6f) {
        int i = 0;
#ifdef __AVX512BW__
        for (; i + 64 <= total_bytes; i += 64) {
            __m512i va = _mm512_loadu_si512((__m512i*)(a + i));
            __m512i vb = _mm512_loadu_si512((__m512i*)(b + i));
            _mm512_storeu_si512((__m512i*)(a + i), _mm512_adds_epi8(va, vb));
        }
#endif
        for (; i < total_bytes; i++) {
            int v = (int)a[i] + (int)b[i];
            a[i] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
    } else {
        /* Different scales: dequant, add, requant */
        float sa = scale_a / scale_out;
        float sb = scale_b / scale_out;
        for (int i = 0; i < total_bytes; i++) {
            int v = (int)roundf((float)a[i] * sa + (float)b[i] * sb);
            a[i] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
    }
}
