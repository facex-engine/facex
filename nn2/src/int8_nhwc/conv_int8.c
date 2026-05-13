/*
 * conv_int8.c — INT8 NHWC convolution: 1x1 and 3x3.
 *
 * NHWC layout: input[H][W][C_pad] — channels are contiguous.
 * For 1x1: input IS the A matrix for GEMM (no im2col!).
 * For 3x3: im2col gathers [spatial, K4] from NHWC — channels contiguous.
 *
 * This is where NHWC pays off: im2col just copies contiguous C-sized
 * blocks from neighboring pixels. No scatter/gather.
 */

#include "nn2_int8.h"
#include "../nn2_internal.h"
#include <string.h>

/* ============ 1x1 Conv (direct GEMM) ============ */

void i8_conv1x1(const I8Conv* L, const int8_t* input, int H, int W,
                int8_t* output)
{
    int spatial = H * W;
    int cin_pad = C_PAD16(L->cin);
    int cout_pad = L->cout_pad;

    /* In NHWC: input is [spatial, cin_pad], treat as A[M=spatial, K=cin_pad]
     * Weight is B_packed[K4, Cout] in VNNI format.
     * Output is [spatial, cout_pad]. */
    i8_gemm_bias_act(input, spatial, L->K4, L->cout,
                      L->w_packed, L->col_sums, L->w_scales, L->bias,
                      L->act_scale, L->out_scale, L->act,
                      output, cout_pad);
}

/* ============ 3x3 im2col in INT8 NHWC ============ */

/* im2col for 3x3 s=1 p=1: NHWC input → [spatial, K4] contiguous.
 * For each output pixel (oh,ow), gather 3×3 neighborhood × C channels.
 * In NHWC, each pixel's channels are contiguous → memcpy per neighbor pixel. */
static void i8_im2col_3x3_s1(const int8_t* input, int C_pad, int H, int W,
                               int8_t* col, int K4)
{
    int C = C_pad; /* use padded channel count for alignment */
    int K = C * 9;

    for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
            int8_t* dst = col + (oh * W + ow) * K4;
            int k = 0;
            for (int kh = 0; kh < 3; kh++) {
                int ih = oh - 1 + kh;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow - 1 + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        /* Copy C contiguous bytes from NHWC input */
                        memcpy(dst + k, input + (ih * W + iw) * C_pad, C);
                    } else {
                        memset(dst + k, 0, C);
                    }
                    k += C;
                }
            }
            /* Zero-pad K to K4 */
            if (k < K4) memset(dst + k, 0, K4 - k);
        }
    }
}

/* im2col for 3x3 s=2 p=1 */
static void i8_im2col_3x3_s2(const int8_t* input, int C_pad, int H, int W,
                               int Hout, int Wout, int8_t* col, int K4)
{
    int C = C_pad;
    for (int oh = 0; oh < Hout; oh++) {
        for (int ow = 0; ow < Wout; ow++) {
            int8_t* dst = col + (oh * Wout + ow) * K4;
            int k = 0;
            for (int kh = 0; kh < 3; kh++) {
                int ih = oh * 2 - 1 + kh;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = ow * 2 - 1 + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        memcpy(dst + k, input + (ih * W + iw) * C_pad, C);
                    } else {
                        memset(dst + k, 0, C);
                    }
                    k += C;
                }
            }
            if (k < K4) memset(dst + k, 0, K4 - k);
        }
    }
}

/* Parallel im2col wrapper */
typedef struct {
    const int8_t* input; int C_pad, H, W, Hout, Wout, K4, stride;
    int8_t* col;
} I8Im2colCtx;

static void i8_im2col_worker(void* arg, int start, int end)
{
    I8Im2colCtx* g = (I8Im2colCtx*)arg;
    int C = g->C_pad;
    int K4 = g->K4;
    int Wout = g->Wout;

    for (int idx = start; idx < end; idx++) {
        int oh = idx / Wout;
        int ow = idx % Wout;
        int8_t* dst = g->col + idx * K4;
        int k = 0;
        for (int kh = 0; kh < 3; kh++) {
            int ih = oh * g->stride - 1 + kh;
            for (int kw = 0; kw < 3; kw++) {
                int iw = ow * g->stride - 1 + kw;
                if (ih >= 0 && ih < g->H && iw >= 0 && iw < g->W) {
                    memcpy(dst + k, g->input + (ih * g->W + iw) * C, C);
                } else {
                    memset(dst + k, 0, C);
                }
                k += C;
            }
        }
        if (k < K4) memset(dst + k, 0, K4 - k);
    }
}

/* ============ 3x3 Conv ============ */

void i8_conv3x3(const I8Conv* L, const int8_t* input, int H, int W,
                int8_t* output, int8_t* im2col_buf)
{
    int Hout = (H + 2 * L->pad - L->k) / L->stride + 1;
    int Wout = (W + 2 * L->pad - L->k) / L->stride + 1;
    int spatial = Hout * Wout;
    int cin_pad = C_PAD16(L->cin);
    int cout_pad = L->cout_pad;

    /* INT8 im2col: NHWC → [spatial, K4] with parallel dispatch */
    I8Im2colCtx ctx = {input, cin_pad, H, W, Hout, Wout, L->K4, L->stride, im2col_buf};
    if (spatial >= 64)
        nn2_tp_parallel_for(i8_im2col_worker, &ctx, spatial, spatial > 256 ? spatial / 12 : 8);
    else
        i8_im2col_worker(&ctx, 0, spatial);

    /* VNNI GEMM */
    i8_gemm_bias_act(im2col_buf, spatial, L->K4, L->cout,
                      L->w_packed, L->col_sums, L->w_scales, L->bias,
                      L->act_scale, L->out_scale, L->act,
                      output, cout_pad);
}

/* ============ Generic conv dispatch ============ */

void i8_conv(const I8Conv* L, const int8_t* input, int H, int W,
             int8_t* output, int8_t* scratch)
{
    if (L->k == 1 && L->stride == 1)
        i8_conv1x1(L, input, H, W, output);
    else
        i8_conv3x3(L, input, H, W, output, scratch);
}
