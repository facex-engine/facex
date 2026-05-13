/*
 * c2f_int8.c — INT8 NHWC C2f block and helper ops.
 *
 * NHWC simplifies C2f greatly:
 * - cv1 output [H,W,2c_pad] → chunk0=[0..c), chunk1=[c..2c) per pixel
 * - Bottleneck: extract chunk as contiguous → cv1 → cv2 → write to cat
 * - cv2 input is the full concat buffer [H,W,total_c_pad]
 *
 * Channel slice in NHWC: for each pixel, copy C bytes at offset.
 * This is sequential access — much faster than NCHW interleave.
 */

#include "nn2_int8.h"
#include "../nn2_internal.h"
#include <string.h>
#include <stdlib.h>

/* Extract channel slice from NHWC: in[H,W,Cin_pad] → out[H,W,slice_pad]
 * Copies channels [ch_start..ch_start+ch_count) per pixel.
 * Threaded for large spatial. */

typedef struct { const int8_t* in; int8_t* out; int Cin_pad, ch_start, ch_count, out_pad, W; } SliceCtx;
static void slice_worker(void* arg, int start, int end) {
    SliceCtx* g = (SliceCtx*)arg;
    for (int i = start; i < end; i++) {
        memcpy(g->out + i * g->out_pad,
               g->in + i * g->Cin_pad + g->ch_start, g->ch_count);
        if (g->ch_count < g->out_pad)
            memset(g->out + i * g->out_pad + g->ch_count, 0, g->out_pad - g->ch_count);
    }
}

static void i8_channel_slice(const int8_t* in, int Cin_pad,
                              int ch_start, int ch_count,
                              int H, int W, int8_t* out, int out_pad)
{
    int spatial = H * W;
    SliceCtx ctx = {in, out, Cin_pad, ch_start, ch_count, out_pad, W};
    if (spatial >= 256)
        nn2_tp_parallel_for(slice_worker, &ctx, spatial, spatial / 12);
    else
        slice_worker(&ctx, 0, spatial);
}

typedef struct { const int8_t* src; int8_t* dst; int src_pad, dst_pad, ch_start, ch_count; } WriteCtx;
static void write_worker(void* arg, int start, int end) {
    WriteCtx* g = (WriteCtx*)arg;
    for (int i = start; i < end; i++)
        memcpy(g->dst + i * g->dst_pad + g->ch_start,
               g->src + i * g->src_pad, g->ch_count);
}

static void i8_channel_write(const int8_t* src, int src_pad, int ch_count,
                              int H, int W, int8_t* dst, int dst_pad, int ch_start)
{
    int spatial = H * W;
    WriteCtx ctx = {src, dst, src_pad, dst_pad, ch_start, ch_count};
    if (spatial >= 256)
        nn2_tp_parallel_for(write_worker, &ctx, spatial, spatial / 12);
    else
        write_worker(&ctx, 0, spatial);
}

/* INT8 shortcut add: a[spatial * c_pad] += b[spatial * c_pad], saturating */
extern void i8_residual_add(int8_t* a, const int8_t* b, int total,
                             float sa, float sb, float so);

/* Forward declare */
extern void i8_conv(const I8Conv* L, const int8_t* input, int H, int W,
                    int8_t* output, int8_t* scratch);

/* ============ C2f block ============ */

void i8_c2f(const I8C2f* blk, const int8_t* input, int Cin_pad,
            int H, int W, int8_t* output, int Cout_pad, int8_t* work)
{
    int spatial = H * W;
    int c = blk->c;
    int c_pad = C_PAD16(c);
    int n = blk->n;
    int total_c = (2 + n) * c;
    int total_pad = C_PAD16(total_c);

    /* Workspace layout:
     * [cv1_out: spatial * cv1.cout_pad]
     * [cat: spatial * total_pad]  ← concat buffer
     * [scratch: 4MB for sub-convs]
     */
    int cv1_out_pad = blk->cv1.cout_pad;
    int8_t* cv1_out = work;
    int8_t* cat = cv1_out + spatial * cv1_out_pad;
    int8_t* scratch = cat + spatial * total_pad;

    /* cv1: input → cv1_out [H,W,2c_pad] */
    i8_conv(&blk->cv1, input, H, W, cv1_out, scratch);

    /* Copy cv1 output channels to cat buffer:
     * cat[pixel][0..2c) = cv1_out[pixel][0..2c) */
    for (int i = 0; i < spatial; i++) {
        memcpy(cat + i * total_pad, cv1_out + i * cv1_out_pad, 2 * c);
        /* Zero-fill rest */
        if (2 * c < total_pad)
            memset(cat + i * total_pad + 2 * c, 0, total_pad - 2 * c);
    }

    /* Bottleneck iterations */
    int bn_ch_offset = c;  /* where the current bottleneck input starts in cat */

    for (int bi = 0; bi < n; bi++) {
        /* Extract bn_in: cat channels [bn_ch_offset .. bn_ch_offset+c) */
        int8_t* bn_in = scratch;
        i8_channel_slice(cat, total_pad, bn_ch_offset, c, H, W, bn_in, c_pad);

        /* bn.cv1: c → c, 3x3 */
        int8_t* bn_mid = scratch + spatial * c_pad;
        i8_conv(&blk->m[bi].cv1, bn_in, H, W, bn_mid, scratch + 2 * spatial * c_pad);

        /* bn.cv2: c → c, 3x3 */
        int8_t* bn_out = scratch + 2 * spatial * c_pad;
        i8_conv(&blk->m[bi].cv2, bn_mid, H, W, bn_out, scratch + 3 * spatial * c_pad);

        /* Shortcut: bn_out += bn_in */
        if (blk->shortcut) {
            i8_residual_add(bn_out, bn_in, spatial * c_pad,
                            blk->m[bi].cv2.out_scale,
                            blk->m[bi].cv1.act_scale,
                            blk->m[bi].cv2.out_scale);
        }

        /* Write bn_out to cat at channel position (2+bi)*c */
        int cat_offset = (2 + bi) * c;
        i8_channel_write(bn_out, c_pad, c, H, W, cat, total_pad, cat_offset);

        bn_ch_offset = cat_offset;
    }

    /* cv2: cat[total_c channels] → output */
    /* Extract active channels from cat into contiguous buffer for cv2 */
    int cv2_in_pad = C_PAD16(total_c);
    int8_t* cv2_in = scratch;
    if (total_pad == cv2_in_pad && total_c == total_pad) {
        cv2_in = cat; /* no copy needed */
    } else {
        for (int i = 0; i < spatial; i++)
            memcpy(cv2_in + i * cv2_in_pad, cat + i * total_pad, total_c);
    }

    i8_conv(&blk->cv2, cv2_in, H, W, output, scratch + spatial * cv2_in_pad);
}
