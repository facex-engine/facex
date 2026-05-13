/*
 * net.c — YOLOv8n hardcoded forward pass + weight loading.
 *
 * Memory strategy:
 *   - 2 ping-pong buffers (buf[0], buf[1]) for sequential layer I/O
 *   - 1 reusable im2col buffer (largest needed: ~4M floats)
 *   - Dedicated save slots for skip connections (P3, P4, P5, N3)
 *   - Temp area for C2f internals
 */

#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* No explicit SIMD here — compiler auto-vectorizes with -O3 */

/* ============ Weight file format ============ */

static float* read_tensor(FILE* f, int* out_size)
{
    uint32_t sz;
    if (fread(&sz, 4, 1, f) != 1) return NULL;
    float* data = (float*)malloc(sz * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), sz, f) != sz) { free(data); return NULL; }
    *out_size = (int)sz;
    return data;
}

static int load_conv(FILE* f, ConvLayer* L, int cout, int cin, int k, int stride, int pad, int act)
{
    int wsz, bsz;
    L->w = read_tensor(f, &wsz);
    L->b = read_tensor(f, &bsz);
    if (!L->w || !L->b) return -1;
    L->cout = cout; L->cin = cin; L->k = k;
    L->stride = stride; L->pad = pad; L->act = act;
    return 0;
}

static int load_c2f(FILE* f, C2fBlock* blk, int cin, int cout, int n, int shortcut)
{
    int c = cout / 2;
    blk->n = n;
    blk->c = c;
    blk->shortcut = shortcut;

    if (load_conv(f, &blk->cv1, 2 * c, cin, 1, 1, 0, 1)) return -1;
    for (int i = 0; i < n; i++) {
        if (load_conv(f, &blk->m[i].cv1, c, c, 3, 1, 1, 1)) return -1;
        if (load_conv(f, &blk->m[i].cv2, c, c, 3, 1, 1, 1)) return -1;
    }
    if (load_conv(f, &blk->cv2, cout, (2 + n) * c, 1, 1, 0, 1)) return -1;
    return 0;
}

int nn2_load_weights(NN2* ctx, const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nn2: cannot open %s\n", path); return -1; }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "NN2W", 4) != 0) {
        fprintf(stderr, "nn2: invalid weight file\n");
        fclose(f); return -1;
    }

    uint32_t nc, isz, ntensors;
    fread(&nc, 4, 1, f);
    fread(&isz, 4, 1, f);
    fread(&ntensors, 4, 1, f);
    ctx->num_classes = (int)nc;
    ctx->input_size = (int)isz;

    int ch[] = {3, 16, 32, 64, 128, 256};
    for (int i = 0; i < 5; i++)
        if (load_conv(f, &ctx->bb_conv[i], ch[i+1], ch[i], 3, 2, 1, 1)) goto fail;

    int c2f_ch[] = {32, 64, 128, 256};
    int c2f_n[]  = {1, 2, 2, 1};
    for (int i = 0; i < 4; i++)
        if (load_c2f(f, &ctx->bb_c2f[i], c2f_ch[i], c2f_ch[i], c2f_n[i], 1)) goto fail;

    if (load_conv(f, &ctx->sppf_cv1, 128, 256, 1, 1, 0, 1)) goto fail;
    if (load_conv(f, &ctx->sppf_cv2, 256, 512, 1, 1, 0, 1)) goto fail;

    int nk_cin[]  = {384, 192, 192, 384};
    int nk_cout[] = {128, 64, 128, 256};
    for (int i = 0; i < 4; i++)
        if (load_c2f(f, &ctx->nk_c2f[i], nk_cin[i], nk_cout[i], 1, 0)) goto fail;

    if (load_conv(f, &ctx->nk_conv[0], 64, 64, 3, 2, 1, 1)) goto fail;
    if (load_conv(f, &ctx->nk_conv[1], 128, 128, 3, 2, 1, 1)) goto fail;

    int head_ch[] = {64, 128, 256};
    int bbox_c2 = 64;
    int min_nc_100 = (int)nc < 100 ? (int)nc : 100;
    int cls_c3 = head_ch[0] > min_nc_100 ? head_ch[0] : min_nc_100;

    for (int s = 0; s < 3; s++) {
        if (load_conv(f, &ctx->head[s].bbox[0], bbox_c2, head_ch[s], 3, 1, 1, 1)) goto fail;
        if (load_conv(f, &ctx->head[s].bbox[1], bbox_c2, bbox_c2, 3, 1, 1, 1)) goto fail;
        if (load_conv(f, &ctx->head[s].bbox[2], 4 * NN2_REG_MAX, bbox_c2, 1, 1, 0, 0)) goto fail;
        if (load_conv(f, &ctx->head[s].cls[0], cls_c3, head_ch[s], 3, 1, 1, 1)) goto fail;
        if (load_conv(f, &ctx->head[s].cls[1], cls_c3, cls_c3, 3, 1, 1, 1)) goto fail;
        if (load_conv(f, &ctx->head[s].cls[2], (int)nc, cls_c3, 1, 1, 0, 0)) goto fail;
    }

    fclose(f);
    return 0;

fail:
    fprintf(stderr, "nn2: failed to load weights\n");
    fclose(f);
    return -1;
}

/* ============ Memory layout ============
 * workspace is carved into fixed regions:
 *
 * [im2col_buf][buf_a][buf_b][save_p3][save_p4][save_p5][save_n3][c2f_tmp]
 *
 * im2col_buf: 4M floats (reused by every conv)
 * buf_a/buf_b: ping-pong, 4M floats each (max feature map)
 * save_*: skip connections, permanently allocated
 * c2f_tmp: scratch for C2f internals, 8M floats
 */

#define IM2COL_SIZE  (4 * 1024 * 1024)   /* 16MB */
#define BUF_SIZE     (4 * 1024 * 1024)   /* 16MB each */
#define C2F_TMP_SIZE (8 * 1024 * 1024)   /* 32MB */

typedef struct {
    float* im2col;    /* reusable im2col buffer */
    float* buf[2];    /* ping-pong buffers */
    int    cur;       /* current output buffer index (0 or 1) */
    float* save[4];   /* P3, P4, P5, N3 */
    float* c2f_tmp;   /* scratch for C2f internals */
} MemPool;

static void conv_run(const ConvLayer* L, const float* in, int H, int W,
                     float* out, float* im2col)
{
    nn2_conv2d(L, in, H, W, out, im2col);
}

static void c2f_run(const C2fBlock* blk, const float* in, int H, int W,
                    float* out, float* im2col, float* tmp)
{
    int spatial = H * W;
    int c = blk->c;

    int total_ch = (2 + blk->n) * c;

    /* Allocate concat buffer FIRST — write all chunks directly into it */
    float* cat = tmp;
    float* scratch = tmp + total_ch * spatial;

    /* cv1 output goes directly into cat[0..2c*spatial-1] */
    conv_run(&blk->cv1, in, H, W, cat, im2col);
    /* cat[0..c-1] = chunk0 (identity branch) */
    /* cat[c..2c-1] = chunk1 (first bottleneck input) */

    float* bn_in = cat + c * spatial;
    for (int i = 0; i < blk->n; i++) {
        /* Bottleneck cv1 → scratch (temporary) */
        conv_run(&blk->m[i].cv1, bn_in, H, W, scratch, im2col);

        /* Bottleneck cv2 → directly into cat[(2+i)*c] position */
        float* bn_out = cat + (2 + i) * c * spatial;
        if (blk->shortcut && blk->m[i].cv2.w_packed && spatial < 2500) {
            /* Fused: im2col + GEMM+bias+act+residual in one store pass */
            const ConvLayer* L = &blk->m[i].cv2;
            int K_gemm = L->cin * L->k * L->k;
            nn2_im2col(scratch, L->cin, H, W, L->k, L->k, L->stride, L->pad, im2col);
            nn2_sgemm_bias_act_res_packed(L->cout, K_gemm, spatial,
                                           L->w_packed, im2col,
                                           bn_out, spatial, L->b, L->act,
                                           bn_in, spatial);
        } else {
            conv_run(&blk->m[i].cv2, scratch, H, W, bn_out, im2col);
            if (blk->shortcut)
                for (int j = 0; j < c * spatial; j++) bn_out[j] += bn_in[j];
        }
        bn_in = bn_out;
    }

    /* cat is already contiguous — no memcpy needed! */
    conv_run(&blk->cv2, cat, H, W, out, im2col);
}

/* ============ Forward pass ============ */

int nn2_forward(NN2* ctx, const float* input, float* bbox_out, float* cls_out)
{
    int S = ctx->input_size;
    float* ws = ctx->workspace;

    /* Carve memory regions */
    MemPool mp;
    mp.im2col  = ws;                          ws += IM2COL_SIZE;
    mp.buf[0]  = ws;                          ws += BUF_SIZE;
    mp.buf[1]  = ws;                          ws += BUF_SIZE;
    mp.save[0] = ws; ws += 64 * (S/8) * (S/8);     /* P3: 64ch @ S/8 */
    mp.save[1] = ws; ws += 128 * (S/16) * (S/16);   /* P4: 128ch @ S/16 */
    mp.save[2] = ws; ws += 256 * (S/32) * (S/32);   /* P5: 256ch @ S/32 */
    mp.save[3] = ws; ws += 128 * (S/16) * (S/16);   /* N3: 128ch @ S/16 */
    mp.c2f_tmp = ws; /* rest is C2f scratch */
    mp.cur = 0;

    #define CUR  mp.buf[mp.cur]
    #define NEXT mp.buf[1 - mp.cur]
    #define SWAP() mp.cur = 1 - mp.cur

    int h = S, w = S;

    /* === Backbone === */

    /* Conv0: 3→16, s=2 → S/2 */
    int h0 = S/2, w0 = S/2;
    conv_run(&ctx->bb_conv[0], input, h, w, CUR, mp.im2col);
    /* CUR = 16ch @ h0×w0 */

    /* Conv1: 16→32, s=2 → S/4 */
    int h1 = S/4, w1 = S/4;
    conv_run(&ctx->bb_conv[1], CUR, h0, w0, NEXT, mp.im2col);
    SWAP();

    /* C2f_0: 32→32, n=1 */
    c2f_run(&ctx->bb_c2f[0], CUR, h1, w1, NEXT, mp.im2col, mp.c2f_tmp);
    SWAP();

    /* Conv2: 32→64, s=2 → S/8 */
    int h2 = S/8, w2 = S/8;
    conv_run(&ctx->bb_conv[2], CUR, h1, w1, NEXT, mp.im2col);
    SWAP();

    /* C2f_1: 64→64, n=2 → P3 */
    c2f_run(&ctx->bb_c2f[1], CUR, h2, w2, NEXT, mp.im2col, mp.c2f_tmp);
    SWAP();
    memcpy(mp.save[0], CUR, 64 * h2 * w2 * sizeof(float)); /* save P3 */

    /* Conv3: 64→128, s=2 → S/16 */
    int h3 = S/16, w3 = S/16;
    conv_run(&ctx->bb_conv[3], CUR, h2, w2, NEXT, mp.im2col);
    SWAP();

    /* C2f_2: 128→128, n=2 → P4 */
    c2f_run(&ctx->bb_c2f[2], CUR, h3, w3, NEXT, mp.im2col, mp.c2f_tmp);
    SWAP();
    memcpy(mp.save[1], CUR, 128 * h3 * w3 * sizeof(float)); /* save P4 */

    /* Conv4: 128→256, s=2 → S/32 */
    int h4 = S/32, w4 = S/32;
    conv_run(&ctx->bb_conv[4], CUR, h3, w3, NEXT, mp.im2col);
    SWAP();

    /* C2f_3: 256→256, n=1 */
    c2f_run(&ctx->bb_c2f[3], CUR, h4, w4, NEXT, mp.im2col, mp.c2f_tmp);
    SWAP();

    /* SPPF: cv1(256→128) → maxpool×3 → concat(512) → cv2(512→256) */
    conv_run(&ctx->sppf_cv1, CUR, h4, w4, NEXT, mp.im2col);
    SWAP();
    /* CUR = 128ch @ h4×w4 */
    {
        int sp = h4 * w4;
        float* sppf_in = CUR;
        float* cat = mp.c2f_tmp; /* concat output: 512ch */

        float* mp1 = cat + 512 * sp;
        float* mp2 = mp1 + 128 * sp;
        float* mp3 = mp2 + 128 * sp;

        nn2_maxpool2d(sppf_in, 128, h4, w4, 5, 1, 2, mp1);
        nn2_maxpool2d(mp1,     128, h4, w4, 5, 1, 2, mp2);
        nn2_maxpool2d(mp2,     128, h4, w4, 5, 1, 2, mp3);

        memcpy(cat,              sppf_in, 128 * sp * sizeof(float));
        memcpy(cat + 128 * sp,  mp1,     128 * sp * sizeof(float));
        memcpy(cat + 256 * sp,  mp2,     128 * sp * sizeof(float));
        memcpy(cat + 384 * sp,  mp3,     128 * sp * sizeof(float));

        conv_run(&ctx->sppf_cv2, cat, h4, w4, NEXT, mp.im2col);
        SWAP();
    }
    /* CUR = 256ch @ h4×w4 = P5 */
    memcpy(mp.save[2], CUR, 256 * h4 * w4 * sizeof(float)); /* save P5 */

    /* === Neck === */

    /* Upsample P5 (h4→h3) + concat P4 → 384ch */
    {
        float* up = mp.c2f_tmp;
        nn2_upsample_2x(CUR, 256, h4, w4, up);  /* 256ch @ h3×w3 */
        float* cat = NEXT;
        nn2_concat(up, 256, mp.save[1], 128, h3, w3, cat); /* 384ch */
        SWAP();
    }

    /* C2f neck_0: 384→128 → N3 */
    c2f_run(&ctx->nk_c2f[0], CUR, h3, w3, NEXT, mp.im2col, mp.c2f_tmp);
    SWAP();
    memcpy(mp.save[3], CUR, 128 * h3 * w3 * sizeof(float)); /* save N3 */

    /* Upsample N3 (h3→h2) + concat P3 → 192ch */
    {
        float* up = mp.c2f_tmp;
        nn2_upsample_2x(CUR, 128, h3, w3, up);  /* 128ch @ h2×w2 */
        float* cat = NEXT;
        nn2_concat(up, 128, mp.save[0], 64, h2, w2, cat); /* 192ch */
        SWAP();
    }

    /* C2f neck_1: 192→64 → N2 (scale 0 feature) */
    float* feat_n2 = mp.save[0]; /* reuse P3 slot (no longer needed) */
    c2f_run(&ctx->nk_c2f[1], CUR, h2, w2, feat_n2, mp.im2col, mp.c2f_tmp);

    /* Downsample N2 + concat N3 → 192ch */
    conv_run(&ctx->nk_conv[0], feat_n2, h2, w2, CUR, mp.im2col); /* 64ch @ h3 */
    {
        float* cat = NEXT;
        nn2_concat(CUR, 64, mp.save[3], 128, h3, w3, cat); /* 192ch */
        SWAP();
    }

    /* C2f neck_2: 192→128 → N4 (scale 1 feature) */
    float* feat_n4 = mp.save[1]; /* reuse P4 slot */
    c2f_run(&ctx->nk_c2f[2], CUR, h3, w3, feat_n4, mp.im2col, mp.c2f_tmp);

    /* Downsample N4 + concat P5 → 384ch */
    conv_run(&ctx->nk_conv[1], feat_n4, h3, w3, CUR, mp.im2col); /* 128ch @ h4 */
    {
        float* cat = NEXT;
        nn2_concat(CUR, 128, mp.save[2], 256, h4, w4, cat); /* 384ch */
        SWAP();
    }

    /* C2f neck_3: 384→256 → N5 (scale 2 feature) */
    float* feat_n5 = mp.save[2]; /* reuse P5 slot */
    c2f_run(&ctx->nk_c2f[3], CUR, h4, w4, feat_n5, mp.im2col, mp.c2f_tmp);

    /* === Head === */
    float* feats[3] = {feat_n2, feat_n4, feat_n5};
    int feat_h[3] = {h2, h3, h4};
    int feat_w[3] = {w2, w3, w4};

    int total_anchors = h2*w2 + h3*w3 + h4*w4;
    int bbox_ch = 4 * NN2_REG_MAX;
    int cls_ch  = ctx->num_classes;

    int offset = 0;
    for (int s = 0; s < 3; s++) {
        int fh = feat_h[s], fw = feat_w[s];
        int spatial = fh * fw;
        float* tmp = mp.c2f_tmp;

        /* Bbox: conv3x3 → conv3x3 → conv1x1 */
        float* bx1 = tmp;
        conv_run(&ctx->head[s].bbox[0], feats[s], fh, fw, bx1, mp.im2col);
        float* bx2 = tmp + ctx->head[s].bbox[0].cout * spatial;
        conv_run(&ctx->head[s].bbox[1], bx1, fh, fw, bx2, mp.im2col);
        float* bx3 = bx2 + ctx->head[s].bbox[1].cout * spatial;
        conv_run(&ctx->head[s].bbox[2], bx2, fh, fw, bx3, mp.im2col);

        /* Cls: conv3x3 → conv3x3 → conv1x1 */
        float* cx1 = bx3 + bbox_ch * spatial;
        conv_run(&ctx->head[s].cls[0], feats[s], fh, fw, cx1, mp.im2col);
        float* cx2 = cx1 + ctx->head[s].cls[0].cout * spatial;
        conv_run(&ctx->head[s].cls[1], cx1, fh, fw, cx2, mp.im2col);
        float* cx3 = cx2 + ctx->head[s].cls[1].cout * spatial;
        conv_run(&ctx->head[s].cls[2], cx2, fh, fw, cx3, mp.im2col);

        /* Copy to output arrays (CHW interleaved across scales) */
        for (int c = 0; c < bbox_ch; c++)
            memcpy(bbox_out + c * total_anchors + offset,
                   bx3 + c * spatial, spatial * sizeof(float));
        for (int c = 0; c < cls_ch; c++)
            memcpy(cls_out + c * total_anchors + offset,
                   cx3 + c * spatial, spatial * sizeof(float));

        offset += spatial;
    }

    #undef CUR
    #undef NEXT
    #undef SWAP
    return total_anchors;
}
