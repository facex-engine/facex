/*
 * forward_spec.c — Specialized forward pass for YOLOv8n@320.
 *
 * Key optimization: direct GEMM calls with hardcoded dimensions,
 * skip conv2d dispatcher overhead (branch checks, im2col dispatch decisions).
 *
 * For 1x1 convs: call packed GEMM directly (no im2col, no dispatch check).
 * For 3x3 convs with spatial >= 2500: use implicit conv (already fast).
 * For 3x3 convs with spatial < 2500: inline im2col + GEMM.
 *
 * Also: single-thread GEMM for layers with few N-tiles (avoid dispatch overhead).
 */

#include "../nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double spec_now(void) {
    static LARGE_INTEGER freq; static int init=0;
    if(!init){QueryPerformanceFrequency(&freq);init=1;}
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart/(double)freq.QuadPart;
}
#endif

/* Direct 1x1 conv: just GEMM, no dispatch overhead */
static inline void conv1x1_direct(const ConvLayer* L, const float* in,
                                    int spatial, float* out)
{
    nn2_sgemm_bias_act_packed(L->cout, L->cin, spatial,
                               L->w_packed, in, out, spatial, L->b, L->act);
}

/* Direct 3x3 s=1 conv: im2col (single-threaded for small) + GEMM */
static inline void conv3x3s1_direct(const ConvLayer* L, const float* in,
                                      int H, int W, float* out, float* im2col)
{
    int spatial = H * W;
    /* Implicit conv for large spatial */
    if (L->w_packed && spatial >= 2500) {
        nn2_conv3x3_s1_implicit(in, L->w, L->b, L->cin, L->cout, H, W,
                                 L->act, out, L->w_packed);
        return;
    }
    int K = L->cin * 9;
    nn2_im2col(in, L->cin, H, W, 3, 3, 1, 1, im2col);
    nn2_sgemm_bias_act_packed(L->cout, K, spatial,
                               L->w_packed, im2col, out, spatial, L->b, L->act);
}

/* Direct 3x3 s=2 conv */
static inline void conv3x3s2_direct(const ConvLayer* L, const float* in,
                                      int H, int W, float* out, float* im2col)
{
    int Hout = (H + 1) / 2, Wout = (W + 1) / 2;
    int spatial = Hout * Wout;
    int K = L->cin * 9;
    nn2_im2col(in, L->cin, H, W, 3, 3, 2, 1, im2col);
    nn2_sgemm_bias_act_packed(L->cout, K, spatial,
                               L->w_packed, im2col, out, spatial, L->b, L->act);
}

/* Direct 3x3 s=2 + fused residual */
static inline void conv3x3s1_res_direct(const ConvLayer* L, const float* in,
                                          int H, int W, float* out, float* im2col,
                                          const float* residual)
{
    int spatial = H * W;
    int K = L->cin * 9;

    if (L->w_packed && spatial >= 2500) {
        nn2_conv3x3_s1_implicit(in, L->w, L->b, L->cin, L->cout, H, W,
                                 L->act, out, L->w_packed);
        for (int i = 0; i < L->cout * spatial; i++) out[i] += residual[i];
        return;
    }

    nn2_im2col(in, L->cin, H, W, 3, 3, 1, 1, im2col);
    nn2_sgemm_bias_act_res_packed(L->cout, K, spatial,
                                   L->w_packed, im2col, out, spatial,
                                   L->b, L->act, residual, spatial);
}

/* Specialized C2f block: all paths known at compile time */
static void c2f_spec(const C2fBlock* blk, const float* in, int H, int W,
                      float* out, float* im2col, float* tmp)
{
    int spatial = H * W;
    int c = blk->c;
    int total_ch = (2 + blk->n) * c;
    float* cat = tmp;
    float* scratch = tmp + total_ch * spatial;

    /* cv1: always 1x1 */
    conv1x1_direct(&blk->cv1, in, spatial, cat);

    float* bn_in = cat + c * spatial;
    for (int i = 0; i < blk->n; i++) {
        /* bn.cv1: always 3x3 s=1 */
        conv3x3s1_direct(&blk->m[i].cv1, bn_in, H, W, scratch, im2col);

        /* bn.cv2: 3x3 s=1 + residual */
        float* bn_out = cat + (2 + i) * c * spatial;
        if (blk->shortcut) {
            conv3x3s1_res_direct(&blk->m[i].cv2, scratch, H, W, bn_out, im2col, bn_in);
        } else {
            conv3x3s1_direct(&blk->m[i].cv2, scratch, H, W, bn_out, im2col);
        }
        bn_in = bn_out;
    }

    /* cv2: always 1x1 */
    conv1x1_direct(&blk->cv2, cat, spatial, out);
}

/* ============ Specialized forward pass ============ */

int nn2_forward_spec(NN2* ctx, const float* input, float* bbox_out, float* cls_out)
{
    int S = ctx->input_size;
    float* ws = ctx->workspace;

    float* im2col = ws; ws += 4*1024*1024;
    float* buf0 = ws; ws += 4*1024*1024;
    float* buf1 = ws; ws += 4*1024*1024;
    ws += 64*(S/8)*(S/8);  /* skip save slots — we use direct pointers */
    ws += 128*(S/16)*(S/16);
    ws += 256*(S/32)*(S/32);
    ws += 128*(S/16)*(S/16);
    float* c2f_tmp = ws;

    int h = S, w = S;
    int h0=S/2,w0=S/2, h1=S/4,w1=S/4, h2=S/8,w2=S/8;
    int h3=S/16,w3=S/16, h4=S/32,w4=S/32;
    float *cur = buf0, *nxt = buf1;
    #define SWAP() { float* t_=cur; cur=nxt; nxt=t_; }

    /* === Backbone — direct calls, no dispatcher === */
    conv3x3s2_direct(&ctx->bb_conv[0], input, h, w, cur, im2col);
    conv3x3s2_direct(&ctx->bb_conv[1], cur, h0, w0, nxt, im2col); SWAP();
    c2f_spec(&ctx->bb_c2f[0], cur, h1, w1, nxt, im2col, c2f_tmp); SWAP();

    conv3x3s2_direct(&ctx->bb_conv[2], cur, h1, w1, nxt, im2col); SWAP();
    c2f_spec(&ctx->bb_c2f[1], cur, h2, w2, nxt, im2col, c2f_tmp); SWAP();
    /* save P3 */
    float* save_p3 = ctx->workspace + 12*1024*1024; /* reuse save slot area */
    memcpy(save_p3, cur, 64*h2*w2*sizeof(float));

    conv3x3s2_direct(&ctx->bb_conv[3], cur, h2, w2, nxt, im2col); SWAP();
    c2f_spec(&ctx->bb_c2f[2], cur, h3, w3, nxt, im2col, c2f_tmp); SWAP();
    float* save_p4 = save_p3 + 64*h2*w2;
    memcpy(save_p4, cur, 128*h3*w3*sizeof(float));

    conv3x3s2_direct(&ctx->bb_conv[4], cur, h3, w3, nxt, im2col); SWAP();
    c2f_spec(&ctx->bb_c2f[3], cur, h4, w4, nxt, im2col, c2f_tmp); SWAP();

    /* SPPF */
    conv1x1_direct(&ctx->sppf_cv1, cur, h4*w4, nxt); SWAP();
    {
        int sp = h4 * w4;
        float* sppf_in = cur;
        float* cat = c2f_tmp;
        float* mp1 = cat + 512*sp, *mp2 = mp1+128*sp, *mp3 = mp2+128*sp;
        nn2_maxpool2d(sppf_in, 128, h4, w4, 5, 1, 2, mp1);
        nn2_maxpool2d(mp1, 128, h4, w4, 5, 1, 2, mp2);
        nn2_maxpool2d(mp2, 128, h4, w4, 5, 1, 2, mp3);
        memcpy(cat, sppf_in, 128*sp*sizeof(float));
        memcpy(cat+128*sp, mp1, 128*sp*sizeof(float));
        memcpy(cat+256*sp, mp2, 128*sp*sizeof(float));
        memcpy(cat+384*sp, mp3, 128*sp*sizeof(float));
        conv1x1_direct(&ctx->sppf_cv2, cat, sp, nxt); SWAP();
    }
    float* save_p5 = save_p4 + 128*h3*w3;
    memcpy(save_p5, cur, 256*h4*w4*sizeof(float));

    /* === Neck === */
    {
        float* up = c2f_tmp;
        nn2_upsample_2x(cur, 256, h4, w4, up);
        nn2_concat(up, 256, save_p4, 128, h3, w3, nxt);
        SWAP();
    }
    c2f_spec(&ctx->nk_c2f[0], cur, h3, w3, nxt, im2col, c2f_tmp); SWAP();
    float* save_n3 = save_p5 + 256*h4*w4;
    memcpy(save_n3, cur, 128*h3*w3*sizeof(float));

    {
        float* up = c2f_tmp;
        nn2_upsample_2x(cur, 128, h3, w3, up);
        nn2_concat(up, 128, save_p3, 64, h2, w2, nxt);
        SWAP();
    }
    float* feat_n2 = save_p3; /* reuse */
    c2f_spec(&ctx->nk_c2f[1], cur, h2, w2, feat_n2, im2col, c2f_tmp);

    conv3x3s2_direct(&ctx->nk_conv[0], feat_n2, h2, w2, cur, im2col);
    { nn2_concat(cur, 64, save_n3, 128, h3, w3, nxt); SWAP(); }
    float* feat_n4 = save_p4;
    c2f_spec(&ctx->nk_c2f[2], cur, h3, w3, feat_n4, im2col, c2f_tmp);

    conv3x3s2_direct(&ctx->nk_conv[1], feat_n4, h3, w3, cur, im2col);
    { nn2_concat(cur, 128, save_p5, 256, h4, w4, nxt); SWAP(); }
    float* feat_n5 = save_p5;
    c2f_spec(&ctx->nk_c2f[3], cur, h4, w4, feat_n5, im2col, c2f_tmp);

    /* === Head — direct calls === */
    float* feats[3] = {feat_n2, feat_n4, feat_n5};
    int feat_h[] = {h2, h3, h4};
    int feat_w[] = {w2, w3, w4};
    int total_anchors = h2*w2 + h3*w3 + h4*w4;
    int bbox_ch = 4 * NN2_REG_MAX;
    int cls_ch = ctx->num_classes;
    int offset = 0;

    for (int s = 0; s < 3; s++) {
        int fh = feat_h[s], fw = feat_w[s], sp = fh * fw;
        float* tmp = c2f_tmp;

        float* bx1 = tmp;
        conv3x3s1_direct(&ctx->head[s].bbox[0], feats[s], fh, fw, bx1, im2col);
        float* bx2 = bx1 + ctx->head[s].bbox[0].cout * sp;
        conv3x3s1_direct(&ctx->head[s].bbox[1], bx1, fh, fw, bx2, im2col);
        float* bx3 = bx2 + ctx->head[s].bbox[1].cout * sp;
        conv1x1_direct(&ctx->head[s].bbox[2], bx2, sp, bx3);

        float* cx1 = bx3 + bbox_ch * sp;
        conv3x3s1_direct(&ctx->head[s].cls[0], feats[s], fh, fw, cx1, im2col);
        float* cx2 = cx1 + ctx->head[s].cls[0].cout * sp;
        conv3x3s1_direct(&ctx->head[s].cls[1], cx1, fh, fw, cx2, im2col);
        float* cx3 = cx2 + ctx->head[s].cls[1].cout * sp;
        conv1x1_direct(&ctx->head[s].cls[2], cx2, sp, cx3);

        for (int c = 0; c < bbox_ch; c++)
            memcpy(bbox_out + c*total_anchors + offset, bx3 + c*sp, sp*sizeof(float));
        for (int c = 0; c < cls_ch; c++)
            memcpy(cls_out + c*total_anchors + offset, cx3 + c*sp, sp*sizeof(float));
        offset += sp;
    }

    #undef SWAP
    return total_anchors;
}

/* ============ Benchmark ============ */

int main(int argc, char** argv)
{
    const char* wpath = "weights/yolov8n_320.bin";
    if (argc > 1) wpath = argv[1];

    NN2* ctx = nn2_init(wpath, 0);
    if (!ctx) { fprintf(stderr, "Failed\n"); return 1; }
    int S = ctx->input_size;

    float* input = (float*)calloc(3*S*S, sizeof(float));
    for (int i = 0; i < 3*S*S; i++) input[i] = 0.5f;
    int ta = (S/8)*(S/8)+(S/16)*(S/16)+(S/32)*(S/32);
    float* bbox = (float*)calloc(64*ta, sizeof(float));
    float* cls = (float*)calloc(80*ta, sizeof(float));

    /* Warmup both paths */
    for (int i = 0; i < 3; i++) {
        nn2_forward(ctx, input, bbox, cls);
        nn2_forward_spec(ctx, input, bbox, cls);
    }

    int RUNS = 50;

    /* Benchmark original */
    nn2_tp_reset_dispatch_count();
    double t0 = spec_now();
    for (int i = 0; i < RUNS; i++)
        nn2_forward(ctx, input, bbox, cls);
    double orig_ms = (spec_now() - t0) / RUNS * 1000.0;
    int orig_disp = nn2_tp_dispatch_count() / RUNS;

    /* Benchmark specialized */
    nn2_tp_reset_dispatch_count();
    t0 = spec_now();
    for (int i = 0; i < RUNS; i++)
        nn2_forward_spec(ctx, input, bbox, cls);
    double spec_ms = (spec_now() - t0) / RUNS * 1000.0;
    int spec_disp = nn2_tp_dispatch_count() / RUNS;

    printf("=== Specialized Forward Pass ===\n");
    printf("  Original:    %.2f ms  (%d dispatches)\n", orig_ms, orig_disp);
    printf("  Specialized: %.2f ms  (%d dispatches)\n", spec_ms, spec_disp);
    printf("  Speedup:     %.2fx\n", orig_ms / spec_ms);
    printf("  Dispatch reduction: %d → %d (%.0f%% fewer)\n",
           orig_disp, spec_disp, 100.0*(1.0 - (double)spec_disp/orig_disp));

    free(input); free(bbox); free(cls);
    nn2_free(ctx);
    return 0;
}
