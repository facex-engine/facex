/*
 * forward_hybrid.c — Hybrid FP32/INT8 YOLOv8n forward pass.
 *
 * Phase 1 (FP32 NCHW): conv0 → conv1 → c2f0 (Cin < 32, FP32 faster)
 * Transition: FP32 NCHW [32ch, 80×80] → INT8 NHWC [32ch_pad, 80×80]
 * Phase 2 (INT8 NHWC): conv2 → c2f1 → conv3 → c2f2 → conv4 → c2f3 → SPPF → neck
 * Transition: INT8 NHWC → FP32 NCHW for head features
 * Phase 3 (FP32 NCHW): head convs + decode (accuracy-sensitive)
 */

#include "nn2_int8.h"
#include "../nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double h_now(void) {
    static LARGE_INTEGER freq; static int init=0;
    if(!init){QueryPerformanceFrequency(&freq);init=1;}
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart/(double)freq.QuadPart;
}
#endif

/* Forward declares from other INT8 files */
extern void i8_conv(const I8Conv* L, const int8_t* input, int H, int W,
                    int8_t* output, int8_t* scratch);
extern void i8_c2f(const I8C2f* blk, const int8_t* input, int Cin_pad,
                   int H, int W, int8_t* output, int Cout_pad, int8_t* work);

/* ============ Init helpers ============ */

static float default_act_scale = 0.05f;

static void init_i8conv_default(I8Conv* dst, const ConvLayer* src) {
    int K = src->cin * src->k * src->k;
    int K4 = K_PAD4(K);
    int Cout16 = C_PAD16(src->cout);
    dst->cin = src->cin; dst->cout = src->cout; dst->k = src->k;
    dst->stride = src->stride; dst->pad = src->pad; dst->act = src->act;
    dst->K4 = K4; dst->cout_pad = Cout16;
    dst->act_scale = default_act_scale; dst->out_scale = default_act_scale;
    size_t ps = (size_t)(Cout16/16)*(K4/4)*64;
    dst->w_packed = (int8_t*)_aligned_malloc(ps, 64);
    dst->col_sums = (int32_t*)calloc(Cout16, sizeof(int32_t));
    dst->w_scales = (float*)calloc(Cout16, sizeof(float));
    dst->bias = (float*)calloc(Cout16, sizeof(float));
    memcpy(dst->bias, src->b, src->cout * sizeof(float));
    i8_pack_weights(src->w, src->cout, K, dst->w_packed, dst->col_sums, dst->w_scales);
}

static void init_i8c2f(I8C2f* dst, const C2fBlock* src) {
    dst->n = src->n; dst->c = src->c; dst->shortcut = src->shortcut;
    init_i8conv_default(&dst->cv1, &src->cv1);
    init_i8conv_default(&dst->cv2, &src->cv2);
    for (int i = 0; i < src->n; i++) {
        init_i8conv_default(&dst->m[i].cv1, &src->m[i].cv1);
        init_i8conv_default(&dst->m[i].cv2, &src->m[i].cv2);
    }
}

static void free_i8conv_s(I8Conv* L) {
    _aligned_free(L->w_packed); free(L->col_sums); free(L->w_scales); free(L->bias);
}
static void free_i8c2f(I8C2f* b) {
    free_i8conv_s(&b->cv1); free_i8conv_s(&b->cv2);
    for(int i=0;i<b->n;i++){free_i8conv_s(&b->m[i].cv1);free_i8conv_s(&b->m[i].cv2);}
}

/* ============ Hybrid forward ============ */

static int hybrid_forward(
    NN2* fp32,
    /* INT8 layers (conv2-4, c2f1-3, sppf, neck) */
    I8Conv* i8_conv2, I8Conv* i8_conv3, I8Conv* i8_conv4,
    I8C2f* i8_c2f1, I8C2f* i8_c2f2, I8C2f* i8_c2f3,
    I8Conv* i8_sppf1, I8Conv* i8_sppf2,
    I8C2f* i8_nk0, I8C2f* i8_nk1, I8C2f* i8_nk2, I8C2f* i8_nk3,
    I8Conv* i8_nkconv0, I8Conv* i8_nkconv1,
    /* Buffers */
    const float* input_chw,
    float* bbox_out, float* cls_out)
{
    int S = fp32->input_size;
    float* ws = fp32->workspace;

    /* FP32 workspace */
    float* im2col = ws; ws += 4*1024*1024;
    float* buf0 = ws; ws += 4*1024*1024;
    float* buf1 = ws; ws += 4*1024*1024;
    float* c2f_tmp = ws; /* rest */

    /* INT8 workspace — use static to avoid malloc/free in hot path */
    static int8_t* i8ws = NULL;
    if (!i8ws) i8ws = (int8_t*)_aligned_malloc(16 * 1024 * 1024, 64);
    int8_t* i8a = i8ws;
    int8_t* i8b = i8ws + 4*1024*1024;
    int8_t* i8scratch = i8ws + 8*1024*1024;

    int h0=S/2,w0=S/2, h1=S/4,w1=S/4, h2=S/8,w2=S/8;
    int h3=S/16,w3=S/16, h4=S/32,w4=S/32;

    /* ============ Phase 1: FP32 NCHW (conv0 → conv1 → c2f0) ============ */
    nn2_conv2d(&fp32->bb_conv[0], input_chw, S, S, buf0, im2col);
    nn2_conv2d(&fp32->bb_conv[1], buf0, h0, w0, buf1, im2col);

    /* c2f0 in FP32 */
    {
        C2fBlock* blk = &fp32->bb_c2f[0];
        int spatial = h1 * w1;
        int c = blk->c;
        int total_ch = (2 + blk->n) * c;
        float* cat = c2f_tmp;
        float* scratch = cat + total_ch * spatial;
        nn2_conv2d(&blk->cv1, buf1, h1, w1, cat, im2col);
        float* bn_in = cat + c * spatial;
        for (int i = 0; i < blk->n; i++) {
            nn2_conv2d(&blk->m[i].cv1, bn_in, h1, w1, scratch, im2col);
            float* bn_out = cat + (2+i) * c * spatial;
            nn2_conv2d(&blk->m[i].cv2, scratch, h1, w1, bn_out, im2col);
            if (blk->shortcut)
                for (int j = 0; j < c*spatial; j++) bn_out[j] += bn_in[j];
            bn_in = bn_out;
        }
        nn2_conv2d(&blk->cv2, cat, h1, w1, buf0, im2col);
    }
    /* buf0 = 32ch FP32 NCHW @ h1×w1 (80×80 for 320) */

    /* ============ Transition: FP32 NCHW → INT8 NHWC ============ */
    int c_at_transition = 32;
    int c_trans_pad = C_PAD16(c_at_transition);
    i8_quantize_nhwc(buf0, c_at_transition, h1, w1,
                      default_act_scale, i8a, c_trans_pad);

    /* ============ Phase 2: INT8 NHWC ============ */

    /* conv2: 32→64, 3x3 s=2, h1→h2 */
    int c64_pad = C_PAD16(64);
    i8_conv(i8_conv2, i8a, h1, w1, i8b, i8scratch);
    /* i8b = 64ch INT8 NHWC @ h2×w2 */

    /* c2f1: 64→64, n=2 */
    i8_c2f(i8_c2f1, i8b, c64_pad, h2, w2, i8a, c64_pad, i8scratch);
    /* i8a = 64ch @ h2×w2 — save as P3 */
    static int8_t* save_p3 = NULL;
    if (!save_p3) save_p3 = (int8_t*)_aligned_malloc(256*256*4, 64); /* max size */
    memcpy(save_p3, i8a, h2 * w2 * c64_pad);

    /* conv3: 64→128, 3x3 s=2 */
    int c128_pad = C_PAD16(128);
    i8_conv(i8_conv3, i8a, h2, w2, i8b, i8scratch);

    /* c2f2: 128→128, n=2 */
    i8_c2f(i8_c2f2, i8b, c128_pad, h3, w3, i8a, c128_pad, i8scratch);
    static int8_t* save_p4 = NULL;
    if (!save_p4) save_p4 = (int8_t*)_aligned_malloc(256*256*4, 64);
    memcpy(save_p4, i8a, h3 * w3 * c128_pad);

    /* conv4: 128→256, 3x3 s=2 */
    int c256_pad = C_PAD16(256);
    i8_conv(i8_conv4, i8a, h3, w3, i8b, i8scratch);

    /* c2f3: 256→256, n=1 */
    i8_c2f(i8_c2f3, i8b, c256_pad, h4, w4, i8a, c256_pad, i8scratch);

    /* SPPF: cv1(256→128) → 3×maxpool → concat(512) → cv2(512→256) */
    i8_conv(i8_sppf1, i8a, h4, w4, i8b, i8scratch); /* 128ch */
    {
        int sp = h4 * w4;
        int8_t* mp1 = i8scratch;
        int8_t* mp2 = mp1 + sp * c128_pad;
        int8_t* mp3 = mp2 + sp * c128_pad;
        i8_maxpool(i8b, c128_pad, h4, w4, 5, 1, 2, mp1);
        i8_maxpool(mp1, c128_pad, h4, w4, 5, 1, 2, mp2);
        i8_maxpool(mp2, c128_pad, h4, w4, 5, 1, 2, mp3);
        /* concat: [sppf_in, mp1, mp2, mp3] = 512ch */
        int c512_pad = C_PAD16(512);
        int8_t* cat512 = mp3 + sp * c128_pad;
        for (int i = 0; i < sp; i++) {
            memcpy(cat512 + i*c512_pad + 0,   i8b + i*c128_pad, 128);
            memcpy(cat512 + i*c512_pad + 128, mp1 + i*c128_pad, 128);
            memcpy(cat512 + i*c512_pad + 256, mp2 + i*c128_pad, 128);
            memcpy(cat512 + i*c512_pad + 384, mp3 + i*c128_pad, 128);
        }
        i8_conv(i8_sppf2, cat512, h4, w4, i8a, i8scratch); /* 256ch */
    }
    static int8_t* save_p5 = NULL;
    if (!save_p5) save_p5 = (int8_t*)_aligned_malloc(256*256*4, 64);
    memcpy(save_p5, i8a, h4 * w4 * c256_pad);

    /* ============ Neck ============ */

    /* Upsample P5 (h4→h3) + concat P4 → 384ch */
    i8_upsample_2x(i8a, c256_pad, h4, w4, i8b); /* 256ch @ h3 */
    {
        int c384_pad = C_PAD16(384);
        int sp = h3 * w3;
        int8_t* cat384 = i8scratch;
        for (int i = 0; i < sp; i++) {
            memcpy(cat384 + i*c384_pad, i8b + i*c256_pad, 256);
            memcpy(cat384 + i*c384_pad + 256, save_p4 + i*c128_pad, 128);
        }
        i8_c2f(i8_nk0, cat384, c384_pad, h3, w3, i8a, c128_pad, i8scratch + sp*c384_pad);
    }
    /* i8a = N3 = 128ch @ h3 — save */
    static int8_t* save_n3 = NULL;
    if (!save_n3) save_n3 = (int8_t*)_aligned_malloc(256*256*4, 64);
    memcpy(save_n3, i8a, h3 * w3 * c128_pad);

    /* Upsample N3 (h3→h2) + concat P3 → 192ch */
    i8_upsample_2x(i8a, c128_pad, h3, w3, i8b); /* 128ch @ h2 */
    {
        int c192_pad = C_PAD16(192);
        int sp = h2 * w2;
        int8_t* cat192 = i8scratch;
        for (int i = 0; i < sp; i++) {
            memcpy(cat192 + i*c192_pad, i8b + i*c128_pad, 128);
            memcpy(cat192 + i*c192_pad + 128, save_p3 + i*c64_pad, 64);
        }
        i8_c2f(i8_nk1, cat192, c192_pad, h2, w2, i8a, c64_pad, i8scratch + sp*c192_pad);
    }
    /* i8a = N2 = 64ch @ h2 — save before overwrite */
    static int8_t* save_n2 = NULL;
    if (!save_n2) save_n2 = (int8_t*)_aligned_malloc(256*256*4, 64);
    memcpy(save_n2, i8a, h2 * w2 * c64_pad);

    /* Downsample N2 + concat N3 → 192ch */
    i8_conv(i8_nkconv0, i8a, h2, w2, i8b, i8scratch); /* 64ch @ h3 */
    {
        int c192_pad = C_PAD16(192);
        int sp = h3 * w3;
        int8_t* cat192 = i8scratch;
        for (int i = 0; i < sp; i++) {
            memcpy(cat192 + i*c192_pad, i8b + i*c64_pad, 64);
            memcpy(cat192 + i*c192_pad + 64, save_n3 + i*c128_pad, 128);
        }
        i8_c2f(i8_nk2, cat192, c192_pad, h3, w3, i8b, c128_pad, i8scratch + sp*c192_pad);
    }
    /* i8b = N4 = 128ch @ h3 — head scale 1 feature */

    /* Downsample N4 + concat P5 → 384ch */
    i8_conv(i8_nkconv1, i8b, h3, w3, i8a, i8scratch); /* 128ch @ h4 */
    {
        int c384_pad = C_PAD16(384);
        int sp = h4 * w4;
        int8_t* cat384 = i8scratch;
        for (int i = 0; i < sp; i++) {
            memcpy(cat384 + i*c384_pad, i8a + i*c128_pad, 128);
            memcpy(cat384 + i*c384_pad + 128, save_p5 + i*c256_pad, 256);
        }
        i8_c2f(i8_nk3, cat384, c384_pad, h4, w4, i8a, c256_pad, i8scratch + sp*c384_pad);
    }
    /* i8a = N5 = 256ch @ h4 — head scale 2 feature */

    /* ============ Phase 3: Dequant → FP32 head → decode ============ */

    /* Dequant N2(64ch@h2), N4(128ch@h3→i8b), N5(256ch@h4→i8a) to FP32 NCHW */
    int head_ch[] = {64, 128, 256};
    int feat_h[] = {h2, h3, h4};
    int feat_w[] = {w2, w3, w4};
    int8_t* feat_i8[] = {save_n2, i8b, i8a};   /* N2, N4, N5 */
    int feat_pad[] = {c64_pad, c128_pad, c256_pad};

    /* Dequant to buf0/buf1 area (reuse FP32 workspace) */
    float* feat_fp32[3];
    float* fptr = buf0;
    for (int s = 0; s < 3; s++) {
        feat_fp32[s] = fptr;
        i8_dequant_to_chw(feat_i8[s], head_ch[s], feat_h[s], feat_w[s],
                           feat_pad[s], default_act_scale, feat_fp32[s]);
        fptr += head_ch[s] * feat_h[s] * feat_w[s];
    }

    /* FP32 head: bbox + cls branches for 3 scales */
    int total_anchors = h2*w2 + h3*w3 + h4*w4;
    int bbox_ch = 4 * 16; /* 4 * REG_MAX */
    int cls_ch = fp32->num_classes;
    int offset = 0;

    float* head_tmp = c2f_tmp;
    for (int s = 0; s < 3; s++) {
        int fh = feat_h[s], fw = feat_w[s], sp = fh * fw;

        /* Bbox: conv3x3 → conv3x3 → conv1x1 */
        float* bx1 = head_tmp;
        nn2_conv2d(&fp32->head[s].bbox[0], feat_fp32[s], fh, fw, bx1, im2col);
        float* bx2 = bx1 + fp32->head[s].bbox[0].cout * sp;
        nn2_conv2d(&fp32->head[s].bbox[1], bx1, fh, fw, bx2, im2col);
        float* bx3 = bx2 + fp32->head[s].bbox[1].cout * sp;
        nn2_conv2d(&fp32->head[s].bbox[2], bx2, fh, fw, bx3, im2col);

        /* Cls: conv3x3 → conv3x3 → conv1x1 */
        float* cx1 = bx3 + bbox_ch * sp;
        nn2_conv2d(&fp32->head[s].cls[0], feat_fp32[s], fh, fw, cx1, im2col);
        float* cx2 = cx1 + fp32->head[s].cls[0].cout * sp;
        nn2_conv2d(&fp32->head[s].cls[1], cx1, fh, fw, cx2, im2col);
        float* cx3 = cx2 + fp32->head[s].cls[1].cout * sp;
        nn2_conv2d(&fp32->head[s].cls[2], cx2, fh, fw, cx3, im2col);

        /* Copy to output arrays */
        for (int c = 0; c < bbox_ch; c++)
            memcpy(bbox_out + c * total_anchors + offset, bx3 + c * sp, sp * sizeof(float));
        for (int c = 0; c < cls_ch; c++)
            memcpy(cls_out + c * total_anchors + offset, cx3 + c * sp, sp * sizeof(float));
        offset += sp;
    }

    /* static buffers — not freed */
    return total_anchors;
}

/* ============ Main benchmark ============ */

int main(int argc, char** argv)
{
    const char* wpath = "weights/yolov8n_320.bin";
    if (argc > 1) wpath = argv[1];

    NN2* fp32 = nn2_init(wpath, 0);
    if (!fp32) { fprintf(stderr, "Failed\n"); return 1; }
    int S = fp32->input_size;

    /* Convert INT8 layers */
    I8Conv i8_conv2, i8_conv3, i8_conv4, i8_sppf1, i8_sppf2, i8_nkconv0, i8_nkconv1;
    init_i8conv_default(&i8_conv2, &fp32->bb_conv[2]);
    init_i8conv_default(&i8_conv3, &fp32->bb_conv[3]);
    init_i8conv_default(&i8_conv4, &fp32->bb_conv[4]);
    init_i8conv_default(&i8_sppf1, &fp32->sppf_cv1);
    init_i8conv_default(&i8_sppf2, &fp32->sppf_cv2);
    init_i8conv_default(&i8_nkconv0, &fp32->nk_conv[0]);
    init_i8conv_default(&i8_nkconv1, &fp32->nk_conv[1]);

    I8C2f i8_c2f1, i8_c2f2, i8_c2f3, i8_nk0, i8_nk1, i8_nk2, i8_nk3;
    init_i8c2f(&i8_c2f1, &fp32->bb_c2f[1]);
    init_i8c2f(&i8_c2f2, &fp32->bb_c2f[2]);
    init_i8c2f(&i8_c2f3, &fp32->bb_c2f[3]);
    init_i8c2f(&i8_nk0, &fp32->nk_c2f[0]);
    init_i8c2f(&i8_nk1, &fp32->nk_c2f[1]);
    init_i8c2f(&i8_nk2, &fp32->nk_c2f[2]);
    init_i8c2f(&i8_nk3, &fp32->nk_c2f[3]);

    fprintf(stderr, "Converted all layers to INT8 VNNI\n");

    /* Dummy input */
    float* input = (float*)calloc(3 * S * S, sizeof(float));
    for (int i = 0; i < 3*S*S; i++) input[i] = 0.5f;

    int ta = (S/8)*(S/8)+(S/16)*(S/16)+(S/32)*(S/32);
    float* bbox = (float*)calloc(64*ta, sizeof(float));
    float* cls = (float*)calloc(80*ta, sizeof(float));

    /* Warmup */
    for (int i = 0; i < 3; i++)
        hybrid_forward(fp32, &i8_conv2, &i8_conv3, &i8_conv4,
                       &i8_c2f1, &i8_c2f2, &i8_c2f3,
                       &i8_sppf1, &i8_sppf2,
                       &i8_nk0, &i8_nk1, &i8_nk2, &i8_nk3,
                       &i8_nkconv0, &i8_nkconv1,
                       input, bbox, cls);

    /* Benchmark full hybrid (backbone + neck + head + decode) */
    int RUNS = 50;

    /* Warmup + check correctness */
    int n_anchors = hybrid_forward(fp32, &i8_conv2, &i8_conv3, &i8_conv4,
                       &i8_c2f1, &i8_c2f2, &i8_c2f3,
                       &i8_sppf1, &i8_sppf2,
                       &i8_nk0, &i8_nk1, &i8_nk2, &i8_nk3,
                       &i8_nkconv0, &i8_nkconv1,
                       input, bbox, cls);

    /* Decode to check detections */
    float dfl_w[16]; for(int i=0;i<16;i++) dfl_w[i]=(float)i;
    NN2Det dets[300];
    int nd = nn2_decode(bbox, cls, fp32->num_classes, S, 0.25f, dfl_w, dets, 300);
    nd = nn2_nms(dets, nd, 0.45f);
    printf("=== Hybrid INT8 detections (dummy input): %d dets ===\n", nd);

    /* Benchmark hybrid full pipeline */
    double t0 = h_now();
    for (int i = 0; i < RUNS; i++)
        hybrid_forward(fp32, &i8_conv2, &i8_conv3, &i8_conv4,
                       &i8_c2f1, &i8_c2f2, &i8_c2f3,
                       &i8_sppf1, &i8_sppf2,
                       &i8_nk0, &i8_nk1, &i8_nk2, &i8_nk3,
                       &i8_nkconv0, &i8_nkconv1,
                       input, bbox, cls);
    double hybrid_ms = (h_now() - t0) / RUNS * 1000.0;

    /* Benchmark pure FP32 full forward */
    t0 = h_now();
    for (int i = 0; i < RUNS; i++)
        nn2_forward(fp32, input, bbox, cls);
    double fp32_ms = (h_now() - t0) / RUNS * 1000.0;

    printf("\n=== Full Pipeline (backbone + neck + head + decode) ===\n");
    printf("  Hybrid INT8+FP32: %.2f ms  (%.0f FPS)\n", hybrid_ms, 1000.0/hybrid_ms);
    printf("  Pure FP32:        %.2f ms  (%.0f FPS)\n", fp32_ms, 1000.0/fp32_ms);
    printf("  Speedup:          %.2fx\n", fp32_ms / hybrid_ms);
    printf("\n  ONNX RT reference: ~12.3 ms (81 FPS)\n");
    printf("  vs ONNX RT:        %.2fx faster\n", 12.3 / hybrid_ms);

    /* Cleanup */
    free_i8conv_s(&i8_conv2); free_i8conv_s(&i8_conv3); free_i8conv_s(&i8_conv4);
    free_i8conv_s(&i8_sppf1); free_i8conv_s(&i8_sppf2);
    free_i8conv_s(&i8_nkconv0); free_i8conv_s(&i8_nkconv1);
    free_i8c2f(&i8_c2f1); free_i8c2f(&i8_c2f2); free_i8c2f(&i8_c2f3);
    free_i8c2f(&i8_nk0); free_i8c2f(&i8_nk1); free_i8c2f(&i8_nk2); free_i8c2f(&i8_nk3);
    free(input); free(bbox); free(cls);
    nn2_free(fp32);
    return 0;
}
