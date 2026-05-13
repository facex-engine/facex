/*
 * net_int8.c — INT8 NHWC forward pass for YOLOv8n.
 *
 * Strategy:
 * - Backbone + SPPF + Neck: fully INT8 NHWC
 * - Head: dequant to FP32 CHW, use existing FP32 head
 *
 * Weight loading: load FP32 weights from existing .bin, quantize+pack at init.
 * Activation scales: from calibration file or default heuristic.
 */

#include "nn2_int8.h"
#include "../nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============ Init I8Conv from FP32 ConvLayer ============ */

static void init_i8conv(I8Conv* dst, const ConvLayer* src, float act_scale, float out_scale)
{
    int K = src->cin * src->k * src->k;
    int K4 = K_PAD4(K);
    int Cout16 = C_PAD16(src->cout);

    dst->cin = src->cin;
    dst->cout = src->cout;
    dst->k = src->k;
    dst->stride = src->stride;
    dst->pad = src->pad;
    dst->act = src->act;
    dst->K4 = K4;
    dst->cout_pad = Cout16;
    dst->act_scale = act_scale;
    dst->out_scale = out_scale;

    /* Pack weights */
    size_t packed_size = (size_t)(Cout16 / 16) * (K4 / 4) * 64;
    dst->w_packed = (int8_t*)_aligned_malloc(packed_size, 64);
    dst->col_sums = (int32_t*)malloc(Cout16 * sizeof(int32_t));
    dst->w_scales = (float*)malloc(Cout16 * sizeof(float));
    dst->bias = (float*)malloc(Cout16 * sizeof(float));

    memset(dst->col_sums, 0, Cout16 * sizeof(int32_t));
    memset(dst->w_scales, 0, Cout16 * sizeof(float));
    memset(dst->bias, 0, Cout16 * sizeof(float));
    memcpy(dst->bias, src->b, src->cout * sizeof(float));

    i8_pack_weights(src->w, src->cout, K, dst->w_packed, dst->col_sums, dst->w_scales);
}

static void free_i8conv(I8Conv* L)
{
    _aligned_free(L->w_packed);
    free(L->col_sums);
    free(L->w_scales);
    free(L->bias);
}

/* Forward-declare conv dispatch */
extern void i8_conv(const I8Conv* L, const int8_t* input, int H, int W,
                    int8_t* output, int8_t* scratch);

/* Forward-declare residual add */
extern void i8_residual_add(int8_t* a, const int8_t* b, int total,
                             float sa, float sb, float so);

/* ============ INT8 C2f block ============ */

static void i8_c2f_run(const I8C2f* blk, const int8_t* in, int H, int W,
                         int8_t* out, int8_t* scratch, int8_t* tmp)
{
    int spatial = H * W;
    int c = blk->c;
    int c_pad = C_PAD16(c);
    int total_ch = (2 + blk->n) * c;
    int total_pad = C_PAD16(total_ch);

    /* cv1: input → cat buffer (2c channels in NHWC) */
    int cv1_out_pad = C_PAD16(blk->cv1.cout);
    int8_t* cat = tmp;  /* [spatial, total_pad] */
    int8_t* cv1_out = scratch;
    i8_conv(&blk->cv1, in, H, W, cv1_out, scratch + spatial * cv1_out_pad);

    /* Split cv1 output (2c channels) into cat buffer:
     * chunk0 = channels [0,c), chunk1 = channels [c,2c)
     * In NHWC: just the first 2c channels per pixel */
    /* Copy cv1 output channels to beginning of cat */
    for (int i = 0; i < spatial; i++) {
        memcpy(cat + i * total_pad, cv1_out + i * cv1_out_pad, 2 * c);
    }

    /* Bottleneck iterations */
    int8_t* bn_in_base = cat; /* points to chunk1 start within cat */
    int bn_in_offset = c;     /* channel offset within cat for bn_in */

    for (int bi = 0; bi < blk->n; bi++) {
        /* Extract bn_in as contiguous [spatial, c_pad] */
        int8_t* bn_in_cont = scratch;
        for (int i = 0; i < spatial; i++)
            memcpy(bn_in_cont + i * c_pad, cat + i * total_pad + bn_in_offset, c);

        /* bn.cv1 */
        int8_t* bn_cv1_out = scratch + spatial * c_pad;
        i8_conv(&blk->m[bi].cv1, bn_in_cont, H, W, bn_cv1_out,
                scratch + 2 * spatial * c_pad);

        /* bn.cv2 */
        int8_t* bn_cv2_out = scratch + 2 * spatial * c_pad;
        i8_conv(&blk->m[bi].cv2, bn_cv1_out, H, W, bn_cv2_out,
                scratch + 3 * spatial * c_pad);

        /* Residual add if shortcut */
        if (blk->shortcut) {
            i8_residual_add(bn_cv2_out, bn_in_cont, spatial * c_pad,
                            blk->m[bi].cv2.out_scale, blk->m[bi].cv1.act_scale,
                            blk->m[bi].cv2.out_scale);
        }

        /* Copy bn_cv2 output to cat at position (2+bi)*c */
        int cat_offset = (2 + bi) * c;
        for (int i = 0; i < spatial; i++)
            memcpy(cat + i * total_pad + cat_offset, bn_cv2_out + i * c_pad, c);

        bn_in_offset = cat_offset;
    }

    /* cv2: cat → output */
    /* Extract cat as contiguous for cv2 input */
    int cv2_cin_pad = C_PAD16(total_ch);
    int8_t* cv2_in = scratch;
    for (int i = 0; i < spatial; i++)
        memcpy(cv2_in + i * cv2_cin_pad, cat + i * total_pad, total_ch);

    i8_conv(&blk->cv2, cv2_in, H, W, out, scratch + spatial * cv2_cin_pad);
}

/* ============ Benchmark entry point ============ */

/* Simple benchmark: load FP32 model, convert to INT8, run forward pass.
 * Uses default activation scales (heuristic). */

#include <float.h>

#ifdef _WIN32
#include <windows.h>
static double i8_now(void) {
    static LARGE_INTEGER freq; static int init=0;
    if(!init){QueryPerformanceFrequency(&freq);init=1;}
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart/(double)freq.QuadPart;
}
#endif

int main(int argc, char** argv)
{
    const char* wpath = "weights/yolov8n_320.bin";
    if (argc > 1) wpath = argv[1];

    /* Load FP32 model first */
    NN2* fp32 = nn2_init(wpath, 0);
    if (!fp32) { fprintf(stderr, "Failed to load model\n"); return 1; }
    int S = fp32->input_size;
    fprintf(stderr, "Loaded FP32 model: %dx%d, %d classes\n", S, S, fp32->num_classes);

    /* Default activation scales (should come from calibration) */
    float default_scale = 0.05f;  /* conservative default */

    /* Convert backbone convs to INT8 */
    I8Conv i8_bb_conv[5];
    for (int i = 0; i < 5; i++)
        init_i8conv(&i8_bb_conv[i], &fp32->bb_conv[i], default_scale, default_scale);

    fprintf(stderr, "Converted %d backbone convs to INT8 VNNI\n", 5);

    /* Quick GEMM benchmark: INT8 vs FP32 for backbone conv0 */
    int H = S, W = S;
    int h0 = S/2, w0 = S/2;

    /* Create dummy INT8 NHWC input */
    int C_in = 3, C_in_pad = C_PAD16(C_in);
    int8_t* input_i8 = (int8_t*)calloc(H * W * C_in_pad, 1);
    for (int i = 0; i < H * W * C_in_pad; i++) input_i8[i] = (int8_t)(i % 100 - 50);

    /* Output buffer */
    int cout0 = fp32->bb_conv[0].cout;
    int cout0_pad = C_PAD16(cout0);
    int8_t* output_i8 = (int8_t*)calloc(h0 * w0 * cout0_pad, 1);
    int8_t* scratch = (int8_t*)calloc(4 * 1024 * 1024, 1); /* 4MB scratch */

    /* FP32 input for comparison */
    float* input_fp32 = (float*)calloc(3 * S * S, sizeof(float));
    for (int i = 0; i < 3 * S * S; i++) input_fp32[i] = 0.5f;
    float* output_fp32 = (float*)calloc(cout0 * h0 * w0, sizeof(float));
    float* col_buf = (float*)calloc(4 * 1024 * 1024, sizeof(float));

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        i8_conv3x3(&i8_bb_conv[0], input_i8, H, W, output_i8, scratch);
        nn2_conv2d(&fp32->bb_conv[0], input_fp32, H, W, output_fp32, col_buf);
    }

    int RUNS = 100;

    /* Benchmark INT8 */
    double t0 = i8_now();
    for (int i = 0; i < RUNS; i++)
        i8_conv3x3(&i8_bb_conv[0], input_i8, H, W, output_i8, scratch);
    double t1 = i8_now();
    double i8_ms = (t1 - t0) / RUNS * 1000.0;

    /* Benchmark FP32 */
    t0 = i8_now();
    for (int i = 0; i < RUNS; i++)
        nn2_conv2d(&fp32->bb_conv[0], input_fp32, H, W, output_fp32, col_buf);
    t1 = i8_now();
    double fp32_ms = (t1 - t0) / RUNS * 1000.0;

    printf("=== Conv0 (3→16, 3x3 s=2, %dx%d → %dx%d) ===\n", S, S, h0, w0);
    printf("  FP32 NCHW: %.3f ms\n", fp32_ms);
    printf("  INT8 NHWC: %.3f ms\n", i8_ms);
    printf("  Speedup:   %.2fx\n", fp32_ms / i8_ms);

    /* Benchmark all 5 backbone convs */
    printf("\n=== All backbone convs ===\n");
    int hh[] = {S, S/2, S/4, S/4, S/16};
    int ww[] = {S, S/2, S/4, S/4, S/16};
    int ho[] = {S/2, S/4, S/8, S/8, S/32};
    int wo[] = {S/2, S/4, S/8, S/8, S/32};

    /* Correct: use cumulative sizes */
    hh[0]=S; ww[0]=S; ho[0]=S/2; wo[0]=S/2;
    hh[1]=S/2; ww[1]=S/2; ho[1]=S/4; wo[1]=S/4;
    hh[2]=S/4; ww[2]=S/4; ho[2]=S/8; wo[2]=S/8;
    hh[3]=S/8; ww[3]=S/8; ho[3]=S/16; wo[3]=S/16;
    hh[4]=S/16; ww[4]=S/16; ho[4]=S/32; wo[4]=S/32;

    double total_i8 = 0, total_fp = 0;
    for (int layer = 0; layer < 5; layer++) {
        int cin = fp32->bb_conv[layer].cin;
        int cout = fp32->bb_conv[layer].cout;
        int cin_pad = C_PAD16(cin);
        int cout_p = C_PAD16(cout);

        int8_t* li = (int8_t*)calloc(hh[layer] * ww[layer] * cin_pad, 1);
        int8_t* lo = (int8_t*)calloc(ho[layer] * wo[layer] * cout_p, 1);
        float* fi = (float*)calloc(cin * hh[layer] * ww[layer], sizeof(float));
        float* fo = (float*)calloc(cout * ho[layer] * wo[layer], sizeof(float));

        for (int r = 0; r < 3; r++) {
            i8_conv3x3(&i8_bb_conv[layer], li, hh[layer], ww[layer], lo, scratch);
            nn2_conv2d(&fp32->bb_conv[layer], fi, hh[layer], ww[layer], fo, col_buf);
        }

        t0 = i8_now();
        for (int r = 0; r < RUNS; r++)
            i8_conv3x3(&i8_bb_conv[layer], li, hh[layer], ww[layer], lo, scratch);
        t1 = i8_now();
        double ms_i8 = (t1-t0)/RUNS*1000.0;

        t0 = i8_now();
        for (int r = 0; r < RUNS; r++)
            nn2_conv2d(&fp32->bb_conv[layer], fi, hh[layer], ww[layer], fo, col_buf);
        t1 = i8_now();
        double ms_fp = (t1-t0)/RUNS*1000.0;

        printf("  conv%d (%d→%d, %dx%d→%dx%d): FP32=%.3f INT8=%.3f  %.2fx\n",
               layer, cin, cout, hh[layer], ww[layer], ho[layer], wo[layer],
               ms_fp, ms_i8, ms_fp/ms_i8);
        total_i8 += ms_i8;
        total_fp += ms_fp;

        free(li); free(lo); free(fi); free(fo);
    }
    printf("  TOTAL backbone: FP32=%.3f INT8=%.3f  %.2fx\n",
           total_fp, total_i8, total_fp/total_i8);

    /* ============ Hybrid pipeline benchmark ============
     * FP32: conv0, conv1, c2f0 (Cin<32 → FP32 faster)
     * Transition: FP32 CHW → INT8 NHWC at 64ch boundary
     * INT8: conv2-4, c2f1-3, SPPF, neck (Cin≥32 → INT8 faster) */

    printf("\n=== Hybrid Pipeline Benchmark ===\n");
    {
        /* Run FP32 layers: conv0 → conv1 → c2f0 → conv2 */
        float* ws = fp32->workspace;
        float* im = ws; ws += 4*1024*1024;
        float* b0 = ws; ws += 4*1024*1024;
        float* b1 = ws; ws += 4*1024*1024;
        float* c2f_tmp = ws;

        /* Dummy input */
        float* inp = (float*)calloc(3*S*S, sizeof(float));
        for(int i=0;i<3*S*S;i++) inp[i]=0.5f;

        int HRUNS = 50;

        /* Measure FP32 portion: conv0→conv1→c2f0→conv2 */
        /* Warmup */
        nn2_conv2d(&fp32->bb_conv[0], inp, S, S, b0, im);
        nn2_conv2d(&fp32->bb_conv[1], b0, S/2, S/2, b1, im);

        double t_fp = i8_now();
        for (int r = 0; r < HRUNS; r++) {
            nn2_conv2d(&fp32->bb_conv[0], inp, S, S, b0, im);
            nn2_conv2d(&fp32->bb_conv[1], b0, S/2, S/2, b1, im);
            /* c2f0 would go here but skip for now */
            nn2_conv2d(&fp32->bb_conv[2], b1, S/4, S/4, b0, im);
        }
        double fp_early_ms = (i8_now() - t_fp) / HRUNS * 1000.0;

        /* Measure INT8 portion: conv3→conv4 (heavy layers) */
        I8Conv i8c3, i8c4;
        init_i8conv(&i8c3, &fp32->bb_conv[3], default_scale, default_scale);
        init_i8conv(&i8c4, &fp32->bb_conv[4], default_scale, default_scale);

        int c3_cin_pad = C_PAD16(fp32->bb_conv[3].cin);
        int c3_cout_pad = C_PAD16(fp32->bb_conv[3].cout);
        int c4_cout_pad = C_PAD16(fp32->bb_conv[4].cout);

        int8_t* i8_in3 = (int8_t*)calloc((S/8)*(S/8)*c3_cin_pad, 1);
        int8_t* i8_out3 = (int8_t*)calloc((S/16)*(S/16)*c3_cout_pad, 1);
        int8_t* i8_out4 = (int8_t*)calloc((S/32)*(S/32)*c4_cout_pad, 1);
        int8_t* i8_scr = (int8_t*)calloc(4*1024*1024, 1);

        /* Warmup */
        i8_conv3x3(&i8c3, i8_in3, S/8, S/8, i8_out3, i8_scr);
        i8_conv3x3(&i8c4, i8_out3, S/16, S/16, i8_out4, i8_scr);

        double t_i8 = i8_now();
        for (int r = 0; r < HRUNS; r++) {
            i8_conv3x3(&i8c3, i8_in3, S/8, S/8, i8_out3, i8_scr);
            i8_conv3x3(&i8c4, i8_out3, S/16, S/16, i8_out4, i8_scr);
        }
        double i8_late_ms = (i8_now() - t_i8) / HRUNS * 1000.0;

        /* Compare with all-FP32 for same layers */
        float* fp_in3 = (float*)calloc(64*(S/8)*(S/8), sizeof(float));
        float* fp_out3 = (float*)calloc(128*(S/16)*(S/16), sizeof(float));
        float* fp_out4 = (float*)calloc(256*(S/32)*(S/32), sizeof(float));

        nn2_conv2d(&fp32->bb_conv[3], fp_in3, S/8, S/8, fp_out3, im);
        nn2_conv2d(&fp32->bb_conv[4], fp_out3, S/16, S/16, fp_out4, im);

        double t_fp2 = i8_now();
        for (int r = 0; r < HRUNS; r++) {
            nn2_conv2d(&fp32->bb_conv[3], fp_in3, S/8, S/8, fp_out3, im);
            nn2_conv2d(&fp32->bb_conv[4], fp_out3, S/16, S/16, fp_out4, im);
        }
        double fp_late_ms = (i8_now() - t_fp2) / HRUNS * 1000.0;

        printf("  FP32 early layers (conv0-2): %.3f ms\n", fp_early_ms);
        printf("  FP32 late layers (conv3-4):  %.3f ms\n", fp_late_ms);
        printf("  INT8 late layers (conv3-4):  %.3f ms\n", i8_late_ms);
        printf("  Late layer speedup:          %.2fx\n", fp_late_ms / i8_late_ms);
        printf("\n  Hybrid estimate: %.3f ms (FP32 early + INT8 late)\n",
               fp_early_ms + i8_late_ms);
        printf("  Full FP32:       %.3f ms (conv0-4 only)\n",
               fp_early_ms + fp_late_ms);

        free_i8conv(&i8c3); free_i8conv(&i8c4);
        free(i8_in3); free(i8_out3); free(i8_out4); free(i8_scr);
        free(fp_in3); free(fp_out3); free(fp_out4);
        free(inp);
    }

    /* ============ Individual 3x3 conv benchmark at C2f sizes ============ */
    printf("\n=== C2f-sized 3x3 convs (INT8 vs FP32) ===\n");
    {
        /* C2f bottleneck convs: c→c, 3x3, s=1 */
        struct { int c; int H; const char* name; } tests[] = {
            {32,  80, "c2f0 (32ch@80x80)"},
            {32,  40, "c2f1 (32ch@40x40)"},
            {64,  20, "c2f2 (64ch@20x20)"},
            {128, 10, "c2f3 (128ch@10x10)"},
            /* head convs */
            {64,  40, "head0 (64ch@40x40)"},
            {128, 20, "head1 (128ch@20x20)"},
        };
        int ntests = sizeof(tests)/sizeof(tests[0]);

        for (int t = 0; t < ntests; t++) {
            int c = tests[t].c, Ht = tests[t].H;
            /* Create dummy FP32 conv layer */
            ConvLayer fl; memset(&fl, 0, sizeof(fl));
            fl.cout = c; fl.cin = c; fl.k = 3; fl.stride = 1; fl.pad = 1; fl.act = 1;
            int K = c * 9;
            fl.w = (float*)calloc(c * K, sizeof(float));
            fl.b = (float*)calloc(c, sizeof(float));
            for (int i = 0; i < c*K; i++) fl.w[i] = 0.001f * (i%37-18);
            for (int i = 0; i < c; i++) fl.b[i] = 0.01f * i;
            fl.w_packed = nn2_pack_weight_a(fl.w, c, K);

            I8Conv il;
            init_i8conv(&il, &fl, default_scale, default_scale);

            int sp = Ht * Ht;
            int c_pad = C_PAD16(c);
            int8_t* ii = (int8_t*)calloc(sp * c_pad, 1);
            int8_t* io = (int8_t*)calloc(sp * c_pad, 1);
            int8_t* is = (int8_t*)calloc(4*1024*1024, 1);
            float* fi = (float*)calloc(c * sp, sizeof(float));
            float* fo = (float*)calloc(c * sp, sizeof(float));
            float* fc = (float*)calloc(4*1024*1024, sizeof(float));

            int R = 200;
            /* Warmup */
            for(int r=0;r<5;r++) {
                i8_conv3x3(&il, ii, Ht, Ht, io, is);
                nn2_conv2d(&fl, fi, Ht, Ht, fo, fc);
            }

            double t0 = i8_now();
            for(int r=0;r<R;r++) i8_conv3x3(&il, ii, Ht, Ht, io, is);
            double ms_i8 = (i8_now()-t0)/R*1000.0;

            t0 = i8_now();
            for(int r=0;r<R;r++) nn2_conv2d(&fl, fi, Ht, Ht, fo, fc);
            double ms_fp = (i8_now()-t0)/R*1000.0;

            printf("  %-25s FP32=%.3f  INT8=%.3f  %.2fx\n",
                   tests[t].name, ms_fp, ms_i8, ms_fp/ms_i8);

            free_i8conv(&il);
            free(fl.w); free(fl.b); _aligned_free(fl.w_packed);
            free(ii); free(io); free(is); free(fi); free(fo); free(fc);
        }
    }

    /* Cleanup */
    for (int i = 0; i < 5; i++) free_i8conv(&i8_bb_conv[i]);
    free(input_i8); free(output_i8); free(scratch);
    free(input_fp32); free(output_fp32); free(col_buf);
    nn2_free(fp32);

    return 0;
}
