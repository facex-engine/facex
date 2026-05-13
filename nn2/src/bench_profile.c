/*
 * bench_profile.c — Per-layer profiling of YOLOv8n forward pass.
 * Identifies which layers consume the most time.
 */

#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double now(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* Global profiling counters */
#define MAX_PROF 200
static struct {
    const char* name;
    double total_ms;
    int calls;
} g_prof[MAX_PROF];
static int g_nprof = 0;

static int prof_slot(const char* name) {
    for (int i = 0; i < g_nprof; i++)
        if (strcmp(g_prof[i].name, name) == 0) return i;
    int s = g_nprof++;
    g_prof[s].name = _strdup(name);
    g_prof[s].total_ms = 0;
    g_prof[s].calls = 0;
    return s;
}

static void prof_add(const char* name, double ms) {
    int s = prof_slot(name);
    g_prof[s].total_ms += ms;
    g_prof[s].calls++;
}

/* Profiled conv */
static void prof_conv(const char* name, const ConvLayer* L, const float* in,
                      int H, int W, float* out, float* im2col) {
    double t0 = now();
    nn2_conv2d(L, in, H, W, out, im2col);
    double t1 = now();
    prof_add(name, (t1 - t0) * 1000.0);
}

/* Profiled C2f */
static void prof_c2f(const char* name, const C2fBlock* blk, const float* in,
                     int H, int W, float* out, float* im2col, float* tmp) {
    int spatial = H * W;
    int c = blk->c;
    int total_ch = (2 + blk->n) * c;
    float* cat = tmp;
    float* scratch = tmp + total_ch * spatial;
    char buf[64];

    double t0 = now();
    sprintf(buf, "%s.cv1", name);
    prof_conv(buf, &blk->cv1, in, H, W, cat, im2col);

    float* bn_in = cat + c * spatial;
    for (int i = 0; i < blk->n; i++) {
        sprintf(buf, "%s.bn%d.cv1", name, i);
        prof_conv(buf, &blk->m[i].cv1, bn_in, H, W, scratch, im2col);

        float* bn_out = cat + (2 + i) * c * spatial;
        sprintf(buf, "%s.bn%d.cv2", name, i);
        prof_conv(buf, &blk->m[i].cv2, scratch, H, W, bn_out, im2col);

        if (blk->shortcut)
            for (int j = 0; j < c * spatial; j++) bn_out[j] += bn_in[j];
        bn_in = bn_out;
    }

    sprintf(buf, "%s.cv2", name);
    prof_conv(buf, &blk->cv2, cat, H, W, out, im2col);

    double t1 = now();
    prof_add(name, (t1 - t0) * 1000.0);
}

int main(int argc, char** argv)
{
    const char* wpath = "weights/yolov8n_320.bin";
    if (argc > 1) wpath = argv[1];

    NN2* ctx = nn2_init(wpath, 0);
    if (!ctx) { fprintf(stderr, "Failed to init\n"); return 1; }

    int S = ctx->input_size;
    float* input = (float*)calloc(3 * S * S, sizeof(float));
    for (int i = 0; i < 3 * S * S; i++) input[i] = 0.5f;

    /* Warmup — run forward pass directly with float input */
    int total_anchors = 0;
    {
        int bbox_ch = 4 * NN2_REG_MAX;
        int nc = ctx->num_classes;
        int ta = (S/8)*(S/8) + (S/16)*(S/16) + (S/32)*(S/32);
        float* bbox = (float*)malloc(bbox_ch * ta * sizeof(float));
        float* cls = (float*)malloc(nc * ta * sizeof(float));
        for (int i = 0; i < 3; i++)
            total_anchors = nn2_forward(ctx, input, bbox, cls);
        free(bbox); free(cls);
    }

    /* Profile: run forward pass manually with timing */
    int RUNS = 20;

    for (int run = 0; run < RUNS; run++) {
        float* ws = ctx->workspace;
        float* im2col = ws; ws += 4*1024*1024;
        float* buf0 = ws; ws += 4*1024*1024;
        float* buf1 = ws; ws += 4*1024*1024;
        float* save_p3 = ws; ws += 64*(S/8)*(S/8);
        float* save_p4 = ws; ws += 128*(S/16)*(S/16);
        float* save_p5 = ws; ws += 256*(S/32)*(S/32);
        float* save_n3 = ws; ws += 128*(S/16)*(S/16);
        float* c2f_tmp = ws;

        float *cur = buf0, *nxt = buf1;
        #define SWAP() { float* t = cur; cur = nxt; nxt = t; }
        int h = S, w = S;
        int h0=S/2,w0=S/2, h1=S/4,w1=S/4, h2=S/8,w2=S/8, h3=S/16,w3=S/16, h4=S/32,w4=S/32;

        prof_conv("bb_conv0", &ctx->bb_conv[0], input, h, w, cur, im2col);
        prof_conv("bb_conv1", &ctx->bb_conv[1], cur, h0, w0, nxt, im2col); SWAP();
        prof_c2f("bb_c2f0", &ctx->bb_c2f[0], cur, h1, w1, nxt, im2col, c2f_tmp); SWAP();
        prof_conv("bb_conv2", &ctx->bb_conv[2], cur, h1, w1, nxt, im2col); SWAP();
        prof_c2f("bb_c2f1", &ctx->bb_c2f[1], cur, h2, w2, nxt, im2col, c2f_tmp); SWAP();
        memcpy(save_p3, cur, 64*h2*w2*sizeof(float));
        prof_conv("bb_conv3", &ctx->bb_conv[3], cur, h2, w2, nxt, im2col); SWAP();
        prof_c2f("bb_c2f2", &ctx->bb_c2f[2], cur, h3, w3, nxt, im2col, c2f_tmp); SWAP();
        memcpy(save_p4, cur, 128*h3*w3*sizeof(float));
        prof_conv("bb_conv4", &ctx->bb_conv[4], cur, h3, w3, nxt, im2col); SWAP();
        prof_c2f("bb_c2f3", &ctx->bb_c2f[3], cur, h4, w4, nxt, im2col, c2f_tmp); SWAP();

        prof_conv("sppf_cv1", &ctx->sppf_cv1, cur, h4, w4, nxt, im2col); SWAP();
        {
            int sp = h4*w4;
            float* cat = c2f_tmp;
            float* mp1 = cat + 512*sp, *mp2 = mp1+128*sp, *mp3 = mp2+128*sp;
            double t0 = now();
            nn2_maxpool2d(cur, 128, h4, w4, 5, 1, 2, mp1);
            nn2_maxpool2d(mp1, 128, h4, w4, 5, 1, 2, mp2);
            nn2_maxpool2d(mp2, 128, h4, w4, 5, 1, 2, mp3);
            memcpy(cat, cur, 128*sp*sizeof(float));
            memcpy(cat+128*sp, mp1, 128*sp*sizeof(float));
            memcpy(cat+256*sp, mp2, 128*sp*sizeof(float));
            memcpy(cat+384*sp, mp3, 128*sp*sizeof(float));
            prof_add("sppf_pool", (now()-t0)*1000.0);
            prof_conv("sppf_cv2", &ctx->sppf_cv2, cat, h4, w4, nxt, im2col); SWAP();
        }
        memcpy(save_p5, cur, 256*h4*w4*sizeof(float));

        /* Neck */
        {
            float* up = c2f_tmp;
            double t0 = now();
            nn2_upsample_2x(cur, 256, h4, w4, up);
            nn2_concat(up, 256, save_p4, 128, h3, w3, nxt);
            prof_add("upsample1", (now()-t0)*1000.0);
            SWAP();
        }
        prof_c2f("nk_c2f0", &ctx->nk_c2f[0], cur, h3, w3, nxt, im2col, c2f_tmp); SWAP();
        memcpy(save_n3, cur, 128*h3*w3*sizeof(float));

        {
            float* up = c2f_tmp;
            double t0 = now();
            nn2_upsample_2x(cur, 128, h3, w3, up);
            nn2_concat(up, 128, save_p3, 64, h2, w2, nxt);
            prof_add("upsample2", (now()-t0)*1000.0);
            SWAP();
        }

        float* feat_n2 = save_p3;
        prof_c2f("nk_c2f1", &ctx->nk_c2f[1], cur, h2, w2, feat_n2, im2col, c2f_tmp);

        prof_conv("nk_conv0", &ctx->nk_conv[0], feat_n2, h2, w2, cur, im2col);
        {
            double t0 = now();
            nn2_concat(cur, 64, save_n3, 128, h3, w3, nxt);
            prof_add("concat3", (now()-t0)*1000.0);
            SWAP();
        }

        float* feat_n4 = save_p4;
        prof_c2f("nk_c2f2", &ctx->nk_c2f[2], cur, h3, w3, feat_n4, im2col, c2f_tmp);

        prof_conv("nk_conv1", &ctx->nk_conv[1], feat_n4, h3, w3, cur, im2col);
        {
            double t0 = now();
            nn2_concat(cur, 128, save_p5, 256, h4, w4, nxt);
            prof_add("concat4", (now()-t0)*1000.0);
            SWAP();
        }

        float* feat_n5 = save_p5;
        prof_c2f("nk_c2f3", &ctx->nk_c2f[3], cur, h4, w4, feat_n5, im2col, c2f_tmp);

        /* Head */
        float* feats[3] = {feat_n2, feat_n4, feat_n5};
        int feat_h[] = {h2, h3, h4};
        int feat_w[] = {w2, w3, w4};
        for (int s = 0; s < 3; s++) {
            char name[32];
            float* tmp = c2f_tmp;
            int fh = feat_h[s], fw = feat_w[s], sp = fh*fw;

            sprintf(name, "head%d_bbox0", s);
            prof_conv(name, &ctx->head[s].bbox[0], feats[s], fh, fw, tmp, im2col);
            sprintf(name, "head%d_bbox1", s);
            prof_conv(name, &ctx->head[s].bbox[1], tmp, fh, fw, tmp+ctx->head[s].bbox[0].cout*sp, im2col);
            sprintf(name, "head%d_bbox2", s);
            prof_conv(name, &ctx->head[s].bbox[2], tmp+ctx->head[s].bbox[0].cout*sp, fh, fw,
                      tmp+ctx->head[s].bbox[0].cout*sp+ctx->head[s].bbox[1].cout*sp, im2col);

            float* cx = tmp + ctx->head[s].bbox[0].cout*sp + ctx->head[s].bbox[1].cout*sp + 64*sp;
            sprintf(name, "head%d_cls0", s);
            prof_conv(name, &ctx->head[s].cls[0], feats[s], fh, fw, cx, im2col);
            sprintf(name, "head%d_cls1", s);
            prof_conv(name, &ctx->head[s].cls[1], cx, fh, fw, cx + ctx->head[s].cls[0].cout*sp, im2col);
            sprintf(name, "head%d_cls2", s);
            prof_conv(name, &ctx->head[s].cls[2], cx + ctx->head[s].cls[0].cout*sp, fh, fw,
                      cx + ctx->head[s].cls[0].cout*sp + ctx->head[s].cls[1].cout*sp, im2col);
        }
        #undef SWAP
    }

    /* Print results sorted by time */
    printf("=== YOLOv8n@%d Per-Layer Profile (%d runs) ===\n\n", S, RUNS);
    printf("%-30s %8s %8s %6s\n", "Layer", "Total ms", "Avg ms", "%%");

    double total = 0;
    for (int i = 0; i < g_nprof; i++) total += g_prof[i].total_ms;

    /* Bubble sort by total_ms desc */
    for (int i = 0; i < g_nprof - 1; i++)
        for (int j = i + 1; j < g_nprof; j++)
            if (g_prof[j].total_ms > g_prof[i].total_ms) {
                typeof(g_prof[0]) tmp = g_prof[i];
                g_prof[i] = g_prof[j]; g_prof[j] = tmp;
            }

    for (int i = 0; i < g_nprof; i++) {
        double avg = g_prof[i].total_ms / g_prof[i].calls;
        double pct = 100.0 * g_prof[i].total_ms / total;
        if (pct < 0.5) continue; /* skip tiny layers */
        printf("%-30s %8.2f %8.3f %5.1f%%\n",
               g_prof[i].name, g_prof[i].total_ms, avg, pct);
    }
    printf("%-30s %8.2f\n", "TOTAL", total);
    printf("\nPer-inference: %.2f ms\n", total / RUNS);

    free(input);
    nn2_free(ctx);
    return 0;
}
