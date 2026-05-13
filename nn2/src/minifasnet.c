/*
 * minifasnet.c — MiniFASNetV2 / V1SE forward pass for the nn2 engine.
 *
 * Mirrors `src/model_lib/MiniFASNet.py` from MinivisionAI's
 * Silent-Face-Anti-Spoofing repo. Apache 2.0 weights.
 *
 * Architecture summary (input 80×80 → spatial 40,20,10,5,1):
 *    conv1            Conv 3x3 s=2   3→32      (80→40)
 *    conv2_dw         DWConv 3x3 s=1 32→32     (40→40)
 *    conv_23          Depth_Wise s=2 (32→64,   non-residual) (40→20)
 *    conv_3           4× Depth_Wise s=1 (64-channel residual)
 *    conv_34          Depth_Wise s=2 (64→128,  non-residual)  (20→10)
 *    conv_4           6× Depth_Wise s=1 (128-channel residual)
 *    conv_45          Depth_Wise s=2 (128→128, non-residual)  (10→5)
 *    conv_5           2× Depth_Wise s=1 (128-channel residual)
 *    conv_6_sep       Conv 1x1 128→512
 *    conv_6_dw        DWConv 5x5 s=1 p=0  (5→1)
 *    linear           FC 512→128 (BN1d folded into bias+weights)
 *    prob             FC 128→3 (no bias)
 *
 * Every Conv_block uses PReLU (per-channel α). Project conv inside a
 * Depth_Wise has NO activation.
 *
 * V1SE inserts an SE module (GAP → FC↓ → ReLU → FC↑ → sigmoid → multiply)
 * after `project` in the LAST Depth_Wise of each Residual section.
 *
 * Weight file format (.bin) written by tools/export_minifasnet.py:
 *   [4]  magic "MFN1"
 *   [4]  uint32 variant   (0=V2, 1=V1SE)
 *   [4]  uint32 input_size (always 80)
 *   [4]  uint32 num_tensors
 *   Per tensor: uint32 size_in_floats; float32[size]
 */

#include "nn2_internal.h"
#include "nn2_antispoof.h"
#include "nn2_antispoof_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>


/* ===== keep_dict from MiniFASNet.py ===== */
static const int KEEP_18M[49] = {
    32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
    64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
    154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
    52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
    128, 26, 26, 128, 26, 26, 128, 512, 512
};
static const int KEEP_18M_[49] = {
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
    13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
    52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
    128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512
};

/* ===== Per-layer structures ===== */

typedef struct {
    float *expand_w, *expand_b, *expand_prelu;     /* 1x1, with PReLU */
    float *dw_w,     *dw_b,     *dw_prelu;          /* 3x3 DW, with PReLU */
    float *project_w, *project_b;                    /* 1x1, no activation */
    int   c1_in,  c1_out;
    int   c2_out;
    int   c3_out;
    int   stride;
    int   residual;
    int   has_se;
    int   se_reduct;
    float *se_fc1_w, *se_fc1_b;
    float *se_fc2_w, *se_fc2_b;
} DepthWise;

struct AntiSpoofModel {
    AntiSpoofVariant variant;
    int input_size;

    /* conv1 / conv2_dw with PReLU */
    float *conv1_w, *conv1_b, *conv1_alpha;
    float *conv2_dw_w, *conv2_dw_b, *conv2_dw_alpha;

    DepthWise conv_23;
    DepthWise conv_3[4];
    DepthWise conv_34;
    DepthWise conv_4[6];
    DepthWise conv_45;
    DepthWise conv_5[2];

    float *conv_6_sep_w, *conv_6_sep_b, *conv_6_sep_alpha;
    float *conv_6_dw_w,  *conv_6_dw_b;

    float *fc1_w, *fc1_b;     /* 128×512 + 128 (BN fused) */
    float *fc2_w;             /* 3×128 no bias */

    /* Scratch buffers. Allocated big enough for the widest intermediate. */
    float *act_a, *act_b;     /* ping-pong activation buffers */
    float *scratch_x;         /* expand output / SE per-block tmp */
    float *scratch_y;         /* DW output */
    float *im2col;            /* im2col buffer for non-DW convs */
    size_t buf_floats;
    size_t im2col_floats;

    int n_threads;
};

/* ===== Allocation helpers ===== */
static float* aalloc(size_t n)
{
    return (float*)_mm_malloc(n * sizeof(float), 64);
}
static void afree(float* p) { if (p) _mm_free(p); }

/* ===== Weight reading ===== */
static int read_u32(FILE* f, uint32_t* out)
{
    return fread(out, 4, 1, f) == 1 ? 0 : -1;
}

static float* read_tensor(FILE* f, int expected)
{
    uint32_t sz;
    if (read_u32(f, &sz) != 0) return NULL;
    if ((int)sz != expected) {
        fprintf(stderr, "minifasnet: tensor size mismatch: got %u expected %d\n", sz, expected);
        return NULL;
    }
    float* buf = aalloc(sz);
    if (!buf) return NULL;
    if (fread(buf, 4, sz, f) != (size_t)sz) { afree(buf); return NULL; }
    return buf;
}

static int load_conv(FILE* f, int cin, int cout, int k, int has_prelu,
                      float** w, float** b, float** prelu)
{
    *w = read_tensor(f, cout * cin * k * k);
    *b = read_tensor(f, cout);
    if (!*w || !*b) return -1;
    if (has_prelu) {
        *prelu = read_tensor(f, cout);
        if (!*prelu) return -1;
    } else if (prelu) {
        *prelu = NULL;
    }
    return 0;
}

static int load_dw_block(FILE* f, int c1_in, int c1_out, int c3_out,
                          int stride, int residual, int has_se, int se_reduct,
                          DepthWise* d)
{
    d->c1_in = c1_in; d->c1_out = c1_out; d->c2_out = c1_out; d->c3_out = c3_out;
    d->stride = stride; d->residual = residual;
    if (load_conv(f, c1_in,  c1_out, 1, 1, &d->expand_w, &d->expand_b, &d->expand_prelu) != 0) return -1;
    if (load_conv(f, 1,      c1_out, 3, 1, &d->dw_w,     &d->dw_b,     &d->dw_prelu) != 0) return -1;
    if (load_conv(f, c1_out, c3_out, 1, 0, &d->project_w, &d->project_b, NULL) != 0) return -1;
    d->has_se = 0;
    if (has_se) {
        d->has_se = 1; d->se_reduct = se_reduct;
        int r = c3_out / se_reduct;
        d->se_fc1_w = read_tensor(f, r * c3_out);
        d->se_fc1_b = read_tensor(f, r);
        d->se_fc2_w = read_tensor(f, c3_out * r);
        d->se_fc2_b = read_tensor(f, c3_out);
        if (!d->se_fc1_w || !d->se_fc1_b || !d->se_fc2_w || !d->se_fc2_b) return -1;
    }
    return 0;
}

static void free_dw(DepthWise* d)
{
    afree(d->expand_w); afree(d->expand_b); afree(d->expand_prelu);
    afree(d->dw_w);     afree(d->dw_b);     afree(d->dw_prelu);
    afree(d->project_w); afree(d->project_b);
    if (d->has_se) {
        afree(d->se_fc1_w); afree(d->se_fc1_b);
        afree(d->se_fc2_w); afree(d->se_fc2_b);
    }
}

/* ===== Conv wrapper: chooses 1x1 (no im2col) or im2col+GEMM ===== */
static void conv_run(const float* in, int Cin, int H, int W,
                     const float* w, const float* b,
                     int Cout, int k, int stride, int pad,
                     float* col_buf, float* out)
{
    int Hout = (H + 2 * pad - k) / stride + 1;
    int Wout = (W + 2 * pad - k) / stride + 1;
    int K = Cin * k * k;
    int N = Hout * Wout;
    if (k == 1 && stride == 1 && pad == 0) {
        nn2_sgemm_bias_act(Cout, K, N, w, K, in, N, out, N, b, 0);
    } else {
        nn2_im2col(in, Cin, H, W, k, k, stride, pad, col_buf);
        nn2_sgemm_bias_act(Cout, K, N, w, K, col_buf, N, out, N, b, 0);
    }
}

/* ===== Depth_Wise forward =====
 * in [c1_in × H×W] → out [c3_out × Hout×Wout]; in and out MUST be different. */
static void run_dw(const DepthWise* d, const float* in, int H, int W,
                   float* col_buf, float* scratch_x, float* scratch_y, float* out)
{
    int Hout = (H + 2 - 3) / d->stride + 1;
    int Wout = (W + 2 - 3) / d->stride + 1;

    /* 1. expand 1x1 → scratch_x [c1_out × H×W] */
    conv_run(in, d->c1_in, H, W, d->expand_w, d->expand_b,
             d->c1_out, 1, 1, 0, col_buf, scratch_x);
    nn2_prelu(scratch_x, d->expand_prelu, d->c1_out, H * W);

    /* 2. DW 3x3 → scratch_y [c2_out × Hout×Wout] */
    nn2_dwconv2d(scratch_x, d->dw_w, d->dw_b, d->c2_out, H, W, 3, d->stride, 1, 0, scratch_y);
    nn2_prelu(scratch_y, d->dw_prelu, d->c2_out, Hout * Wout);

    /* 3. project 1x1 → out [c3_out × Hout×Wout], no activation */
    conv_run(scratch_y, d->c2_out, Hout, Wout, d->project_w, d->project_b,
             d->c3_out, 1, 1, 0, col_buf, out);

    /* 4. Optional SE: out *= sigmoid(fc2(relu(fc1(GAP(out))))) */
    if (d->has_se) {
        int r = d->c3_out / d->se_reduct;
        float gap[256], v1[64], v2[256];
        nn2_global_avg_pool(out, d->c3_out, Hout * Wout, gap);
        nn2_linear(gap, d->se_fc1_w, d->se_fc1_b, d->c3_out, r, v1);
        for (int i = 0; i < r; i++) if (v1[i] < 0) v1[i] = 0;
        nn2_linear(v1, d->se_fc2_w, d->se_fc2_b, r, d->c3_out, v2);
        nn2_sigmoid_inplace(v2, d->c3_out);
        nn2_channel_mul(out, v2, d->c3_out, Hout * Wout);
    }

    /* 5. Residual add — only valid when channels match and stride=1 */
    if (d->residual && d->stride == 1 && d->c1_in == d->c3_out) {
        nn2_add_inplace(out, in, d->c3_out * Hout * Wout);
    }
}

/* ===== Loader ===== */

AntiSpoofModel* nn2_antispoof_load(const char* path, AntiSpoofVariant variant, int n_threads)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "minifasnet: cannot open %s\n", path); return NULL; }

    char magic[4];
    uint32_t var_u, sz_u, nt_u;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MFN1", 4) != 0
        || read_u32(f, &var_u) || read_u32(f, &sz_u) || read_u32(f, &nt_u))
    {
        fprintf(stderr, "minifasnet: bad header\n"); fclose(f); return NULL;
    }
    if ((int)var_u != (int)variant) {
        fprintf(stderr, "minifasnet: variant mismatch\n"); fclose(f); return NULL;
    }

    if (n_threads <= 0) n_threads = 4;
    nn2_tp_init(n_threads);

    AntiSpoofModel* m = (AntiSpoofModel*)calloc(1, sizeof(*m));
    m->variant = variant;
    m->input_size = (int)sz_u;
    m->n_threads = n_threads;

    const int* keep = (variant == NN2_AS_V2) ? KEEP_18M_ : KEEP_18M;
    int se = (variant == NN2_AS_V1SE) ? 1 : 0;
    int rc = 0;

    /* conv1 3→32 k=3 s=2 */
    rc |= load_conv(f, 3, keep[0], 3, 1, &m->conv1_w, &m->conv1_b, &m->conv1_alpha);
    /* conv2_dw 32 dw k=3 s=1 */
    rc |= load_conv(f, 1, keep[1], 3, 1, &m->conv2_dw_w, &m->conv2_dw_b, &m->conv2_dw_alpha);

    /* conv_23: c1=(keep[1],keep[2]), c3_out=keep[4], stride=2, NOT residual, no SE */
    rc |= load_dw_block(f, keep[1], keep[2], keep[4], 2, 0, 0, 0, &m->conv_23);

    /* conv_3: 4 blocks. block i uses c1=(keep[4+3i], keep[5+3i]), c3_out=keep[7+3i]. All stride=1, residual, SE on last if V1SE. */
    for (int i = 0; i < 4; i++) {
        int ki = 4 + 3 * i;
        int has_se_here = (se && i == 3) ? 1 : 0;
        rc |= load_dw_block(f, keep[ki], keep[ki + 1], keep[ki + 3], 1, 1,
                             has_se_here, 4, &m->conv_3[i]);
    }
    /* conv_34: c1=(keep[16], keep[17]), c3_out=keep[19], stride=2 */
    rc |= load_dw_block(f, keep[16], keep[17], keep[19], 2, 0, 0, 0, &m->conv_34);

    /* conv_4: 6 blocks starting at keep[19] */
    for (int i = 0; i < 6; i++) {
        int ki = 19 + 3 * i;
        int has_se_here = (se && i == 5) ? 1 : 0;
        rc |= load_dw_block(f, keep[ki], keep[ki + 1], keep[ki + 3], 1, 1,
                             has_se_here, 4, &m->conv_4[i]);
    }
    /* conv_45: c1=(keep[37], keep[38]), c3_out=keep[40], stride=2 */
    rc |= load_dw_block(f, keep[37], keep[38], keep[40], 2, 0, 0, 0, &m->conv_45);

    /* conv_5: 2 blocks starting at keep[40] */
    for (int i = 0; i < 2; i++) {
        int ki = 40 + 3 * i;
        int has_se_here = (se && i == 1) ? 1 : 0;
        rc |= load_dw_block(f, keep[ki], keep[ki + 1], keep[ki + 3], 1, 1,
                             has_se_here, 4, &m->conv_5[i]);
    }

    /* conv_6_sep 1x1 keep[46]→keep[47] */
    rc |= load_conv(f, keep[46], keep[47], 1, 1, &m->conv_6_sep_w, &m->conv_6_sep_b, &m->conv_6_sep_alpha);
    /* conv_6_dw 5x5 DW (no activation, no PReLU) */
    m->conv_6_dw_w = read_tensor(f, keep[48] * 5 * 5);
    m->conv_6_dw_b = read_tensor(f, keep[48]);

    /* Linear 512→128 (BN1d folded into [w, b]) */
    m->fc1_w = read_tensor(f, 128 * 512);
    m->fc1_b = read_tensor(f, 128);
    /* prob 128→3, no bias */
    m->fc2_w = read_tensor(f, 3 * 128);

    if (rc != 0 || !m->fc2_w) {
        fprintf(stderr, "minifasnet: weight load failed\n");
        fclose(f); nn2_antispoof_free(m); return NULL;
    }
    fclose(f);

    /* Allocate workspace. Widest intermediate: expand-hidden in conv_3/4
     * with 231 channels at 20×20 (V1SE conv_3 first block: keep[5]=13 hmm
     * but V2 has keep[5]=13 too). Actually keep[17]=231 at HxW=20×20=400 spatial.
     * Max activation = 231 * 20 * 20 = 92400.
     * For safety allocate 512 * 40 * 40 = 819200 floats = 3.2 MB. */
    m->buf_floats = 512 * 40 * 40;
    m->act_a = aalloc(m->buf_floats);
    m->act_b = aalloc(m->buf_floats);
    m->scratch_x = aalloc(m->buf_floats);
    m->scratch_y = aalloc(m->buf_floats);
    /* im2col worst case: conv1 needs 3*9 = 27 cols × (40*40) = 43200 floats */
    m->im2col_floats = 27 * 40 * 40;
    m->im2col = aalloc(m->im2col_floats);

    if (!m->act_a || !m->act_b || !m->scratch_x || !m->scratch_y || !m->im2col) {
        nn2_antispoof_free(m); return NULL;
    }
    return m;
}

void nn2_antispoof_free(AntiSpoofModel* m)
{
    if (!m) return;
    afree(m->conv1_w); afree(m->conv1_b); afree(m->conv1_alpha);
    afree(m->conv2_dw_w); afree(m->conv2_dw_b); afree(m->conv2_dw_alpha);
    free_dw(&m->conv_23);
    for (int i = 0; i < 4; i++) free_dw(&m->conv_3[i]);
    free_dw(&m->conv_34);
    for (int i = 0; i < 6; i++) free_dw(&m->conv_4[i]);
    free_dw(&m->conv_45);
    for (int i = 0; i < 2; i++) free_dw(&m->conv_5[i]);
    afree(m->conv_6_sep_w); afree(m->conv_6_sep_b); afree(m->conv_6_sep_alpha);
    afree(m->conv_6_dw_w);  afree(m->conv_6_dw_b);
    afree(m->fc1_w); afree(m->fc1_b); afree(m->fc2_w);
    afree(m->act_a); afree(m->act_b); afree(m->scratch_x); afree(m->scratch_y);
    afree(m->im2col);
    free(m);
}

/* ===== Forward ===== */

int nn2_antispoof_predict(AntiSpoofModel* m, const unsigned char* bgr_80x80,
                           float prob_out[3])
{
    if (!m || !bgr_80x80 || !prob_out) return -1;
    const int S = 80;
    int N = S * S;

    /* Preprocess HWC BGR uint8 [0,255] → CHW float, NO /255. */
    float* input = m->act_a;
    for (int i = 0; i < N; i++) {
        input[0 * N + i] = (float)bgr_80x80[i * 3 + 0];
        input[1 * N + i] = (float)bgr_80x80[i * 3 + 1];
        input[2 * N + i] = (float)bgr_80x80[i * 3 + 2];
    }

    float *cur = m->act_a, *nxt = m->act_b;

    /* conv1: 3→32, k=3, s=2  (80→40) */
    int H = 40, W = 40;
    conv_run(cur, 3, S, S, m->conv1_w, m->conv1_b, 32, 3, 2, 1, m->im2col, nxt);
    nn2_prelu(nxt, m->conv1_alpha, 32, H * W);
    { float* t = cur; cur = nxt; nxt = t; }

    /* conv2_dw: 32 DW, s=1  (40→40) */
    nn2_dwconv2d(cur, m->conv2_dw_w, m->conv2_dw_b, 32, H, W, 3, 1, 1, 0, nxt);
    nn2_prelu(nxt, m->conv2_dw_alpha, 32, H * W);
    { float* t = cur; cur = nxt; nxt = t; }

    /* conv_23: s=2, 32→hidden→64  (40→20) */
    run_dw(&m->conv_23, cur, H, W, m->im2col, m->scratch_x, m->scratch_y, nxt);
    { float* t = cur; cur = nxt; nxt = t; }
    H = 20; W = 20;

    /* conv_3: 4× s=1 residual, 64-channel */
    for (int i = 0; i < 4; i++) {
        run_dw(&m->conv_3[i], cur, H, W, m->im2col, m->scratch_x, m->scratch_y, nxt);
        { float* t = cur; cur = nxt; nxt = t; }
    }

    /* conv_34: s=2, 64→hidden→128 (20→10) */
    run_dw(&m->conv_34, cur, H, W, m->im2col, m->scratch_x, m->scratch_y, nxt);
    { float* t = cur; cur = nxt; nxt = t; }
    H = 10; W = 10;

    /* conv_4: 6× s=1, 128-channel */
    for (int i = 0; i < 6; i++) {
        run_dw(&m->conv_4[i], cur, H, W, m->im2col, m->scratch_x, m->scratch_y, nxt);
        { float* t = cur; cur = nxt; nxt = t; }
    }

    /* conv_45: s=2, 128→hidden→128 (10→5) */
    run_dw(&m->conv_45, cur, H, W, m->im2col, m->scratch_x, m->scratch_y, nxt);
    { float* t = cur; cur = nxt; nxt = t; }
    H = 5; W = 5;

    /* conv_5: 2× s=1 */
    for (int i = 0; i < 2; i++) {
        run_dw(&m->conv_5[i], cur, H, W, m->im2col, m->scratch_x, m->scratch_y, nxt);
        { float* t = cur; cur = nxt; nxt = t; }
    }

    /* conv_6_sep: 1x1 128→512 with PReLU */
    conv_run(cur, 128, H, W, m->conv_6_sep_w, m->conv_6_sep_b, 512, 1, 1, 0, m->im2col, nxt);
    nn2_prelu(nxt, m->conv_6_sep_alpha, 512, H * W);
    { float* t = cur; cur = nxt; nxt = t; }

    /* conv_6_dw: DW 5x5 s=1 p=0  (5→1), no activation. Output: 512 floats. */
    nn2_dwconv2d(cur, m->conv_6_dw_w, m->conv_6_dw_b, 512, H, W, 5, 1, 0, 0, nxt);
    cur = nxt;

    /* Linear 512→128, then prob 128→3 */
    float fc1_out[128];
    nn2_linear(cur, m->fc1_w, m->fc1_b, 512, 128, fc1_out);
    float logits[3];
    nn2_linear(fc1_out, m->fc2_w, NULL, 128, 3, logits);
    nn2_softmax(logits, 3, prob_out);
    return 0;
}

int nn2_antispoof_predict_rgb(AntiSpoofModel* m, const unsigned char* rgb,
                               int src_w, int src_h,
                               int bbox_x, int bbox_y, int bbox_w, int bbox_h,
                               float scale, float prob_out[3])
{
    /* CropImage._get_new_box logic (forward port of generate_patches.py). */
    float s = scale;
    float bh = (float)(src_h - 1) / bbox_h;
    float bw = (float)(src_w - 1) / bbox_w;
    if (bh < s) s = bh;
    if (bw < s) s = bw;
    float new_w = bbox_w * s, new_h = bbox_h * s;
    float cx = bbox_x + bbox_w / 2.0f, cy = bbox_y + bbox_h / 2.0f;
    float ltx = cx - new_w / 2, lty = cy - new_h / 2;
    float rbx = cx + new_w / 2, rby = cy + new_h / 2;
    if (ltx < 0) { rbx -= ltx; ltx = 0; }
    if (lty < 0) { rby -= lty; lty = 0; }
    if (rbx > src_w - 1) { ltx -= rbx - src_w + 1; rbx = src_w - 1; }
    if (rby > src_h - 1) { lty -= rby - src_h + 1; rby = src_h - 1; }
    int ix0 = (int)ltx, iy0 = (int)lty;
    int ix1 = (int)rbx + 1, iy1 = (int)rby + 1;
    int cw = ix1 - ix0, ch = iy1 - iy0;
    if (cw <= 0 || ch <= 0) return -1;

    /* Bilinear resize to 80×80 BGR uint8. */
    const int S = 80;
    unsigned char bgr[80 * 80 * 3];
    for (int y = 0; y < S; y++) {
        float fy = (float)y * (ch - 1) / (S - 1);
        int y0 = (int)fy; int y1 = y0 + 1; if (y1 >= ch) y1 = ch - 1;
        float wy = fy - y0;
        for (int x = 0; x < S; x++) {
            float fx = (float)x * (cw - 1) / (S - 1);
            int x0 = (int)fx; int x1 = x0 + 1; if (x1 >= cw) x1 = cw - 1;
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                int src_c = 2 - c;     /* input RGB → output BGR */
                float v00 = rgb[((iy0 + y0) * src_w + (ix0 + x0)) * 3 + src_c];
                float v01 = rgb[((iy0 + y0) * src_w + (ix0 + x1)) * 3 + src_c];
                float v10 = rgb[((iy0 + y1) * src_w + (ix0 + x0)) * 3 + src_c];
                float v11 = rgb[((iy0 + y1) * src_w + (ix0 + x1)) * 3 + src_c];
                float v0 = v00 * (1 - wx) + v01 * wx;
                float v1 = v10 * (1 - wx) + v11 * wx;
                float v  = v0  * (1 - wy) + v1  * wy;
                bgr[(y * S + x) * 3 + c] = (unsigned char)(v + 0.5f);
            }
        }
    }
    return nn2_antispoof_predict(m, bgr, prob_out);
}
