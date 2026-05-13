/*
 * facex_mfn.c — Parametric MobileFaceNet inference engine.
 *
 * Supports 4 size variants (Nano / Tiny / Standard / XS) via a binary
 * arch header in the EFM3 .bin file. BatchNorm is folded into the
 * preceding convolution at load time, so the runtime forward pass is
 * Conv + Bias + (optional) PReLU.
 *
 * Activation layout: NCHW throughout. Convs are pure-C reference
 * implementations. AVX2 micro-kernels can be added without changing
 * the API.
 */

#include "../include/facex_mfn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define MFN_MAX_STAGES 8
#define MFN_MAX_BLOCKS 64    /* sum of n across all stages */
#define MFN_INPUT_HW 112


/* ============ On-disk header layout (must match binformat.py) ============ */
#define EFM3_MAGIC "EFM3"
#define EFM3_VERSION 3
#define HDR_FIXED_SIZE 16
#define STAGE_REC_SIZE 8
#define ARCH_HDR_SIZE (HDR_FIXED_SIZE + MFN_MAX_STAGES * STAGE_REC_SIZE)  /* 80 */

typedef struct {
    int n_stages;
    int stem_kernel, stem_stride, stem_channels;
    int block_channels, final_channels, embedding_dim, gdc_kernel;
    int stage_t[MFN_MAX_STAGES];
    int stage_n[MFN_MAX_STAGES];
    int stage_s[MFN_MAX_STAGES];
} MfnArch;


/* ============ Layer descriptors (BN already folded in) ============ */
typedef struct {
    float *w;           /* [Cout, Cin/groups, K, K] */
    float *b;           /* [Cout] */
    float *prelu;       /* [Cout] PReLU slope, or NULL for no activation */
    int c_in, c_out, kernel, stride, padding, groups;
} MfnConv;

typedef struct {
    MfnConv expand, dw, project;
    int has_residual;
} MfnBlock;

struct MfnEngine {
    MfnArch arch;
    MfnConv stem, dw_stem;
    MfnBlock blocks[MFN_MAX_BLOCKS];
    int n_blocks;
    MfnConv final_conv, gdconv, embed_proj;

    /* Raw file buffer — all weight pointers index into freed memory below. */
    uint8_t *raw;
    size_t raw_size;

    /* Workspace: two ping-pong NCHW buffers + a "residual save" slot. */
    float *work_a, *work_b, *residual_buf;
    size_t work_floats;
};


/* ============ Header parsing ============ */

static int parse_header(const uint8_t *raw, size_t raw_size,
                        MfnArch *out, size_t *out_payload_off,
                        uint32_t *out_n_tensors)
{
    if (raw_size < 4 + 4 + ARCH_HDR_SIZE + 4 + 4) return -1;
    if (memcmp(raw, EFM3_MAGIC, 4) != 0) return -2;
    uint32_t ver = *(const uint32_t*)(raw + 4);
    if (ver != EFM3_VERSION) return -3;

    const uint8_t *hdr = raw + 8;
    uint8_t hdr_v = hdr[0];
    if (hdr_v != 1) return -4;
    out->n_stages       = hdr[1];
    out->stem_kernel    = hdr[2];
    out->stem_stride    = hdr[3];
    out->stem_channels  = *(const uint16_t*)(hdr + 4);
    out->block_channels = *(const uint16_t*)(hdr + 6);
    out->final_channels = *(const uint16_t*)(hdr + 8);
    out->embedding_dim  = *(const uint16_t*)(hdr + 10);
    out->gdc_kernel     = hdr[12];

    if (out->n_stages > MFN_MAX_STAGES) return -5;
    for (int i = 0; i < out->n_stages; i++) {
        const uint8_t *p = hdr + HDR_FIXED_SIZE + i * STAGE_REC_SIZE;
        out->stage_t[i] = p[0];
        out->stage_n[i] = p[1];
        out->stage_s[i] = p[2];
    }

    /* Skip JSON debug header */
    size_t off = 8 + ARCH_HDR_SIZE;
    uint32_t json_len = *(const uint32_t*)(raw + off); off += 4;
    off += json_len;
    if (off + 4 > raw_size) return -6;
    *out_n_tensors = *(const uint32_t*)(raw + off); off += 4;
    *out_payload_off = off;
    return 0;
}


/* ============ Tensor table from .bin ============ */

typedef struct {
    float **tensors;
    int n;
} TensorTable;

static int build_tensor_table(const uint8_t *raw, size_t raw_size,
                               size_t payload_off, int n_expected,
                               TensorTable *out)
{
    out->tensors = (float**)calloc(n_expected, sizeof(float*));
    out->n = n_expected;
    size_t off = payload_off;
    for (int i = 0; i < n_expected; i++) {
        if (off + 4 > raw_size) return -1;
        uint32_t sz = *(const uint32_t*)(raw + off); off += 4;
        if (off + sz > raw_size) return -1;
        /* Note: pointer indexes into raw buffer — alignment may not be
         * 4-byte. On x86 this is fine; would need a copy on strict-align. */
        out->tensors[i] = (float*)(raw + off);
        off += sz;
    }
    return 0;
}


/* ============ Fold BatchNorm into preceding convolution ============
 *
 * Given Conv weight W [Cout, Cin/g, K, K], BN params (scale γ, bias β,
 * running mean μ, running var σ²) with eps=1e-5, the BN-folded conv is:
 *
 *   inv = γ / sqrt(σ² + eps)
 *   W'[c, *, *, *] = W[c, *, *, *] * inv[c]
 *   b'[c]          = β[c] - μ[c] * inv[c]
 *
 * After this we have y = conv(W') + b' which equals BN(conv(W)) exactly.
 */
static void fold_bn(float *w, int c_out, int weights_per_filter,
                    const float *bn_w, const float *bn_b,
                    const float *bn_mean, const float *bn_var,
                    float *out_bias, float eps)
{
    for (int c = 0; c < c_out; c++) {
        float inv = bn_w[c] / sqrtf(bn_var[c] + eps);
        for (int i = 0; i < weights_per_filter; i++) {
            w[c * weights_per_filter + i] *= inv;
        }
        out_bias[c] = bn_b[c] - bn_mean[c] * inv;
    }
}


/* ============ Tensor layout (mirrors binformat.tensor_layout) ============
 *
 * We walk the architecture in the same order Python writes tensors, and
 * for each conv layer pull (w, bn.w, bn.b, bn.mean, bn.var, [act.w]),
 * fold BN, and stash pointers into the MfnConv struct.
 */

static void load_conv(MfnConv *c, int c_in, int c_out, int k, int stride,
                       int padding, int groups, int with_act,
                       float **tensors, int *idx,
                       float *bias_storage, /* pre-allocated [c_out] */
                       float eps)
{
    float *w        = tensors[(*idx)++];
    float *bn_w     = tensors[(*idx)++];
    float *bn_b     = tensors[(*idx)++];
    float *bn_mean  = tensors[(*idx)++];
    float *bn_var   = tensors[(*idx)++];
    float *prelu    = with_act ? tensors[(*idx)++] : NULL;

    int weights_per_filter = (c_in / groups) * k * k;
    fold_bn(w, c_out, weights_per_filter, bn_w, bn_b, bn_mean, bn_var,
            bias_storage, eps);

    c->w = w;
    c->b = bias_storage;
    c->prelu = prelu;
    c->c_in = c_in;
    c->c_out = c_out;
    c->kernel = k;
    c->stride = stride;
    c->padding = padding;
    c->groups = groups;
}


/* ============ Public init ============ */

int mfn_engine_init(const char *path, MfnEngine *e)
{
    memset(e, 0, sizeof(*e));

    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    e->raw_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    e->raw = (uint8_t*)malloc(e->raw_size);
    if (!e->raw) { fclose(f); return -2; }
    if (fread(e->raw, 1, e->raw_size, f) != e->raw_size) { fclose(f); return -3; }
    fclose(f);

    size_t payload_off = 0;
    uint32_t n = 0;
    int rc = parse_header(e->raw, e->raw_size, &e->arch, &payload_off, &n);
    if (rc != 0) return -10 + rc;

    TensorTable tt = {0};
    if (build_tensor_table(e->raw, e->raw_size, payload_off, (int)n, &tt) != 0)
        return -20;

    const float eps = 1e-5f;
    int idx = 0;

    /* Pre-allocate one bias buffer per conv (BN-folded biases need new
     * storage since the original .bin has no bias slot for these convs). */
    /* Count convs first so we can allocate one big bias slab. */
    int n_convs = 2; /* stem, dw_stem */
    for (int i = 0; i < e->arch.n_stages; i++)
        n_convs += e->arch.stage_n[i] * 3;  /* expand, dw, project */
    n_convs += 3; /* final_conv, gdconv, embed_proj */

    /* Per-conv bias is sized by Cout; pre-compute total Cout sum. */
    int total_cout = 0;
    {
        int c_in = e->arch.stem_channels;
        total_cout += e->arch.stem_channels;     /* stem */
        total_cout += e->arch.stem_channels;     /* dw_stem */
        for (int i = 0; i < e->arch.n_stages; i++) {
            int c_out_stage = (i == 0) ? e->arch.stem_channels
                                         : e->arch.block_channels;
            int t = e->arch.stage_t[i];
            for (int j = 0; j < e->arch.stage_n[i]; j++) {
                int ci = (j == 0) ? c_in : c_out_stage;
                int hidden = ci * t;
                total_cout += hidden;             /* expand Cout */
                total_cout += hidden;             /* dw Cout */
                total_cout += c_out_stage;        /* project Cout */
            }
            c_in = c_out_stage;
        }
        total_cout += e->arch.final_channels;     /* final_conv */
        total_cout += e->arch.final_channels;     /* gdconv */
        total_cout += e->arch.embedding_dim;      /* embed_proj */
    }

    float *bias_slab = (float*)calloc((size_t)total_cout, sizeof(float));
    if (!bias_slab) return -30;
    float *bp = bias_slab;

    /* stem: Conv3 s=2 + BN + PReLU */
    load_conv(&e->stem, 3, e->arch.stem_channels,
              e->arch.stem_kernel, e->arch.stem_stride,
              e->arch.stem_kernel / 2, 1, /*with_act*/1,
              tt.tensors, &idx, bp, eps);
    bp += e->stem.c_out;

    /* dw_stem: DW Conv3 s=1 + BN + PReLU */
    load_conv(&e->dw_stem, e->arch.stem_channels, e->arch.stem_channels,
              3, 1, 1, e->arch.stem_channels, /*with_act*/1,
              tt.tensors, &idx, bp, eps);
    bp += e->dw_stem.c_out;

    /* bottleneck stages */
    int c_in = e->arch.stem_channels;
    e->n_blocks = 0;
    for (int i = 0; i < e->arch.n_stages; i++) {
        int c_out_stage = (i == 0) ? e->arch.stem_channels
                                     : e->arch.block_channels;
        int t = e->arch.stage_t[i];
        for (int j = 0; j < e->arch.stage_n[i]; j++) {
            MfnBlock *blk = &e->blocks[e->n_blocks++];
            int ci = (j == 0) ? c_in : c_out_stage;
            int stride = (j == 0) ? e->arch.stage_s[i] : 1;
            int hidden = ci * t;

            /* 1x1 expand + BN + PReLU */
            load_conv(&blk->expand, ci, hidden, 1, 1, 0, 1, 1,
                      tt.tensors, &idx, bp, eps);
            bp += blk->expand.c_out;

            /* DW 3x3 + BN + PReLU */
            load_conv(&blk->dw, hidden, hidden, 3, stride, 1, hidden, 1,
                      tt.tensors, &idx, bp, eps);
            bp += blk->dw.c_out;

            /* 1x1 project + BN (no activation — linear bottleneck) */
            load_conv(&blk->project, hidden, c_out_stage, 1, 1, 0, 1, 0,
                      tt.tensors, &idx, bp, eps);
            bp += blk->project.c_out;

            blk->has_residual = (stride == 1 && ci == c_out_stage) ? 1 : 0;
        }
        c_in = c_out_stage;
    }

    /* final_conv: 1x1 + BN + PReLU */
    load_conv(&e->final_conv, c_in, e->arch.final_channels, 1, 1, 0, 1, 1,
              tt.tensors, &idx, bp, eps);
    bp += e->final_conv.c_out;

    /* gdconv: DW kxk no padding + BN, no act */
    load_conv(&e->gdconv, e->arch.final_channels, e->arch.final_channels,
              e->arch.gdc_kernel, 1, 0, e->arch.final_channels, 0,
              tt.tensors, &idx, bp, eps);
    bp += e->gdconv.c_out;

    /* embed_proj: 1x1 + BN, no act */
    load_conv(&e->embed_proj, e->arch.final_channels, e->arch.embedding_dim,
              1, 1, 0, 1, 0,
              tt.tensors, &idx, bp, eps);

    if (idx != tt.n) {
        fprintf(stderr, "MFN: consumed %d tensors but file has %d\n", idx, tt.n);
        free(tt.tensors);
        free(bias_slab);
        return -40;
    }
    free(tt.tensors);

    /* Stash bias slab in raw_size+something? Actually we point at it from
     * each MfnConv.b. We need to remember to free it on engine_free.
     * Track it by hijacking work_a as the bias slab "owner"? No — add
     * a separate field. */
    /* Use the existing raw[] field for raw, and add a dedicated owner. */
    /* (Simplest: keep the bias_slab pointer in stem.b being the start —
     *  we'll free starting from bias_slab.) For that we need to remember
     *  where it starts. Save in a new field. */
    /* Workaround: stash in residual_buf temporarily; we'll allocate real
     * residual_buf below and free this slab on engine_free. To avoid
     * confusing the alias, expose via a private field. */
    /* Cleaner: add to struct. We did not — for now, the bias_slab leaks at
     * engine_free time. */
    /* TODO: leak fixed below by storing in residual_buf trick — see free path. */

    /* Allocate workspace: large enough for the biggest intermediate tensor.
     * After stem (s=2): 56*56*stem_c
     * After s=2 transitions, spatial shrinks: 56 -> 28 -> 14 -> 7 -> 1.
     * Channels grow: stem_c -> stem_c -> block_c -> ... -> final_c.
     * Worst case is usually the first DW post-expand at 56x56 spatial.
     */
    int H = MFN_INPUT_HW;
    int max_floats = 0;
    /* After stem: floor((112 - stem_k) / stem_stride) + 1 */
    int Hcur = (H - e->arch.stem_kernel) / e->arch.stem_stride + 1;
    int Ccur = e->arch.stem_channels;
    int blk_idx = 0;
    int stage_c_in = e->arch.stem_channels;
    int reserve = Hcur * Hcur * Ccur;
    if (reserve > max_floats) max_floats = reserve;
    /* dw_stem: same spatial */
    reserve = Hcur * Hcur * Ccur;
    if (reserve > max_floats) max_floats = reserve;
    for (int i = 0; i < e->arch.n_stages; i++) {
        int c_out_stage = (i == 0) ? e->arch.stem_channels
                                      : e->arch.block_channels;
        int t = e->arch.stage_t[i];
        for (int j = 0; j < e->arch.stage_n[i]; j++) {
            int ci = (j == 0) ? stage_c_in : c_out_stage;
            int hidden = ci * t;
            int stride = (j == 0) ? e->arch.stage_s[i] : 1;
            int Hin = Hcur;
            /* Expand: same spatial, Cout = hidden */
            reserve = Hin * Hin * hidden;
            if (reserve > max_floats) max_floats = reserve;
            /* DW with stride: spatial may halve */
            int Hout = (Hin - 3 + 2 * 1) / stride + 1;
            reserve = Hout * Hout * hidden;
            if (reserve > max_floats) max_floats = reserve;
            /* Project: same spatial as DW output, Cout = c_out_stage */
            reserve = Hout * Hout * c_out_stage;
            if (reserve > max_floats) max_floats = reserve;
            Hcur = Hout;
            blk_idx++;
        }
        stage_c_in = c_out_stage;
    }
    Ccur = e->arch.final_channels;
    reserve = Hcur * Hcur * Ccur;
    if (reserve > max_floats) max_floats = reserve;
    /* gdconv: 1x1 spatial */
    reserve = e->arch.embedding_dim;
    if (reserve > max_floats) max_floats = reserve;

    e->work_floats = (size_t)max_floats + 64;
    e->work_a = (float*)malloc(e->work_floats * sizeof(float));
    e->work_b = (float*)malloc(e->work_floats * sizeof(float));
    e->residual_buf = (float*)malloc(e->work_floats * sizeof(float));
    if (!e->work_a || !e->work_b || !e->residual_buf) return -50;

    /* Smuggle bias_slab so engine_free can release it. We tack it onto
     * the work allocation by placing the slab pointer just before
     * residual_buf — but cleaner: leave it as a small leak documented
     * in TODO. */
    /* Actually free path: bias_slab is referenced by stem.b. Track it. */
    /* Store its head in stem.b — but stem.b is already that. So just call
     * free(e->stem.b - 0) at free time, since stem is the first conv. */
    return 0;
}


/* ============ AVX2 specialized conv kernels ============ */

#ifdef __AVX2__
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif

/* 1x1 conv: y[co, hw] = sum_ci w[co, ci] * x[ci, hw] + b[co]
 * Most of MFN compute is here. We tile over output channels (4 at a
 * time) and use AVX2 along the spatial dim. */
static void conv2d_1x1_nchw(const float *x, const MfnConv *c,
                            int H, int W, float *y)
{
    int Cin = c->c_in, Cout = c->c_out;
    int HW = H * W;
    /* Weight layout from PyTorch: [Cout, Cin, 1, 1] -> [Cout, Cin] row-major. */
    const float *W_ = c->w;
    const float *bias = c->b;

#ifdef __AVX2__
    /* Process 1 Cout at a time, vectorize along HW. */
    for (int co = 0; co < Cout; co++) {
        float bi = bias[co];
        float *yc = y + (size_t)co * HW;
        /* init with bias */
        int i = 0;
        __m256 vb = _mm256_set1_ps(bi);
        for (; i + 8 <= HW; i += 8) _mm256_storeu_ps(yc + i, vb);
        for (; i < HW; i++) yc[i] = bi;
        /* accumulate sum_ci w[co, ci] * x[ci, hw] */
        for (int ci = 0; ci < Cin; ci++) {
            float wv = W_[co * Cin + ci];
            const float *xc = x + (size_t)ci * HW;
            __m256 vw = _mm256_set1_ps(wv);
            i = 0;
            for (; i + 8 <= HW; i += 8) {
                __m256 vy = _mm256_loadu_ps(yc + i);
                __m256 vx = _mm256_loadu_ps(xc + i);
                _mm256_storeu_ps(yc + i, _mm256_fmadd_ps(vw, vx, vy));
            }
            for (; i < HW; i++) yc[i] += wv * xc[i];
        }
    }
#else
    for (int co = 0; co < Cout; co++) {
        float bi = bias[co];
        for (int hw = 0; hw < HW; hw++) y[co * HW + hw] = bi;
        for (int ci = 0; ci < Cin; ci++) {
            float wv = W_[co * Cin + ci];
            for (int hw = 0; hw < HW; hw++) y[co * HW + hw] += wv * x[ci * HW + hw];
        }
    }
#endif
}


/* Depthwise 3x3 conv stride={1,2}, padding=1. Per-channel SIMD over spatial. */
static void conv2d_dw3x3_nchw(const float *x, const MfnConv *c,
                              int Hin, int Win, float *y)
{
    int S = c->stride;
    int C = c->c_in;  /* == c_out == groups */
    int Hout = (Hin + 2 - 3) / S + 1;
    int Wout = (Win + 2 - 3) / S + 1;
    const float *bias = c->b;

    for (int cc = 0; cc < C; cc++) {
        const float *xc = x + (size_t)cc * Hin * Win;
        const float *kw = c->w + (size_t)cc * 9;  /* 3x3 kernel for channel cc */
        float *yc = y + (size_t)cc * Hout * Wout;
        float bi = bias[cc];

        for (int oy = 0; oy < Hout; oy++) {
            for (int ox = 0; ox < Wout; ox++) {
                float sum = bi;
                int iy_base = oy * S - 1;
                int ix_base = ox * S - 1;
                for (int ky = 0; ky < 3; ky++) {
                    int iy = iy_base + ky;
                    if (iy < 0 || iy >= Hin) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = ix_base + kx;
                        if (ix < 0 || ix >= Win) continue;
                        sum += xc[iy * Win + ix] * kw[ky * 3 + kx];
                    }
                }
                yc[oy * Wout + ox] = sum;
            }
        }
    }
}


/* Depthwise k×k no padding -> single output position (GDConv).
 * Hout = Wout = 1 by construction. */
static void conv2d_gdconv_nchw(const float *x, const MfnConv *c,
                               int Hin, int Win, float *y)
{
    int K = c->kernel;
    int C = c->c_in;
    const float *bias = c->b;
    /* For each channel: y[c] = bias[c] + sum_ij w[c, i, j] * x[c, i, j] */
    for (int cc = 0; cc < C; cc++) {
        const float *xc = x + (size_t)cc * Hin * Win;
        const float *kw = c->w + (size_t)cc * K * K;
        float sum = bias[cc];
        int KK = K * K;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= KK; i += 8) {
            __m256 a = _mm256_loadu_ps(xc + i);
            __m256 b = _mm256_loadu_ps(kw + i);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        sum += hsum256(acc);
        for (; i < KK; i++) sum += xc[i] * kw[i];
#else
        for (int i = 0; i < KK; i++) sum += xc[i] * kw[i];
#endif
        y[cc] = sum;
    }
}


/* ============ NCHW conv2d (reference, fallback path) ============ */
/* y[B=1, Cout, Hout, Wout] = conv(x[1, Cin, Hin, Win], w[Cout, Cin/g, K, K]) + b */
static void conv2d_nchw(const float *x, const MfnConv *c,
                        int Hin, int Win, float *y)
{
    /* Fast paths */
    if (c->kernel == 1 && c->stride == 1 && c->padding == 0 && c->groups == 1) {
        conv2d_1x1_nchw(x, c, Hin, Win, y);
        return;
    }
    if (c->kernel == 3 && c->padding == 1 && c->groups == c->c_in
        && c->c_in == c->c_out) {
        conv2d_dw3x3_nchw(x, c, Hin, Win, y);
        return;
    }
    if (c->groups == c->c_in && c->c_in == c->c_out
        && c->padding == 0 && c->stride == 1
        && c->kernel == Hin && c->kernel == Win) {
        conv2d_gdconv_nchw(x, c, Hin, Win, y);
        return;
    }

    /* Generic reference (stem 3x3 s=2 falls here). */
    int K = c->kernel, S = c->stride, P = c->padding;
    int Cin = c->c_in, Cout = c->c_out, g = c->groups;
    int Hout = (Hin + 2 * P - K) / S + 1;
    int Wout = (Win + 2 * P - K) / S + 1;
    int Cin_per_g = Cin / g;
    int Cout_per_g = Cout / g;

    for (int og = 0; og < g; og++) {
        for (int oc = 0; oc < Cout_per_g; oc++) {
            int co = og * Cout_per_g + oc;
            float bias = c->b[co];
            for (int oy = 0; oy < Hout; oy++) {
                for (int ox = 0; ox < Wout; ox++) {
                    float sum = bias;
                    for (int ic = 0; ic < Cin_per_g; ic++) {
                        int ci = og * Cin_per_g + ic;
                        for (int ky = 0; ky < K; ky++) {
                            int iy = oy * S - P + ky;
                            if (iy < 0 || iy >= Hin) continue;
                            for (int kx = 0; kx < K; kx++) {
                                int ix = ox * S - P + kx;
                                if (ix < 0 || ix >= Win) continue;
                                float xv = x[((size_t)ci * Hin + iy) * Win + ix];
                                float wv = c->w[(((size_t)co * Cin_per_g + ic) * K + ky) * K + kx];
                                sum += xv * wv;
                            }
                        }
                    }
                    y[((size_t)co * Hout + oy) * Wout + ox] = sum;
                }
            }
        }
    }
}


/* ============ Activations ============ */
/* PReLU per-channel: y = (x > 0) ? x : slope[c] * x */
static void prelu_nchw(float *x, int C, int HW, const float *slope)
{
    for (int c = 0; c < C; c++) {
        float a = slope[c];
        float *row = x + (size_t)c * HW;
        int i = 0;
#ifdef __AVX2__
        __m256 va = _mm256_set1_ps(a);
        __m256 vz = _mm256_setzero_ps();
        for (; i + 8 <= HW; i += 8) {
            __m256 v = _mm256_loadu_ps(row + i);
            __m256 m = _mm256_cmp_ps(v, vz, _CMP_LT_OQ);
            __m256 neg = _mm256_mul_ps(v, va);
            _mm256_storeu_ps(row + i, _mm256_blendv_ps(v, neg, m));
        }
#endif
        for (; i < HW; i++) {
            if (row[i] < 0) row[i] *= a;
        }
    }
}

/* Residual add (in-place: y += x). */
static void add_inplace(float *y, const float *x, size_t n)
{
    size_t i = 0;
#ifdef __AVX2__
    for (; i + 8 <= n; i += 8) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_add_ps(vy, vx));
    }
#endif
    for (; i < n; i++) y[i] += x[i];
}


/* ============ Output Hout from Conv spec ============ */
static int conv_hout(const MfnConv *c, int Hin)
{
    return (Hin + 2 * c->padding - c->kernel) / c->stride + 1;
}

/* ============ Run one Conv (+ optional PReLU) ============ */
static int run_conv(const float *x, const MfnConv *c, int Hin, int Win,
                     float *y)
{
    conv2d_nchw(x, c, Hin, Win, y);
    int Hout = conv_hout(c, Hin);
    int Wout = conv_hout(c, Win);
    if (c->prelu) prelu_nchw(y, c->c_out, Hout * Wout, c->prelu);
    return Hout;
}


/* ============ Forward pass ============ */
void mfn_engine_forward(const MfnEngine *e, const float *input_chw, float *embedding)
{
    float *A = e->work_a;
    float *B = e->work_b;
    float *R = e->residual_buf;

    /* stem: 3x3 s=2, 112 -> 56 */
    int H = run_conv(input_chw, &e->stem, 112, 112, A);

    /* dw_stem: 3x3 s=1, 56 */
    H = run_conv(A, &e->dw_stem, H, H, B);
    /* B is now the current activation */
    float *cur = B, *nxt = A;

    /* bottleneck blocks */
    for (int bi = 0; bi < e->n_blocks; bi++) {
        const MfnBlock *blk = &e->blocks[bi];

        /* Save residual if needed */
        if (blk->has_residual) {
            memcpy(R, cur, (size_t)blk->expand.c_in * H * H * sizeof(float));
        }

        /* expand 1x1 + PReLU */
        run_conv(cur, &blk->expand, H, H, nxt);
        float *t = cur; cur = nxt; nxt = t;

        /* dw 3x3 (maybe stride 2) + PReLU */
        int Hnew = run_conv(cur, &blk->dw, H, H, nxt);
        t = cur; cur = nxt; nxt = t;
        H = Hnew;

        /* project 1x1 (linear, no act) */
        run_conv(cur, &blk->project, H, H, nxt);
        t = cur; cur = nxt; nxt = t;

        /* residual */
        if (blk->has_residual) {
            add_inplace(cur, R,
                        (size_t)blk->project.c_out * H * H);
        }
    }

    /* final_conv 1x1 + PReLU */
    run_conv(cur, &e->final_conv, H, H, nxt);
    { float *t = cur; cur = nxt; nxt = t; }

    /* gdconv kxk DW, no padding -> 1x1 spatial */
    run_conv(cur, &e->gdconv, H, H, nxt);
    { float *t = cur; cur = nxt; nxt = t; }
    /* Now spatial is 1x1, cur holds [final_c, 1, 1] */

    /* embed_proj 1x1 -> embedding_dim, 1x1 spatial */
    run_conv(cur, &e->embed_proj, 1, 1, nxt);
    /* nxt holds [embedding_dim, 1, 1] -- treat as [embedding_dim] */

    /* L2 normalize */
    int D = e->arch.embedding_dim;
    float norm = 0;
    for (int i = 0; i < D; i++) norm += nxt[i] * nxt[i];
    norm = 1.0f / sqrtf(norm + 1e-12f);
    for (int i = 0; i < D; i++) embedding[i] = nxt[i] * norm;
}


int mfn_embedding_dim(const MfnEngine *e) {
    return e ? e->arch.embedding_dim : 0;
}

float mfn_similarity(const float *a, const float *b, int dim) {
    float s = 0;
    for (int i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

void mfn_engine_free(MfnEngine *e) {
    if (!e) return;
    /* All BN-folded biases live in a contiguous slab whose head is
     * e->stem.b (first conv loaded). */
    if (e->stem.b) free(e->stem.b);
    free(e->raw);
    free(e->work_a);
    free(e->work_b);
    free(e->residual_buf);
    memset(e, 0, sizeof(*e));
}


/* ============ Optional standalone CLI ============ */
#ifdef MFN_STANDALONE
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s weights.bin\n", argv[0]); return 1; }
    MfnEngine e;
    int rc = mfn_engine_init(argv[1], &e);
    if (rc != 0) { fprintf(stderr, "init failed: %d\n", rc); return 1; }
    fprintf(stderr, "Loaded EFM3: stages=%d emb=%d  blocks=%d\n",
            e.arch.n_stages, e.arch.embedding_dim, e.n_blocks);

    static float input[3 * 112 * 112];
    for (int i = 0; i < 3 * 112 * 112; i++) input[i] = (float)(i % 256) / 128.0f - 1.0f;
    float *emb = (float*)malloc(e.arch.embedding_dim * sizeof(float));
    mfn_engine_forward(&e, input, emb);
    fprintf(stderr, "emb[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
            emb[0], emb[1], emb[2], emb[3], emb[4]);
    /* Check unit norm */
    float n = 0; for (int i = 0; i < e.arch.embedding_dim; i++) n += emb[i]*emb[i];
    fprintf(stderr, "||emb||^2 = %.6f (should be 1.0)\n", n);
    mfn_engine_free(&e);
    free(emb);
    return 0;
}
#endif
