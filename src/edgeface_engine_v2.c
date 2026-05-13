/*
 * edgeface_engine_v2.c — Parametric FP32 EdgeFace embedding engine.
 *
 * Reads the EFX2 .bin format (binary header describes architecture)
 * and runs the forward pass dynamically based on the loaded arch.
 * Supports the four FaceX size variants (Nano / Tiny / Standard / XS)
 * with one engine binary — "swap weights, same engine".
 *
 * The legacy EFXS (v1) format is still handled by edgeface_engine.c.
 * This file uses block primitives (convnext_block, xca_block,
 * conv2d_hwc, depthwise_conv) defined in edgeface_engine.c — to do
 * that, those functions are made non-static via FX_BLOCK_PUBLIC.
 *
 * Build: same flags as edgeface_engine.c (AVX2 + FMA), compile both
 * .c files into the library. Use facex_init() with an EFX2 weights
 * file to dispatch into v2.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* External primitives shared with v1 engine */
extern void layer_norm_fp32(const float*, int, int, const float*, const float*, float, float*);
extern void gelu_fp32(float*, int);
extern void l2_normalize_fp32(float*, int, int, float);
extern void depthwise_conv_nxn_hwc_fp32(const float*, int, int, int, const float*, const float*, int, float*);
extern void pack_b_fp32(const float*, int, int, float*);
extern int  packed_b_fp32_size(int, int);
extern void conv2d_hwc_reorder_weights(float* w, int Cout, int Cin, int KK);
extern void conv2d_hwc(const float* in_hwc, int Cin, int H, int W,
                       const float* w_packed, const float* b, int Cout,
                       int K, int stride, float* out_hwc);

/* PackedFP32 type — must match definition in edgeface_engine.c */
typedef struct {
    float* data;
    int K, N;
} PackedFP32;

/* Block primitives from edgeface_engine.c (now exposed by removing static). */
extern void convnext_block(float* x_hwc, int H, int W, int C,
                           const float* gamma,
                           const float* dw_w, const float* dw_b, int K,
                           const float* ln_w, const float* ln_b,
                           const float* mlp_w0, const float* mlp_w1,
                           const float* mlp_b1,
                           const float* mlp_w2, const float* mlp_w3,
                           const float* mlp_b3,
                           int rank, int hidden,
                           float* tmp,
                           const PackedFP32* fp0, const PackedFP32* fp1,
                           const PackedFP32* fp2, const PackedFP32* fp3);

extern void xca_block(float* x_hwc, int H, int W, int C,
                      const float* gamma_xca, const float* gamma,
                      int n_dw_splits, const int* dw_split_sizes,
                      int n_dw_convs, const float** dw_ws, const float** dw_bs, int dw_K,
                      const float* pos_const_nchw,
                      const float* pos_w, const float* pos_b,
                      const float* xca_ln_w, const float* xca_ln_b,
                      const float* qkv_w0, const float* qkv_w1, const float* qkv_b,
                      const float* temperature, int n_heads,
                      const float* proj_w0, const float* proj_w1, const float* proj_b,
                      const float* ln_w, const float* ln_b,
                      const float* mlp_w0, const float* mlp_w1, const float* mlp_b1,
                      const float* mlp_w2, const float* mlp_w3, const float* mlp_b3,
                      int rank, int hidden,
                      float* tmp,
                      const PackedFP32* fp_qkv0, const PackedFP32* fp_qkv1,
                      const PackedFP32* fp_proj0, const PackedFP32* fp_proj1,
                      const PackedFP32* fp_mlp0, const PackedFP32* fp_mlp1,
                      const PackedFP32* fp_mlp2, const PackedFP32* fp_mlp3);

/* ============ Architecture descriptor (loaded from .bin header) ============ */
#define V2_MAX_STAGES 6
#define V2_MAX_SPLITS 4

typedef struct {
    int channels;
    int n_convnext;
    int dw_kernel;
    int rank;
    int hidden;
    int has_xca;
    int has_pos_embed;
    int pos_hw;
    int n_dw_splits;
    int n_dw_convs;
    int dw_splits[V2_MAX_SPLITS];
    int xca_dw_kernel;
    int xca_n_heads;
    int has_downsample;
    int downsample_kernel;
    int downsample_stride;
    int downsample_to;
} V2_Stage;

typedef struct {
    int n_stages;
    int stem_in_c, stem_kernel, stem_stride;
    int stem_out;
    int head_rank, embedding_dim;
    V2_Stage stages[V2_MAX_STAGES];
} V2_Arch;

/* ============ Per-block weight pointers ============ */

typedef struct {
    const float *dw_w, *dw_b;
    const float *ln_w, *ln_b;
    /* MLP weights (PyTorch-native [out, in] orientation in the .bin —
     * we transpose at load time to [in, out] for the matmul kernel). */
    float *mlp_w0, *mlp_w1, *mlp_w2, *mlp_w3;
    const float *mlp_b1, *mlp_b3;
    const float *gamma;
    PackedFP32 fp0, fp1, fp2, fp3;
} V2_CN;

typedef struct {
    /* Cascaded DW conv */
    const float *dw_ws[V2_MAX_SPLITS];
    const float *dw_bs[V2_MAX_SPLITS];
    int dw_split_sizes[V2_MAX_SPLITS];
    /* Optional positional embedding */
    float *pos_const;        /* may be NULL; if present, pre-cached as HWC at load time */
    const float *pos_w, *pos_b;
    /* QKV projection (LoRA) */
    const float *ln_attn_w, *ln_attn_b;
    float *qkv_a_w, *qkv_b_w;
    const float *qkv_b_b;
    const float *temperature;
    /* Output projection (LoRA) */
    float *proj_a_w, *proj_b_w;
    const float *proj_b_b;
    const float *gamma_xca;
    /* MLP */
    const float *ln_mlp_w, *ln_mlp_b;
    float *mlp_w0, *mlp_w1, *mlp_w2, *mlp_w3;
    const float *mlp_b1, *mlp_b3;
    const float *gamma;
    /* Packed FP32 panels for matmul */
    PackedFP32 fp_qkv_a, fp_qkv_b;
    PackedFP32 fp_proj_a, fp_proj_b;
    PackedFP32 fp_mlp0, fp_mlp1, fp_mlp2, fp_mlp3;
} V2_XCA;

typedef struct {
    const float *ln_w, *ln_b;
    float *conv_w;             /* reordered at load time for HWC kernel */
    const float *conv_b;
} V2_DS;

typedef struct {
    V2_Stage cfg;
    V2_CN *cn_blocks;          /* cfg.n_convnext entries */
    V2_XCA xca;                /* valid iff cfg.has_xca */
    V2_DS  ds;                 /* valid iff cfg.has_downsample */
} V2_StageRT;

typedef struct {
    V2_Arch arch;
    V2_StageRT stages[V2_MAX_STAGES];

    const float *stem_conv_w;  /* [Cout, Cin, K, K] OIHW */
    const float *stem_conv_b;
    const float *stem_ln_w, *stem_ln_b;
    const float *head_ln_w, *head_ln_b;
    float *head_fc1_w;         /* PyTorch native [head_rank, last_C], used directly */
    float *head_fc2_w;         /* [emb, head_rank] */
    const float *head_fc2_b;

    /* raw file buffer (pointers above point into this) */
    uint8_t *raw;
    size_t raw_size;
    /* tensor table: pointers into raw, shape sizes (in bytes) */
    int n_tensors;
    float **tensors;
    uint32_t *tensor_sizes;   /* in bytes */

    /* workspace allocated dynamically based on max H*W*C */
    float *work;               /* scratch for blocks */
    size_t work_floats;
    float *x_buf;              /* primary HWC tensor */
    size_t x_floats;
} V2_Engine;


/* ============ Header parsing ============ */

static int v2_parse_header(const uint8_t *raw, size_t raw_size,
                            V2_Arch *out_arch, size_t *out_payload_off,
                            uint32_t *out_n_tensors)
{
    if (raw_size < 4 + 4 + 204 + 4) return -1;
    if (memcmp(raw, "EFX2", 4) != 0) return -2;
    uint32_t ver = *(const uint32_t*)(raw + 4);
    if (ver != 2) return -3;

    const uint8_t *hdr = raw + 8;
    /* fixed header (12 bytes): hdr_v u8, n_stages u8, stem_in_c u8,
     * stem_kernel u8, stem_stride u8, pad u8, stem_out u16, head_rank u16, embedding_dim u16 */
    uint8_t hdr_v = hdr[0];
    if (hdr_v != 1) return -4;
    int n_stages = hdr[1];
    if (n_stages < 1 || n_stages > V2_MAX_STAGES) return -5;
    out_arch->n_stages = n_stages;
    out_arch->stem_in_c = hdr[2];
    out_arch->stem_kernel = hdr[3];
    out_arch->stem_stride = hdr[4];
    /* pad hdr[5] */
    out_arch->stem_out      = *(const uint16_t*)(hdr + 6);
    out_arch->head_rank     = *(const uint16_t*)(hdr + 8);
    out_arch->embedding_dim = *(const uint16_t*)(hdr + 10);

    const uint8_t *st = hdr + 12;
    for (int i = 0; i < n_stages; i++) {
        const uint8_t *p = st + i * 32;
        V2_Stage *s = &out_arch->stages[i];
        s->channels        = *(const uint16_t*)(p + 0);
        s->n_convnext      = p[2];
        s->dw_kernel       = p[3];
        s->rank            = *(const uint16_t*)(p + 4);
        s->hidden          = *(const uint16_t*)(p + 6);
        s->has_xca         = p[8];
        s->has_pos_embed   = p[9];
        s->pos_hw          = p[10];
        s->n_dw_splits     = p[11];
        s->n_dw_convs      = p[12];
        s->xca_dw_kernel   = p[13];
        s->xca_n_heads     = p[14];
        /* p[15] pad */
        for (int k = 0; k < V2_MAX_SPLITS; k++) {
            s->dw_splits[k] = *(const uint16_t*)(p + 16 + k * 2);
        }
        /* p[24] has_downsample, p[25] kernel, p[26] stride, p[27] pad */
        s->has_downsample      = p[24];
        s->downsample_kernel   = p[25];
        s->downsample_stride   = p[26];
        s->downsample_to       = *(const uint16_t*)(p + 28);
    }

    /* Skip JSON header (length-prefixed) */
    size_t off = 8 + 204;
    uint32_t json_len = *(const uint32_t*)(raw + off); off += 4;
    off += json_len;
    if (off + 4 > raw_size) return -6;
    *out_n_tensors = *(const uint32_t*)(raw + off); off += 4;
    *out_payload_off = off;
    return 0;
}


/* ============ Tensor table builder ============ */
/* Walk the architecture in the same order as Python's tensor_layout(),
 * assigning sequential tensor indices to named pointers in the engine. */

static int v2_alloc_tensor_table(V2_Engine *e, size_t payload_off)
{
    /* First pass: scan record sizes to populate tensors[i] / tensor_sizes[i] */
    size_t off = payload_off;
    int idx = 0;
    while (off + 4 <= e->raw_size) {
        uint32_t sz = *(const uint32_t*)(e->raw + off); off += 4;
        if (off + sz > e->raw_size) return -1;
        if (idx >= e->n_tensors) return -1;
        e->tensors[idx] = (float*)(e->raw + off);
        e->tensor_sizes[idx] = sz;
        off += sz;
        idx++;
    }
    if (idx != e->n_tensors) return -2;
    return 0;
}

/* Helper: pull next tensor pointer, advance counter */
#define NEXT_T(e, i) ((const float*)((e)->tensors[(*(i))++]))
#define NEXT_T_RW(e, i) ((float*)((e)->tensors[(*(i))++]))

static int v2_assign_tensors(V2_Engine *e)
{
    int idx = 0;
    int *ip = &idx;
    e->stem_conv_w = NEXT_T(e, ip);
    e->stem_conv_b = NEXT_T(e, ip);
    e->stem_ln_w   = NEXT_T(e, ip);
    e->stem_ln_b   = NEXT_T(e, ip);

    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        rt->cfg = e->arch.stages[si];
        V2_Stage *s = &rt->cfg;

        rt->cn_blocks = (V2_CN*)calloc(s->n_convnext, sizeof(V2_CN));
        if (!rt->cn_blocks) return -10;

        for (int b = 0; b < s->n_convnext; b++) {
            V2_CN *cn = &rt->cn_blocks[b];
            cn->dw_w  = NEXT_T(e, ip);
            cn->dw_b  = NEXT_T(e, ip);
            cn->ln_w  = NEXT_T(e, ip);
            cn->ln_b  = NEXT_T(e, ip);
            cn->mlp_w0 = NEXT_T_RW(e, ip);   /* will transpose below */
            cn->mlp_w1 = NEXT_T_RW(e, ip);
            cn->mlp_b1 = NEXT_T(e, ip);
            cn->mlp_w2 = NEXT_T_RW(e, ip);
            cn->mlp_w3 = NEXT_T_RW(e, ip);
            cn->mlp_b3 = NEXT_T(e, ip);
            cn->gamma  = NEXT_T(e, ip);
        }

        if (s->has_xca) {
            V2_XCA *x = &rt->xca;
            for (int k = 0; k < s->n_dw_convs; k++) {
                x->dw_ws[k] = NEXT_T(e, ip);
                x->dw_bs[k] = NEXT_T(e, ip);
            }
            for (int k = 0; k < s->n_dw_splits; k++) {
                x->dw_split_sizes[k] = s->dw_splits[k];
            }
            if (s->has_pos_embed) {
                x->pos_const = NEXT_T_RW(e, ip);     /* [1, C, H, W] NCHW */
                x->pos_w     = NEXT_T(e, ip);
                x->pos_b     = NEXT_T(e, ip);
            } else {
                x->pos_const = NULL;
                x->pos_w = x->pos_b = NULL;
            }
            x->ln_attn_w = NEXT_T(e, ip);
            x->ln_attn_b = NEXT_T(e, ip);
            x->qkv_a_w   = NEXT_T_RW(e, ip);
            x->qkv_b_w   = NEXT_T_RW(e, ip);
            x->qkv_b_b   = NEXT_T(e, ip);
            x->temperature = NEXT_T(e, ip);
            x->proj_a_w  = NEXT_T_RW(e, ip);
            x->proj_b_w  = NEXT_T_RW(e, ip);
            x->proj_b_b  = NEXT_T(e, ip);
            x->gamma_xca = NEXT_T(e, ip);
            x->ln_mlp_w  = NEXT_T(e, ip);
            x->ln_mlp_b  = NEXT_T(e, ip);
            x->mlp_w0    = NEXT_T_RW(e, ip);
            x->mlp_w1    = NEXT_T_RW(e, ip);
            x->mlp_b1    = NEXT_T(e, ip);
            x->mlp_w2    = NEXT_T_RW(e, ip);
            x->mlp_w3    = NEXT_T_RW(e, ip);
            x->mlp_b3    = NEXT_T(e, ip);
            x->gamma     = NEXT_T(e, ip);
        }

        if (s->has_downsample) {
            rt->ds.ln_w   = NEXT_T(e, ip);
            rt->ds.ln_b   = NEXT_T(e, ip);
            rt->ds.conv_w = NEXT_T_RW(e, ip);
            rt->ds.conv_b = NEXT_T(e, ip);
        }
    }

    e->head_ln_w   = NEXT_T(e, ip);
    e->head_ln_b   = NEXT_T(e, ip);
    e->head_fc1_w  = NEXT_T_RW(e, ip);
    e->head_fc2_w  = NEXT_T_RW(e, ip);
    e->head_fc2_b  = NEXT_T(e, ip);

    if (idx != e->n_tensors) return -20;
    return 0;
}


/* ============ Linear weight transpose [out, in] -> [in, out] ============ */
/* PyTorch nn.Linear stores weight as [out, in]. The convnext_block /
 * xca_block primitives expect [in, out] (so that A[M,in] @ W[in,out] = C[M,out]).
 * Transpose in-place in the buffer (memory was malloc'd as part of the file). */
static void transpose_inplace(float *w, int rows, int cols)
{
    /* Transpose rows x cols matrix in row-major into a temporary, then memcpy */
    float *tmp = (float*)malloc((size_t)rows * cols * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            tmp[c * rows + r] = w[r * cols + c];
    memcpy(w, tmp, (size_t)rows * cols * sizeof(float));
    free(tmp);
}

/* For LoRA projection MLP, the weight orientation matters:
 *   w0 = Linear(c, rank).weight        -> stored [rank, c]; want [c, rank]
 *   w1 = Linear(rank, hidden).weight   -> stored [hidden, rank]; want [rank, hidden]
 *   w2 = Linear(hidden, rank).weight   -> stored [rank, hidden]; want [hidden, rank]
 *   w3 = Linear(rank, c).weight        -> stored [c, rank]; want [rank, c]
 */
static void v2_transpose_mlp(V2_CN *cn, int C, int rank, int hidden)
{
    transpose_inplace(cn->mlp_w0, rank, C);
    transpose_inplace(cn->mlp_w1, hidden, rank);
    transpose_inplace(cn->mlp_w2, rank, hidden);
    transpose_inplace(cn->mlp_w3, C, rank);
}

static void v2_transpose_xca(V2_XCA *x, int C, int rank, int hidden)
{
    /* QKV: a = Linear(C, rank), b = Linear(rank, 3C) */
    transpose_inplace(x->qkv_a_w, rank, C);
    transpose_inplace(x->qkv_b_w, 3 * C, rank);
    /* proj: a = Linear(C, rank), b = Linear(rank, C) */
    transpose_inplace(x->proj_a_w, rank, C);
    transpose_inplace(x->proj_b_w, C, rank);
    /* MLP same as ConvNeXt */
    transpose_inplace(x->mlp_w0, rank, C);
    transpose_inplace(x->mlp_w1, hidden, rank);
    transpose_inplace(x->mlp_w2, rank, hidden);
    transpose_inplace(x->mlp_w3, C, rank);
}


/* ============ Pre-pack FP32 MatMul weights into column-panel format ============ */

static int v2_prepack_fp(float **packed_data, int K, int N)
{
    int sz = packed_b_fp32_size(K, N);
    float *p = NULL;
#ifdef _WIN32
    p = (float*)_aligned_malloc((size_t)sz * sizeof(float), 64);
#else
    if (posix_memalign((void**)&p, 64, (size_t)sz * sizeof(float)) != 0) p = NULL;
#endif
    if (!p) return -1;
    *packed_data = p;
    return sz;
}

static void v2_pack_cn(V2_CN *cn, int C, int rank, int hidden)
{
    int sz;
    sz = v2_prepack_fp(&cn->fp0.data, C, rank);    cn->fp0.K = C;      cn->fp0.N = rank;
    pack_b_fp32(cn->mlp_w0, C, rank, cn->fp0.data);
    sz = v2_prepack_fp(&cn->fp1.data, rank, hidden); cn->fp1.K = rank; cn->fp1.N = hidden;
    pack_b_fp32(cn->mlp_w1, rank, hidden, cn->fp1.data);
    sz = v2_prepack_fp(&cn->fp2.data, hidden, rank); cn->fp2.K = hidden; cn->fp2.N = rank;
    pack_b_fp32(cn->mlp_w2, hidden, rank, cn->fp2.data);
    sz = v2_prepack_fp(&cn->fp3.data, rank, C);    cn->fp3.K = rank;   cn->fp3.N = C;
    pack_b_fp32(cn->mlp_w3, rank, C, cn->fp3.data);
    (void)sz;
}

static void v2_pack_xca(V2_XCA *x, int C, int rank, int hidden)
{
    int sz;
    sz = v2_prepack_fp(&x->fp_qkv_a.data, C, rank);    x->fp_qkv_a.K = C;    x->fp_qkv_a.N = rank;
    pack_b_fp32(x->qkv_a_w, C, rank, x->fp_qkv_a.data);
    sz = v2_prepack_fp(&x->fp_qkv_b.data, rank, 3*C);  x->fp_qkv_b.K = rank; x->fp_qkv_b.N = 3*C;
    pack_b_fp32(x->qkv_b_w, rank, 3*C, x->fp_qkv_b.data);
    sz = v2_prepack_fp(&x->fp_proj_a.data, C, rank);   x->fp_proj_a.K = C;   x->fp_proj_a.N = rank;
    pack_b_fp32(x->proj_a_w, C, rank, x->fp_proj_a.data);
    sz = v2_prepack_fp(&x->fp_proj_b.data, rank, C);   x->fp_proj_b.K = rank; x->fp_proj_b.N = C;
    pack_b_fp32(x->proj_b_w, rank, C, x->fp_proj_b.data);
    sz = v2_prepack_fp(&x->fp_mlp0.data, C, rank);     x->fp_mlp0.K = C;     x->fp_mlp0.N = rank;
    pack_b_fp32(x->mlp_w0, C, rank, x->fp_mlp0.data);
    sz = v2_prepack_fp(&x->fp_mlp1.data, rank, hidden); x->fp_mlp1.K = rank; x->fp_mlp1.N = hidden;
    pack_b_fp32(x->mlp_w1, rank, hidden, x->fp_mlp1.data);
    sz = v2_prepack_fp(&x->fp_mlp2.data, hidden, rank); x->fp_mlp2.K = hidden; x->fp_mlp2.N = rank;
    pack_b_fp32(x->mlp_w2, hidden, rank, x->fp_mlp2.data);
    sz = v2_prepack_fp(&x->fp_mlp3.data, rank, C);     x->fp_mlp3.K = rank;  x->fp_mlp3.N = C;
    pack_b_fp32(x->mlp_w3, rank, C, x->fp_mlp3.data);
    (void)sz;
}


/* ============ Public init / forward / free ============ */

int v2_engine_init(const char *path, V2_Engine *e)
{
    /* Load file into memory */
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    e->raw_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    e->raw = (uint8_t*)malloc(e->raw_size);
    if (!e->raw) { fclose(f); return -2; }
    if (fread(e->raw, 1, e->raw_size, f) != e->raw_size) { fclose(f); return -3; }
    fclose(f);

    /* Parse header */
    size_t payload_off = 0;
    uint32_t n = 0;
    int rc = v2_parse_header(e->raw, e->raw_size, &e->arch, &payload_off, &n);
    if (rc != 0) return -10 + rc;
    e->n_tensors = (int)n;

    /* Build tensor table */
    e->tensors = (float**)calloc(n, sizeof(float*));
    e->tensor_sizes = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (v2_alloc_tensor_table(e, payload_off) != 0) return -20;

    /* Assign named pointers */
    if (v2_assign_tensors(e) != 0) return -21;

    /* Transpose all Linear weights from PyTorch [out,in] to [in,out] */
    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        V2_Stage *s = &rt->cfg;
        for (int b = 0; b < s->n_convnext; b++) {
            v2_transpose_mlp(&rt->cn_blocks[b], s->channels, s->rank, s->hidden);
        }
        if (s->has_xca) {
            v2_transpose_xca(&rt->xca, s->channels, s->rank, s->hidden);
        }
    }
    /* Head FC1: PyTorch shape [head_rank, last_C], we use it as is for the
     * vector-style dot product in the head (M=1 so transpose isn't needed).
     * Same for FC2 [emb, head_rank]. */

    /* Pre-pack FP32 MatMul weights */
    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        V2_Stage *s = &rt->cfg;
        for (int b = 0; b < s->n_convnext; b++) {
            v2_pack_cn(&rt->cn_blocks[b], s->channels, s->rank, s->hidden);
        }
        if (s->has_xca) v2_pack_xca(&rt->xca, s->channels, s->rank, s->hidden);
    }

    /* Reorder downsample conv weights from OIHW to HWC kernel layout */
    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        V2_Stage *s = &rt->cfg;
        if (s->has_downsample) {
            int K = s->downsample_kernel;
            conv2d_hwc_reorder_weights(rt->ds.conv_w,
                                        s->downsample_to, s->channels, K * K);
        }
    }
    /* Stem conv kept in OIHW (handled by stem inline forward below). */

    /* Pre-cache positional embedding (Conv1x1 of constant) — done at load
     * time in v1 for stage1; we generalize. Result is [HW, C] HWC stored
     * back into pos_const slot. */
    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        V2_Stage *s = &rt->cfg;
        if (s->has_xca && s->has_pos_embed && rt->xca.pos_const) {
            int HW = s->pos_hw * s->pos_hw;
            int C = s->channels;
            float *cached = (float*)malloc((size_t)HW * C * sizeof(float));
            const float *pc = rt->xca.pos_const;     /* [1, C, H, W] NCHW */
            const float *pw = rt->xca.pos_w;         /* [C_out, C_in, 1, 1] = [C, C] effectively */
            const float *pb = rt->xca.pos_b;
            for (int hw = 0; hw < HW; hw++) {
                for (int co = 0; co < C; co++) {
                    float sum = pb[co];
                    for (int ci = 0; ci < C; ci++) {
                        sum += pc[ci * HW + hw] * pw[co * C + ci];
                    }
                    cached[hw * C + co] = sum;
                }
            }
            rt->xca.pos_const = cached;  /* leak the original — small constant */
        }
    }

    /* Allocate workspace based on max H*W*C across stages, plus margin
     * for residual + dw_out + t1 + t2 (used in convnext_block). */
    size_t max_x = 0;
    int H = 112 / e->arch.stem_stride;     /* spatial after stem */
    int W = H;
    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        V2_Stage *s = &rt->cfg;
        size_t hwc = (size_t)H * W * s->channels;
        if (hwc > max_x) max_x = hwc;
        size_t wb = hwc * 6 + (size_t)H * W * s->hidden + (size_t)H * W * s->rank * 3;
        if (wb > e->work_floats) e->work_floats = wb;
        if (s->has_downsample) {
            H = (H - s->downsample_kernel) / s->downsample_stride + 1;
            W = H;
        }
    }
    e->x_floats = max_x + 64;          /* small margin */
    e->x_buf  = (float*)malloc(e->x_floats * sizeof(float));
    e->work   = (float*)malloc(e->work_floats * sizeof(float));
    if (!e->x_buf || !e->work) return -30;

    return 0;
}


/* ============ Stem Conv 3 -> stem_out (k x k, stride s) ============ */
static void v2_stem(const float *input_chw, const V2_Engine *e, float *out_hwc)
{
    int K = e->arch.stem_kernel;
    int S = e->arch.stem_stride;
    int Cout = e->arch.stem_out;
    int Cin  = e->arch.stem_in_c;
    int Hout = (112 - K) / S + 1;
    int Wout = Hout;
    const float *sw = e->stem_conv_w;     /* [Cout, Cin, K, K] OIHW */
    const float *sb = e->stem_conv_b;     /* [Cout] */

    for (int oy = 0; oy < Hout; oy++) {
        for (int ox = 0; ox < Wout; ox++) {
            float *o = out_hwc + (oy * Wout + ox) * Cout;
            for (int co = 0; co < Cout; co++) o[co] = sb[co];
            for (int ci = 0; ci < Cin; ci++) {
                for (int ky = 0; ky < K; ky++) {
                    for (int kx = 0; kx < K; kx++) {
                        float iv = input_chw[(size_t)ci * 112 * 112
                                              + (oy * S + ky) * 112 + ox * S + kx];
                        for (int co = 0; co < Cout; co++) {
                            o[co] += iv * sw[((co * Cin + ci) * K + ky) * K + kx];
                        }
                    }
                }
            }
        }
    }
}

/* ============ Forward pass ============ */
void v2_engine_forward(const V2_Engine *e, const float *input_chw, float *embedding)
{
    float *x = e->x_buf;
    int H = (112 - e->arch.stem_kernel) / e->arch.stem_stride + 1;
    int W = H;
    int C = e->arch.stem_out;

    /* Stem */
    v2_stem(input_chw, e, x);
    layer_norm_fp32(x, H * W, C, e->stem_ln_w, e->stem_ln_b, 1e-6f, x);

    /* Stages */
    for (int si = 0; si < e->arch.n_stages; si++) {
        const V2_StageRT *rt = &e->stages[si];
        const V2_Stage *s = &rt->cfg;

        /* ConvNeXt blocks */
        for (int b = 0; b < s->n_convnext; b++) {
            const V2_CN *cn = &rt->cn_blocks[b];
            convnext_block(x, H, W, C,
                           cn->gamma,
                           cn->dw_w, cn->dw_b, s->dw_kernel,
                           cn->ln_w, cn->ln_b,
                           cn->mlp_w0, cn->mlp_w1, cn->mlp_b1,
                           cn->mlp_w2, cn->mlp_w3, cn->mlp_b3,
                           s->rank, s->hidden, e->work,
                           &cn->fp0, &cn->fp1, &cn->fp2, &cn->fp3);
        }

        /* XCA block */
        if (s->has_xca) {
            const V2_XCA *xc = &rt->xca;
            xca_block(x, H, W, C,
                      xc->gamma_xca, xc->gamma,
                      s->n_dw_splits, xc->dw_split_sizes,
                      s->n_dw_convs, xc->dw_ws, xc->dw_bs, s->xca_dw_kernel,
                      xc->pos_const, xc->pos_w, xc->pos_b,
                      xc->ln_attn_w, xc->ln_attn_b,
                      xc->qkv_a_w, xc->qkv_b_w, xc->qkv_b_b,
                      xc->temperature, s->xca_n_heads,
                      xc->proj_a_w, xc->proj_b_w, xc->proj_b_b,
                      xc->ln_mlp_w, xc->ln_mlp_b,
                      xc->mlp_w0, xc->mlp_w1, xc->mlp_b1,
                      xc->mlp_w2, xc->mlp_w3, xc->mlp_b3,
                      s->rank, s->hidden, e->work,
                      &xc->fp_qkv_a, &xc->fp_qkv_b,
                      &xc->fp_proj_a, &xc->fp_proj_b,
                      &xc->fp_mlp0, &xc->fp_mlp1, &xc->fp_mlp2, &xc->fp_mlp3);
        }

        /* Downsample */
        if (s->has_downsample) {
            layer_norm_fp32(x, H * W, C, rt->ds.ln_w, rt->ds.ln_b, 1e-6f, x);
            int Hout = (H - s->downsample_kernel) / s->downsample_stride + 1;
            int Wout = Hout;
            int Cout = s->downsample_to;
            float *tmp = (float*)malloc((size_t)Hout * Wout * Cout * sizeof(float));
            conv2d_hwc(x, C, H, W, rt->ds.conv_w, rt->ds.conv_b,
                       Cout, s->downsample_kernel, s->downsample_stride, tmp);
            memcpy(x, tmp, (size_t)Hout * Wout * Cout * sizeof(float));
            free(tmp);
            H = Hout; W = Wout; C = Cout;
        }
    }

    /* Head: GAP -> LN -> FC1 -> FC2 -> L2 norm */
    int last_c = C;
    int rank = e->arch.head_rank;
    int emb = e->arch.embedding_dim;
    int HW = H * W;

    float *pooled = (float*)malloc(last_c * sizeof(float));
    for (int c = 0; c < last_c; c++) {
        float sum = 0;
        for (int hw = 0; hw < HW; hw++) sum += x[hw * last_c + c];
        pooled[c] = sum / HW;
    }
    layer_norm_fp32(pooled, 1, last_c, e->head_ln_w, e->head_ln_b, 1e-6f, pooled);

    float *fc1 = (float*)malloc(rank * sizeof(float));
    /* head_fc1_w stored as PyTorch [rank, last_c]. Compute fc1[n] = sum_k pooled[k] * W[n,k]. */
    for (int n = 0; n < rank; n++) {
        float sum = 0;
        for (int k = 0; k < last_c; k++) sum += pooled[k] * e->head_fc1_w[n * last_c + k];
        fc1[n] = sum;
    }

    /* head_fc2_w stored as PyTorch [emb, rank]. */
    for (int n = 0; n < emb; n++) {
        float sum = e->head_fc2_b[n];
        for (int k = 0; k < rank; k++) sum += fc1[k] * e->head_fc2_w[n * rank + k];
        embedding[n] = sum;
    }

    /* L2 normalize */
    float norm = 0;
    for (int n = 0; n < emb; n++) norm += embedding[n] * embedding[n];
    norm = 1.0f / sqrtf(norm + 1e-12f);
    for (int n = 0; n < emb; n++) embedding[n] *= norm;

    free(pooled);
    free(fc1);
}


int v2_embedding_dim(const V2_Engine *e) { return e ? e->arch.embedding_dim : 0; }

void v2_engine_free(V2_Engine *e)
{
    if (!e) return;
    for (int si = 0; si < e->arch.n_stages; si++) {
        V2_StageRT *rt = &e->stages[si];
        V2_Stage *s = &rt->cfg;
        if (rt->cn_blocks) {
            for (int b = 0; b < s->n_convnext; b++) {
                V2_CN *cn = &rt->cn_blocks[b];
#ifdef _WIN32
                if (cn->fp0.data) _aligned_free(cn->fp0.data);
                if (cn->fp1.data) _aligned_free(cn->fp1.data);
                if (cn->fp2.data) _aligned_free(cn->fp2.data);
                if (cn->fp3.data) _aligned_free(cn->fp3.data);
#else
                free(cn->fp0.data); free(cn->fp1.data); free(cn->fp2.data); free(cn->fp3.data);
#endif
            }
            free(rt->cn_blocks);
        }
        if (s->has_xca) {
            V2_XCA *x = &rt->xca;
#ifdef _WIN32
#define FREE_PACKED(p) if ((p).data) _aligned_free((p).data)
#else
#define FREE_PACKED(p) free((p).data)
#endif
            FREE_PACKED(x->fp_qkv_a); FREE_PACKED(x->fp_qkv_b);
            FREE_PACKED(x->fp_proj_a); FREE_PACKED(x->fp_proj_b);
            FREE_PACKED(x->fp_mlp0); FREE_PACKED(x->fp_mlp1);
            FREE_PACKED(x->fp_mlp2); FREE_PACKED(x->fp_mlp3);
#undef FREE_PACKED
        }
    }
    free(e->tensors);
    free(e->tensor_sizes);
    free(e->raw);
    free(e->x_buf);
    free(e->work);
}


/* ============ Optional standalone CLI for v2 ============ */
#ifdef V2_STANDALONE
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s weights.bin\n", argv[0]); return 1; }
    extern void tp_init(int);
    tp_init(4);
    V2_Engine e = {0};
    int rc = v2_engine_init(argv[1], &e);
    if (rc != 0) { fprintf(stderr, "init failed: %d\n", rc); return 1; }
    fprintf(stderr, "Loaded EFX2: stages=%d emb_dim=%d\n",
            e.arch.n_stages, e.arch.embedding_dim);

    float input[3 * 112 * 112];
    for (int i = 0; i < 3 * 112 * 112; i++) input[i] = (float)(i % 256) / 128.0f - 1.0f;
    float *embedding = (float*)malloc(e.arch.embedding_dim * sizeof(float));
    v2_engine_forward(&e, input, embedding);
    fprintf(stderr, "emb[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
            embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]);
    v2_engine_free(&e);
    free(embedding);
    return 0;
}
#endif
