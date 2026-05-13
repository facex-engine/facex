/*
 * nn2.c — Public API: init, detect, free.
 *
 * Preprocessing: resize + normalize (0-255 → 0-1) + HWC→CHW.
 */

#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

const char* nn2_version(void) { return "0.1.0"; }

/* Declared in net.c */
int nn2_load_weights(NN2* ctx, const char* path);

NN2* nn2_init(const char* weights_path, int n_threads)
{
    NN2* ctx = (NN2*)calloc(1, sizeof(NN2));
    if (!ctx) return NULL;

    ctx->conf_thresh = 0.25f;
    ctx->nms_thresh = 0.45f;

    /* DFL weights: [0, 1, 2, ..., 15] */
    for (int i = 0; i < NN2_REG_MAX; i++)
        ctx->dfl_weight[i] = (float)i;

    /* Load weights */
    if (nn2_load_weights(ctx, weights_path) != 0) {
        free(ctx);
        return NULL;
    }

    /* Allocate workspace: 256MB for 640, 64MB for 320 */
    ctx->ws_size = (ctx->input_size >= 640) ? (256 << 20) : (64 << 20);
    ctx->workspace = (float*)malloc(ctx->ws_size);
    if (!ctx->workspace) {
        fprintf(stderr, "nn2: workspace alloc failed\n");
        nn2_free(ctx);
        return NULL;
    }

    /* Pre-pack weight matrices for all conv layers (BLIS-style A-packing).
     * Eliminates stride-lda access during GEMM — weights become sequential in memory. */
    {
        /* Helper to pack a ConvLayer's weights */
        #define PACK_CONV(layer) do { \
            int _K = (layer)->cin * (layer)->k * (layer)->k; \
            (layer)->w_packed = nn2_pack_weight_a((layer)->w, (layer)->cout, _K); \
        } while(0)

        for (int i = 0; i < 5; i++) PACK_CONV(&ctx->bb_conv[i]);
        for (int i = 0; i < 4; i++) {
            PACK_CONV(&ctx->bb_c2f[i].cv1);
            PACK_CONV(&ctx->bb_c2f[i].cv2);
            for (int j = 0; j < ctx->bb_c2f[i].n; j++) {
                PACK_CONV(&ctx->bb_c2f[i].m[j].cv1);
                PACK_CONV(&ctx->bb_c2f[i].m[j].cv2);
            }
        }
        PACK_CONV(&ctx->sppf_cv1); PACK_CONV(&ctx->sppf_cv2);
        for (int i = 0; i < 4; i++) {
            PACK_CONV(&ctx->nk_c2f[i].cv1);
            PACK_CONV(&ctx->nk_c2f[i].cv2);
            for (int j = 0; j < ctx->nk_c2f[i].n; j++) {
                PACK_CONV(&ctx->nk_c2f[i].m[j].cv1);
                PACK_CONV(&ctx->nk_c2f[i].m[j].cv2);
            }
        }
        for (int i = 0; i < 2; i++) PACK_CONV(&ctx->nk_conv[i]);
        for (int s = 0; s < 3; s++) {
            for (int j = 0; j < 3; j++) { PACK_CONV(&ctx->head[s].bbox[j]); PACK_CONV(&ctx->head[s].cls[j]); }
        }
        #undef PACK_CONV
    }

    /* Detect fused normalization: if first conv weights are very small (< 0.1),
     * they were pre-divided by 255 at export time → skip normalize in preprocessing. */
    {
        float maxw = 0;
        int wsize = ctx->bb_conv[0].cout * ctx->bb_conv[0].cin *
                    ctx->bb_conv[0].k * ctx->bb_conv[0].k;
        for (int i = 0; i < wsize; i++) {
            float a = ctx->bb_conv[0].w[i];
            if (a < 0) a = -a;
            if (a > maxw) maxw = a;
        }
        ctx->_fused_norm = (maxw < 0.5f) ? 1 : 0;  /* normal weights are ~1-20 range */
        if (ctx->_fused_norm)
            fprintf(stderr, "nn2: detected fused normalization in weights\n");
    }

    /* Thread pool */
    ctx->n_threads = n_threads;
    nn2_tp_init(n_threads);

    fprintf(stderr, "nn2: loaded model (classes=%d, input=%d, threads=%d)\n",
            ctx->num_classes, ctx->input_size, nn2_tp_num_threads());

    return ctx;
}

/* ============ Preprocessing ============ */

typedef struct { const uint8_t* rgb; float* ch0; float* ch1; float* ch2;
                 int src_w,src_h,S,new_w,pad_x,pad_y; float inv_scale,inv255; } PPCtx;

void preprocess_worker(void* arg, int y_start, int y_end) {
    PPCtx* p = (PPCtx*)arg;
    for (int y = y_start; y < y_end; y++) {
        float sy = y * p->inv_scale;
        int iy = (int)sy; float fy = sy - iy;
        if (iy >= p->src_h - 1) { iy = p->src_h - 2; fy = 1.0f; }
        float fy1 = 1.0f - fy;
        int dy = y + p->pad_y;
        if (dy >= p->S) continue;
        const uint8_t* row0 = p->rgb + iy * p->src_w * 3;
        const uint8_t* row1 = row0 + p->src_w * 3;
        for (int x = 0; x < p->new_w; x++) {
            float sx = x * p->inv_scale;
            int ix = (int)sx; float fx = sx - ix;
            if (ix >= p->src_w - 1) { ix = p->src_w - 2; fx = 1.0f; }
            float fx1 = 1.0f - fx;
            int dx = x + p->pad_x;
            if (dx >= p->S) continue;
            int idx = dy * p->S + dx;
            const uint8_t* p00 = row0+ix*3, *p01 = p00+3;
            const uint8_t* p10 = row1+ix*3, *p11 = p10+3;
            float w00=fx1*fy1, w01=fx*fy1, w10=fx1*fy, w11=fx*fy;
            p->ch0[idx] = (p00[0]*w00+p01[0]*w01+p10[0]*w10+p11[0]*w11) * p->inv255;
            p->ch1[idx] = (p00[1]*w00+p01[1]*w01+p10[1]*w10+p11[1]*w11) * p->inv255;
            p->ch2[idx] = (p00[2]*w00+p01[2]*w01+p10[2]*w10+p11[2]*w11) * p->inv255;
        }
    }
}

static void preprocess(const uint8_t* rgb_hwc, int src_w, int src_h,
                       int dst_size, float* chw_out, int fused_norm)
{
    float scale = (float)dst_size / (src_w > src_h ? src_w : src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    int pad_x = (dst_size - new_w) / 2;
    int pad_y = (dst_size - new_h) / 2;
    int S = dst_size;
    /* If normalization fused into first conv, skip /255 here — just cast to float */
    float inv255 = fused_norm ? 1.0f : (1.0f / 255.0f);
    float gray = fused_norm ? 114.0f : (114.0f / 255.0f);

    /* Init gray padding */
    int total = 3 * S * S;
    int i = 0;
#ifdef __AVX2__
    __m256 vg = _mm256_set1_ps(gray);
    for (; i + 8 <= total; i += 8)
        _mm256_storeu_ps(chw_out + i, vg);
#endif
    for (; i < total; i++)
        chw_out[i] = gray;

    /* HWC→CHW + normalize (with bilinear resize if needed) */
    float inv_scale = 1.0f / scale;
    float* ch0 = chw_out;
    float* ch1 = chw_out + S * S;
    float* ch2 = chw_out + 2 * S * S;

    /* Fast path: scale is exactly 1.0 (input fits width) */
    if (new_w == src_w && scale > 0.999f && scale < 1.001f) {
        for (int y = 0; y < new_h; y++) {
            int dy = y + pad_y;
            const uint8_t* row = rgb_hwc + y * src_w * 3;
            int base = dy * S + pad_x;
            int x = 0;
#ifdef __AVX2__
            /* Process 8 pixels: load 24 bytes RGB, deinterleave, convert float, normalize */
            __m256 vinv = _mm256_set1_ps(1.0f / 255.0f);
            for (; x + 8 <= new_w; x += 8) {
                /* Load 8 RGB pixels (24 bytes) and deinterleave to R,G,B */
                const uint8_t* p = row + x * 3;
                /* Manual deinterleave: extract R,G,B into separate int arrays */
                __m256i r_i32 = _mm256_setr_epi32(p[0], p[3], p[6], p[9], p[12], p[15], p[18], p[21]);
                __m256i g_i32 = _mm256_setr_epi32(p[1], p[4], p[7], p[10], p[13], p[16], p[19], p[22]);
                __m256i b_i32 = _mm256_setr_epi32(p[2], p[5], p[8], p[11], p[14], p[17], p[20], p[23]);
                __m256 rf = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i32), vinv);
                __m256 gf = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i32), vinv);
                __m256 bf = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i32), vinv);
                _mm256_storeu_ps(ch0 + base + x, rf);
                _mm256_storeu_ps(ch1 + base + x, gf);
                _mm256_storeu_ps(ch2 + base + x, bf);
            }
#endif
            for (; x < new_w; x++) {
                const uint8_t* p = row + x * 3;
                ch0[base + x] = p[0] * inv255;
                ch1[base + x] = p[1] * inv255;
                ch2[base + x] = p[2] * inv255;
            }
        }
    } else {
        /* General bilinear resize path — parallelized across rows */
        typedef struct { const uint8_t* rgb; float* ch0; float* ch1; float* ch2;
                         int src_w,src_h,S,new_w,pad_x,pad_y; float inv_scale,inv255; } PPCtx;
        /* Worker declared at file scope below */
        PPCtx pp = {rgb_hwc, ch0, ch1, ch2, src_w, src_h, S, new_w, pad_x, pad_y, inv_scale, inv255};
        nn2_tp_parallel_for(preprocess_worker, &pp, new_h, new_h > 60 ? new_h / 6 : 1);
    }
    return;
    /* Dead code — original loop kept for reference */
    {
        int new_h_unused = new_h;
        (void)new_h_unused;
        for (int y = 0; y < new_h; y++) {
            float sy = y * inv_scale;
            int iy = (int)sy;
            float fy = sy - iy;
            if (iy >= src_h - 1) { iy = src_h - 2; fy = 1.0f; }
            float fy1 = 1.0f - fy;
            int dy = y + pad_y;
            if (dy >= S) continue;

            const uint8_t* row0 = rgb_hwc + iy * src_w * 3;
            const uint8_t* row1 = row0 + src_w * 3;

            for (int x = 0; x < new_w; x++) {
                float sx = x * inv_scale;
                int ix = (int)sx;
                float fx = sx - ix;
                if (ix >= src_w - 1) { ix = src_w - 2; fx = 1.0f; }
                float fx1 = 1.0f - fx;
                int dx = x + pad_x;
                if (dx >= S) continue;

                int idx = dy * S + dx;
                const uint8_t* p00 = row0 + ix * 3;
                const uint8_t* p01 = p00 + 3;
                const uint8_t* p10 = row1 + ix * 3;
                const uint8_t* p11 = p10 + 3;

                float w00 = fx1 * fy1, w01 = fx * fy1;
                float w10 = fx1 * fy, w11 = fx * fy;

                ch0[idx] = (p00[0]*w00 + p01[0]*w01 + p10[0]*w10 + p11[0]*w11) * inv255;
                ch1[idx] = (p00[1]*w00 + p01[1]*w01 + p10[1]*w10 + p11[1]*w11) * inv255;
                ch2[idx] = (p00[2]*w00 + p01[2]*w01 + p10[2]*w10 + p11[2]*w11) * inv255;
            }
        }
    }
}

/* ============ Detection ============ */

int nn2_detect(NN2* ctx, const uint8_t* rgb_hwc, int width, int height,
               NN2Det* out, int max_det)
{
    int S = ctx->input_size;

    /* Preprocess: letterbox + normalize + HWC→CHW */
    float* input = ctx->workspace;
    preprocess(rgb_hwc, width, height, S, input, ctx->_fused_norm);

    /* Compute scale for mapping back to original coords */
    float scale = (float)S / (width > height ? width : height);
    int new_w = (int)(width * scale);
    int new_h = (int)(height * scale);
    float pad_x = (float)((S - new_w) / 2);
    float pad_y = (float)((S - new_h) / 2);

    /* Forward pass */
    int total_anchors = 0;
    for (int s = 0; s < 3; s++) {
        int gs = S / (8 << s);
        total_anchors += gs * gs;
    }

    /* Use end of workspace for bbox/cls feat (avoid malloc per frame) */
    size_t feat_offset = ctx->ws_size / sizeof(float)
                         - (4 * NN2_REG_MAX + ctx->num_classes) * total_anchors;
    float* bbox_feat = ctx->workspace + feat_offset;
    float* cls_feat  = bbox_feat + 4 * NN2_REG_MAX * total_anchors;

    /* Shift workspace past input to avoid overwriting */
    float* saved_ws = ctx->workspace;
    size_t saved_sz = ctx->ws_size;
    ctx->workspace = saved_ws + 3 * S * S;
    ctx->ws_size = saved_sz - 3 * S * S * sizeof(float);

    nn2_forward(ctx, input, bbox_feat, cls_feat);

    ctx->workspace = saved_ws;
    ctx->ws_size = saved_sz;

    /* Decode + NMS (use stack for raw dets — max ~2100 for 320, ~8400 for 640) */
    NN2Det raw[NN2_MAX_DET * 2]; /* generous stack buffer */
    int max_raw = sizeof(raw) / sizeof(raw[0]);
    int n = nn2_decode(bbox_feat, cls_feat, ctx->num_classes, S,
                       ctx->conf_thresh, ctx->dfl_weight, raw, max_raw);
    n = nn2_nms(raw, n, ctx->nms_thresh);

    /* Map coords back to original image */
    int n_out = n < max_det ? n : max_det;
    for (int i = 0; i < n_out; i++) {
        out[i] = raw[i];
        out[i].x1 = (out[i].x1 - pad_x) / scale;
        out[i].y1 = (out[i].y1 - pad_y) / scale;
        out[i].x2 = (out[i].x2 - pad_x) / scale;
        out[i].y2 = (out[i].y2 - pad_y) / scale;
    }

    return n_out;
}

void nn2_set_conf_threshold(NN2* ctx, float threshold) { ctx->conf_thresh = threshold; }
void nn2_set_nms_threshold(NN2* ctx, float threshold)  { ctx->nms_thresh = threshold; }
void nn2_set_input_size(NN2* ctx, int size)             { ctx->input_size = size; }

/* ============ Cleanup ============ */

static void free_conv(ConvLayer* L) { free(L->w); _aligned_free(L->w_packed); free(L->b); free(L->w_wino); }
static void free_c2f(C2fBlock* blk) {
    free_conv(&blk->cv1);
    free_conv(&blk->cv2);
    for (int i = 0; i < blk->n; i++) {
        free_conv(&blk->m[i].cv1);
        free_conv(&blk->m[i].cv2);
    }
}

void nn2_free(NN2* ctx)
{
    if (!ctx) return;
    for (int i = 0; i < 5; i++) free_conv(&ctx->bb_conv[i]);
    for (int i = 0; i < 4; i++) free_c2f(&ctx->bb_c2f[i]);
    free_conv(&ctx->sppf_cv1);
    free_conv(&ctx->sppf_cv2);
    for (int i = 0; i < 4; i++) free_c2f(&ctx->nk_c2f[i]);
    for (int i = 0; i < 2; i++) free_conv(&ctx->nk_conv[i]);
    for (int s = 0; s < 3; s++) {
        for (int j = 0; j < 3; j++) free_conv(&ctx->head[s].bbox[j]);
        for (int j = 0; j < 3; j++) free_conv(&ctx->head[s].cls[j]);
    }
    free(ctx->workspace);
    nn2_tp_destroy();
    free(ctx);
}
