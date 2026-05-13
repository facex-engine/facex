/*
 * FaceX Detect — YuNet face detector, pure C, FP32.
 *
 * Architecture: depthwise-separable backbone + FPN + multi-scale heads.
 * 53K params, ~208KB FP32 weights. Outputs bbox + 5 keypoints.
 */

#include "detect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#ifdef __wasm_simd128__
#include "../include/wasm_compat.h"
#endif
#endif

#define DETECT_VERSION_STR "0.3.0-fp32"

#define DETECT_DEFAULT_SCORE_THRESHOLD 0.5f
#define DETECT_DEFAULT_NMS_THRESHOLD   0.4f
#define DETECT_MAX_TENSORS 128

struct Detect {
    float score_threshold;
    float nms_threshold;
    uint8_t* raw;
    size_t   raw_len;
    float*  tensors[DETECT_MAX_TENSORS];
    int     n_tensors;
    float*  work;
    size_t  work_size;
};

static int load_file(const char* path, uint8_t** out, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return -1; }
    rewind(f);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    fclose(f);
    *out = buf; *out_len = (size_t)sz;
    return 0;
}

Detect* detect_init(const char* weights_path) {
    if (!weights_path) return NULL;
    Detect* det = (Detect*)calloc(1, sizeof(Detect));
    if (!det) return NULL;
    det->score_threshold = DETECT_DEFAULT_SCORE_THRESHOLD;
    det->nms_threshold   = DETECT_DEFAULT_NMS_THRESHOLD;

    if (load_file(weights_path, &det->raw, &det->raw_len) != 0) { free(det); return NULL; }
    if (det->raw_len < 8 || memcmp(det->raw, "YNET", 4) != 0) {
        fprintf(stderr, "detect: bad magic\n"); free(det->raw); free(det); return NULL;
    }
    uint32_t n = *(uint32_t*)(det->raw + 4);
    if (n > DETECT_MAX_TENSORS) { free(det->raw); free(det); return NULL; }
    det->n_tensors = (int)n;

    size_t off = 8;
    for (int i = 0; i < det->n_tensors; i++) {
        if (off + 4 > det->raw_len) { free(det->raw); free(det); return NULL; }
        uint32_t sz = *(uint32_t*)(det->raw + off); off += 4;
        if (off + sz > det->raw_len) { free(det->raw); free(det); return NULL; }
        det->tensors[i] = (float*)(det->raw + off); off += sz;
    }

    det->work_size = 4 * 1024 * 1024;
    det->work = (float*)calloc(det->work_size, sizeof(float));
    if (!det->work) { free(det->raw); free(det); return NULL; }
    return det;
}

/* ============ Conv helpers ============ */
static void conv2d(const float* in, int Ci, int Hi, int Wi,
                   const float* w, const float* b, int Co, int K, int s, int pad, float* out) {
    int Ho = (Hi + 2*pad - K) / s + 1, Wo = (Wi + 2*pad - K) / s + 1;
    for (int co = 0; co < Co; co++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                float sum = b ? b[co] : 0;
                for (int ci = 0; ci < Ci; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
                            int iy = oy*s - pad + ky, ix = ox*s - pad + kx;
                            if (iy >= 0 && iy < Hi && ix >= 0 && ix < Wi)
                                sum += in[(size_t)ci*Hi*Wi + iy*Wi + ix] *
                                       w[((size_t)co*Ci + ci)*K*K + ky*K + kx];
                        }
                out[(size_t)co*Ho*Wo + oy*Wo + ox] = sum;
            }
}

static void conv_dw(const float* in, int C, int Hi, int Wi,
                    const float* w, const float* b, int K, int s, int pad, float* out) {
    int Ho = (Hi + 2*pad - K) / s + 1, Wo = (Wi + 2*pad - K) / s + 1;
    /* Fast path: K=3, s=1, pad=1 — most common in YuNet */
    if (K == 3 && s == 1 && pad == 1) {
        for (int c = 0; c < C; c++) {
            const float* inp_c = in + (size_t)c*Hi*Wi;
            const float* wc = w + (size_t)c*9;
            float bc = b ? b[c] : 0;
            float w00=wc[0],w01=wc[1],w02=wc[2];
            float w10=wc[3],w11=wc[4],w12=wc[5];
            float w20=wc[6],w21=wc[7],w22=wc[8];
            for (int oy = 0; oy < Ho; oy++) {
                for (int ox = 0; ox < Wo; ox++) {
                    float sum = bc;
                    int iy = oy - 1, ix = ox - 1;
                    /* Unrolled 3×3 with boundary checks */
                    if(iy>=0) {
                        if(ix>=0) sum += inp_c[iy*Wi+ix]*w00;
                        sum += inp_c[iy*Wi+ox]*w01;
                        if(ox+1<Wi) sum += inp_c[iy*Wi+ox+1]*w02;
                    }
                    if(ix>=0) sum += inp_c[oy*Wi+ix]*w10;
                    sum += inp_c[oy*Wi+ox]*w11;
                    if(ox+1<Wi) sum += inp_c[oy*Wi+ox+1]*w12;
                    if(oy+1<Hi) {
                        if(ix>=0) sum += inp_c[(oy+1)*Wi+ix]*w20;
                        sum += inp_c[(oy+1)*Wi+ox]*w21;
                        if(ox+1<Wi) sum += inp_c[(oy+1)*Wi+ox+1]*w22;
                    }
                    out[(size_t)c*Ho*Wo + oy*Wo + ox] = sum;
                }
            }
        }
        return;
    }
    /* Generic path */
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                float sum = b ? b[c] : 0;
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++) {
                        int iy = oy*s - pad + ky, ix = ox*s - pad + kx;
                        if (iy >= 0 && iy < Hi && ix >= 0 && ix < Wi)
                            sum += in[(size_t)c*Hi*Wi + iy*Wi + ix] * w[(size_t)c*K*K + ky*K + kx];
                    }
                out[(size_t)c*Ho*Wo + oy*Wo + ox] = sum;
            }
}

static void conv1x1(const float* in, int Ci, int HW, const float* w, const float* b, int Co, float* out) {
    /* Initialize output with bias */
    for (int co = 0; co < Co; co++) {
        float bias = b ? b[co] : 0;
        float* orow = out + (size_t)co * HW;
        for (int i = 0; i < HW; i++) orow[i] = bias;
    }
    /* Accumulate: for each input channel, broadcast weight and FMA across all pixels.
     * This is cache-friendly: sequential reads on both in[] and out[]. */
    for (int ci = 0; ci < Ci; ci++) {
        const float* in_ch = in + (size_t)ci * HW;
        for (int co = 0; co < Co; co++) {
            float wt = w[(size_t)co * Ci + ci];
            float* orow = out + (size_t)co * HW;
            int i = 0;
#ifdef __AVX2__
            __m256 vw = _mm256_set1_ps(wt);
            for (; i + 8 <= HW; i += 8) {
                __m256 vi = _mm256_loadu_ps(in_ch + i);
                __m256 vo = _mm256_loadu_ps(orow + i);
                _mm256_storeu_ps(orow + i, _mm256_fmadd_ps(vw, vi, vo));
            }
#endif
            for (; i < HW; i++) orow[i] += wt * in_ch[i];
        }
    }
}

static void relu_(float* x, int n) { for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0; }

static void maxpool2x2(const float* in, int C, int H, int W, float* out) {
    int Ho = H/2, Wo = W/2;
    for (int c = 0; c < C; c++) {
        const float* ic = in + (size_t)c*H*W;
        float* oc = out + (size_t)c*Ho*Wo;
        for (int oy = 0; oy < Ho; oy++) {
            const float* r0 = ic + oy*2*W;
            const float* r1 = r0 + W;
            for (int ox = 0; ox < Wo; ox++) {
                float m = r0[ox*2]; float v;
                v=r0[ox*2+1]; if(v>m)m=v;
                v=r1[ox*2];   if(v>m)m=v;
                v=r1[ox*2+1]; if(v>m)m=v;
                oc[oy*Wo+ox] = m;
            }
        }
    }
}

static void upsample2x(const float* in, int C, int H, int W, float* out) {
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H*2; y++)
            for (int x = 0; x < W*2; x++)
                out[(size_t)c*4*H*W + y*W*2 + x] = in[(size_t)c*H*W + (y/2)*W + x/2];
}

static void add_(float* a, const float* b, int n) { for (int i = 0; i < n; i++) a[i] += b[i]; }
static float sigm(float x) { return 1.0f/(1.0f+expf(-x)); }

static int nms_sort(DetectFace* f, int n, float iou_t) {
    for (int i = 1; i < n; i++) { DetectFace t = f[i]; int j=i; while(j>0&&f[j-1].score<t.score){f[j]=f[j-1];j--;} f[j]=t; }
    int k = 0;
    for (int i = 0; i < n; i++) {
        int drop = 0;
        for (int j = 0; j < k; j++) {
            float ix1=fmaxf(f[i].x1,f[j].x1), iy1=fmaxf(f[i].y1,f[j].y1);
            float ix2=fminf(f[i].x2,f[j].x2), iy2=fminf(f[i].y2,f[j].y2);
            float inter=fmaxf(0,ix2-ix1)*fmaxf(0,iy2-iy1);
            float u=(f[i].x2-f[i].x1)*(f[i].y2-f[i].y1)+(f[j].x2-f[j].x1)*(f[j].y2-f[j].y1)-inter;
            if(u>0&&inter/u>iou_t){drop=1;break;}
        }
        if(!drop){ if(k!=i)f[k]=f[i]; k++; }
    }
    return k;
}

/* ============ YuNet Forward ============ */
#define T(i) (det->tensors[i])

int detect_run(Detect* det, const uint8_t* rgb_hwc, int width, int height,
               DetectFace* out, int max_faces) {
    if (!det || !rgb_hwc || !out || max_faces < 0) return -1;
    if (width <= 0 || height <= 0) return -1;

    int H = height, W = width;
    int h2=H/2, w2=W/2, h4=h2/2, w4=w2/2, h8=h4/2, w8=w4/2, h16=h8/2, w16=w8/2, h32=h16/2, w32=w16/2;

    /* Workspace layout: dedicated buffers */
    float* wp = det->work;
    float* inp  = wp; wp += 3*H*W;
    float* a    = wp; wp += 64*h2*w2;
    float* b    = wp; wp += 64*h2*w2;
    float* c    = wp; wp += 64*h2*w2;
    float* f8   = wp; wp += 64*h8*w8;    /* saved stride-8 */
    float* f16  = wp; wp += 64*h16*w16;  /* saved stride-16 */
    float* f32_ = wp; wp += 64*h32*w32;  /* saved stride-32 */
    float* hd   = wp; wp += 16*h8*w8;    /* head scratch */

    /* uint8 HWC → float32 CHW [0..255] */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int ch = 0; ch < 3; ch++)
                inp[(size_t)ch*H*W + y*W + x] = (float)rgb_hwc[(y*W+x)*3 + ch];

    /* Stem: Conv3×3 s2 (3→16) + ReLU + PW(16→16) */
    conv2d(inp, 3, H, W, T(0), T(1), 16, 3, 2, 1, a);
    relu_(a, 16*h2*w2);
    conv1x1(a, 16, h2*w2, T(33), T(32), 16, b);

    /* model0: DW3×3 + ReLU + MaxPool */
    conv_dw(b, 16, h2, w2, T(2), T(3), 3, 1, 1, a);
    relu_(a, 16*h2*w2);
    maxpool2x2(a, 16, h2, w2, b); /* → [16, h4, w4] */

    /* model1: PW(16→16) + DW+ReLU + PW(16→32) */
    conv1x1(b, 16, h4*w4, T(35), T(34), 16, a);
    conv_dw(a, 16, h4, w4, T(4), T(5), 3, 1, 1, b);
    relu_(b, 16*h4*w4);
    conv1x1(b, 16, h4*w4, T(37), T(36), 32, a);

    /* model2 part1: DW(32)+ReLU + PW(32→32) */
    conv_dw(a, 32, h4, w4, T(6), T(7), 3, 1, 1, b);
    relu_(b, 32*h4*w4);
    conv1x1(b, 32, h4*w4, T(39), T(38), 32, a);

    /* model2 part2: DW(32)+ReLU + PW(32→64) */
    conv_dw(a, 32, h4, w4, T(8), T(9), 3, 1, 1, b);
    relu_(b, 32*h4*w4);
    conv1x1(b, 32, h4*w4, T(41), T(40), 64, a);

    /* DW(64)+ReLU + MaxPool → stride 8 */
    conv_dw(a, 64, h4, w4, T(10), T(11), 3, 1, 1, b);
    relu_(b, 64*h4*w4);
    maxpool2x2(b, 64, h4, w4, a); /* → [64, h8, w8] */

    /* model3: PW+DW+ReLU+PW+DW+ReLU */
    conv1x1(a, 64, h8*w8, T(43), T(42), 64, b);
    conv_dw(b, 64, h8, w8, T(12), T(13), 3, 1, 1, a);
    relu_(a, 64*h8*w8);
    conv1x1(a, 64, h8*w8, T(45), T(44), 64, b);
    conv_dw(b, 64, h8, w8, T(14), T(15), 3, 1, 1, a);
    relu_(a, 64*h8*w8);
    memcpy(f8, a, 64*h8*w8*sizeof(float));

    /* MaxPool → stride 16 */
    maxpool2x2(a, 64, h8, w8, b);

    /* model4: PW+DW+ReLU+PW+DW+ReLU */
    conv1x1(b, 64, h16*w16, T(47), T(46), 64, a);
    conv_dw(a, 64, h16, w16, T(16), T(17), 3, 1, 1, b);
    relu_(b, 64*h16*w16);
    conv1x1(b, 64, h16*w16, T(49), T(48), 64, a);
    conv_dw(a, 64, h16, w16, T(18), T(19), 3, 1, 1, b);
    relu_(b, 64*h16*w16);
    memcpy(f16, b, 64*h16*w16*sizeof(float));

    /* MaxPool → stride 32 */
    maxpool2x2(b, 64, h16, w16, a);

    /* model5: PW+DW+ReLU+PW+DW+ReLU */
    conv1x1(a, 64, h32*w32, T(51), T(50), 64, b);
    conv_dw(b, 64, h32, w32, T(20), T(21), 3, 1, 1, a);
    relu_(a, 64*h32*w32);
    conv1x1(a, 64, h32*w32, T(53), T(52), 64, b);
    conv_dw(b, 64, h32, w32, T(22), T(23), 3, 1, 1, a);
    relu_(a, 64*h32*w32);

    /* === FPN === */
    /* Lateral s32 + DW+ReLU */
    conv1x1(a, 64, h32*w32, T(107), T(106), 64, b);
    conv_dw(b, 64, h32, w32, T(24), T(25), 3, 1, 1, a);
    relu_(a, 64*h32*w32);
    memcpy(f32_, a, 64*h32*w32*sizeof(float));

    /* Upsample s32→s16, add f16 */
    upsample2x(a, 64, h32, w32, b);
    add_(b, f16, 64*h16*w16);
    conv1x1(b, 64, h16*w16, T(105), T(104), 64, a);
    conv_dw(a, 64, h16, w16, T(26), T(27), 3, 1, 1, b);
    relu_(b, 64*h16*w16);
    memcpy(f16, b, 64*h16*w16*sizeof(float));

    /* Upsample s16→s8, add f8 */
    upsample2x(b, 64, h16, w16, a);
    add_(a, f8, 64*h8*w8);
    conv1x1(a, 64, h8*w8, T(103), T(102), 64, b);
    conv_dw(b, 64, h8, w8, T(28), T(29), 3, 1, 1, a);
    relu_(a, 64*h8*w8);
    memcpy(f8, a, 64*h8*w8*sizeof(float));

    /* === HEADS + DECODE === */
    int nf = 0;
    struct { float* feat; int fh, fw, stride; } levels[3] = {
        {f8, h8, w8, 8}, {f16, h16, w16, 16}, {f32_, h32, w32, 32}
    };
    /* Head weight indices: [stride_idx] → tensor indices
     * cls:  PW w/b = {67,66}, {71,70}, {75,74}  DW w/b = {69,68}, {73,72}, {77,76}
     * obj:  PW = {91,90},{95,94},{99,98}  DW = {93,92},{97,96},{101,100}
     * bbox: PW = {55,54},{59,58},{63,62}  DW = {57,56},{61,60},{65,64}
     * kps:  PW = {79,78},{83,82},{87,86}  DW = {81,80},{85,84},{89,88}
     */
    int cw1[]={67,71,75}, cb1[]={66,70,74}, cw2[]={69,73,77}, cb2[]={68,72,76};
    int ow1[]={91,95,99}, ob1[]={90,94,98}, ow2[]={93,97,101}, ob2[]={92,96,100};
    int bw1[]={55,59,63}, bb1[]={54,58,62}, bw2[]={57,61,65}, bb2[]={56,60,64};
    int kw1[]={79,83,87}, kb1[]={78,82,86}, kw2[]={81,85,89}, kb2[]={80,84,88};

    for (int si = 0; si < 3; si++) {
        int fh = levels[si].fh, fw = levels[si].fw, st = levels[si].stride;
        float* feat = levels[si].feat;
        if (fh <= 0 || fw <= 0) continue;
        int hw = fh * fw;

        /* cls: PW(64→1) + DW3×3(1) */
        conv1x1(feat, 64, hw, T(cw1[si]), T(cb1[si]), 1, a);
        conv_dw(a, 1, fh, fw, T(cw2[si]), T(cb2[si]), 3, 1, 1, b);
        float* cls = b;

        /* obj: PW(64→1) + DW3×3(1) */
        conv1x1(feat, 64, hw, T(ow1[si]), T(ob1[si]), 1, a);
        conv_dw(a, 1, fh, fw, T(ow2[si]), T(ob2[si]), 3, 1, 1, c);
        float* obj = c;

        /* bbox: PW(64→4) + DW3×3(4) */
        conv1x1(feat, 64, hw, T(bw1[si]), T(bb1[si]), 4, a);
        conv_dw(a, 4, fh, fw, T(bw2[si]), T(bb2[si]), 3, 1, 1, hd);

        /* kps: PW(64→10) + DW3×3(10) */
        conv1x1(feat, 64, hw, T(kw1[si]), T(kb1[si]), 10, a);
        conv_dw(a, 10, fh, fw, T(kw2[si]), T(kb2[si]), 3, 1, 1, hd + 4*hw);

        for (int gy = 0; gy < fh && nf < max_faces; gy++)
            for (int gx = 0; gx < fw && nf < max_faces; gx++) {
                int idx = gy * fw + gx;
                float sc = sigm(cls[idx]) * sigm(obj[idx]);
                if (sc < det->score_threshold) continue;
                float cx = (gx + 0.5f) * st, cy = (gy + 0.5f) * st;
                out[nf].x1 = fmaxf(0, cx - hd[0*hw+idx] * st);
                out[nf].y1 = fmaxf(0, cy - hd[1*hw+idx] * st);
                out[nf].x2 = fminf((float)width, cx + hd[2*hw+idx] * st);
                out[nf].y2 = fminf((float)height, cy + hd[3*hw+idx] * st);
                /* Filter degenerate boxes */
                float bw = out[nf].x2 - out[nf].x1;
                float bh = out[nf].y2 - out[nf].y1;
                if (bw < 16 || bh < 16) continue;         /* too small */
                if (bw > width*0.95f || bh > height*0.95f) continue; /* too large */
                float aspect = bw / (bh + 1e-6f);
                if (aspect < 0.3f || aspect > 3.0f) continue; /* bad aspect ratio */
                out[nf].score = sc;
                for (int k = 0; k < 5; k++) {
                    out[nf].kps[k*2]   = fmaxf(0, fminf((float)width,  cx + hd[(4+k*2)*hw+idx] * st));
                    out[nf].kps[k*2+1] = fmaxf(0, fminf((float)height, cy + hd[(4+k*2+1)*hw+idx] * st));
                }
                nf++;
            }
    }
    if (nf > 1) nf = nms_sort(out, nf, det->nms_threshold);

    /* Cross-scale merge: if two faces have nose keypoints within 20px,
     * keep only the one with better score. This handles multi-scale
     * detections of the same face at different strides. */
    if (nf > 1) {
        for (int i = 0; i < nf; i++) {
            if (out[i].score < 0) continue; /* already merged */
            for (int j = i + 1; j < nf; j++) {
                if (out[j].score < 0) continue;
                /* Compare nose keypoints (index 4,5) */
                float dx = out[i].kps[4] - out[j].kps[4];
                float dy = out[i].kps[5] - out[j].kps[5];
                float dist = sqrtf(dx*dx + dy*dy);
                /* Also compare eye midpoints */
                float ei_x = (out[i].kps[0] + out[i].kps[2]) * 0.5f;
                float ei_y = (out[i].kps[1] + out[i].kps[3]) * 0.5f;
                float ej_x = (out[j].kps[0] + out[j].kps[2]) * 0.5f;
                float ej_y = (out[j].kps[1] + out[j].kps[3]) * 0.5f;
                float eye_dist = sqrtf((ei_x-ej_x)*(ei_x-ej_x) + (ei_y-ej_y)*(ei_y-ej_y));
                if (dist < 25.0f || eye_dist < 20.0f) {
                    out[j].score = -1; /* suppress */
                }
            }
        }
        /* Compact */
        int k = 0;
        for (int i = 0; i < nf; i++)
            if (out[i].score > 0) { if (k != i) out[k] = out[i]; k++; }
        nf = k;
    }

    return nf;
}

void detect_free(Detect* det) {
    if (!det) return;
    free(det->work); free(det->raw); free(det);
}

const char* detect_version(void) { return DETECT_VERSION_STR; }

void detect_set_score_threshold(Detect* det, float t) {
    if(det) det->score_threshold = t < 0 ? 0 : t > 1 ? 1 : t;
}
void detect_set_nms_threshold(Detect* det, float t) {
    if(det) det->nms_threshold = t < 0 ? 0 : t > 1 ? 1 : t;
}
