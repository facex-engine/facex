/*
 * nanodet.c — NanoDet-Plus-m complete forward pass.
 *
 * Architecture (110 convolutions):
 *   Backbone: ShuffleNetV2 (stem + 3 stages)
 *   Neck: Ghost-PAN (3 reduce + 3 ghost blocks + 2 downsample)
 *   Head: GFL (4 scales × DW-PW-DW-PW-PW + split + sigmoid)
 *
 * ShuffleNetV2 block:
 *   Stride: branch1(DW→PW) + branch2(PW→DW→PW) → concat
 *   Regular: split → PW→DW→PW + identity → concat → shuffle
 *
 * Ghost module (in PAN):
 *   primary = PW(in→out/2)
 *   ghost = DW(primary)
 *   concat(primary, ghost) → output
 *
 * Ghost block (in PAN):
 *   ghost1 → ghost2 → add(input_PW, ghost2)
 */

#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ============ Conv layer with weights ============ */

typedef struct {
    float* w;
    float* b;
    int cout, cin_per_g, k, stride, pad, groups, act;
} NDConv;

/* Run conv: dispatches to DW, PW, or regular */
static void nd_run(const NDConv* L, const float* in, int H, int W,
                   float* out, float* col)
{
    if (L->groups > 1) {
        nn2_dwconv2d(in, L->w, L->b, L->cout, H, W,
                     L->k, L->stride, L->pad, L->act, out);
    } else {
        int Hout = (H + 2*L->pad - L->k) / L->stride + 1;
        int Wout = (W + 2*L->pad - L->k) / L->stride + 1;
        int sp = Hout * Wout;
        int cin = L->cin_per_g;

        if (L->k == 1) {
            nn2_sgemm_bias_act(L->cout, cin, sp, L->w, cin, in, sp,
                               out, sp, L->b, L->act);
        } else {
            int K = cin * L->k * L->k;
            nn2_im2col(in, cin, H, W, L->k, L->k, L->stride, L->pad, col);
            nn2_sgemm_bias_act(L->cout, K, sp, L->w, K, col, sp,
                               out, sp, L->b, L->act);
        }
    }
}

/* ============ NanoDet model ============ */

typedef struct NanoDet {
    NDConv cv[110];   /* all 110 conv layers */
    int num_classes;
    int input_size;
    float* ws;        /* workspace */
    size_t ws_size;
} NanoDet;

/* ============ Load ============ */

NanoDet* nanodet_init(const char* path, int n_threads)
{
    FILE* f = fopen(path, "rb");

    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "NN2N", 4)) { fclose(f); return NULL; }

    NanoDet* nd = (NanoDet*)calloc(1, sizeof(NanoDet));
    uint32_t nc, isz, nl;
    fread(&nc, 4, 1, f); fread(&isz, 4, 1, f); fread(&nl, 4, 1, f);
    nd->num_classes = nc; nd->input_size = isz;

    for (int i = 0; i < (int)nl && i < 110; i++) {
        NDConv* L = &nd->cv[i];
        uint32_t v[7]; fread(v, 4, 7, f);
        L->cout = v[0]; L->cin_per_g = v[1]; L->k = v[2];
        L->stride = v[3]; L->pad = v[4]; L->groups = v[5]; L->act = v[6];
        L->b = (float*)malloc(L->cout * 4);
        fread(L->b, 4, L->cout, f);
        int wsz = L->cout * L->cin_per_g * L->k * L->k;
        L->w = (float*)malloc(wsz * 4);
        fread(L->w, 4, wsz, f);
    }
    fclose(f);

    nd->ws_size = 256 << 20; /* 256MB */
    nd->ws = (float*)malloc(nd->ws_size);

    nn2_tp_init(n_threads);
    return nd;
}

/* ============ Helpers ============ */

/* Channel split: first half / second half */
static void split_channels(const float* in, int C, int sp,
                           const float** left, const float** right)
{
    *left = in;
    *right = in + (C / 2) * sp;
}

/* Channel shuffle (groups=2) */
static void shuffle2(const float* in, int C, int sp, float* out)
{
    int half = C / 2;
    for (int c = 0; c < half; c++) {
        memcpy(out + (2*c)     * sp, in + c * sp,        sp * 4);
        memcpy(out + (2*c + 1) * sp, in + (half+c) * sp, sp * 4);
    }
}

/* Nearest-neighbor upsample 2x */
static void upsample2x(const float* in, int C, int H, int W, float* out)
{
    nn2_upsample_2x(in, C, H, W, out);
}

/* ============ ShuffleNetV2 blocks ============ */

/* Stride block: no split, two branches, concat.
 * branch1: DW(s=2) → PW
 * branch2: PW → DW(s=2) → PW
 * cvs[0]=DW, cvs[1]=PW, cvs[2]=PW, cvs[3]=DW, cvs[4]=PW */
static void shufflev2_stride(const NDConv cvs[5], const float* in,
                              int C_in, int H, int W,
                              float* out, float* tmp, float* col)
{
    int Ho = H / 2, Wo = W / 2;
    int c_out = cvs[1].cout; /* output channels per branch */


    /* Branch 1: DW(s=2) → PW */
    float* br1_dw = tmp;
    nd_run(&cvs[0], in, H, W, br1_dw, col);
    float* br1 = br1_dw + C_in * Ho * Wo;
    nd_run(&cvs[1], br1_dw, Ho, Wo, br1, col);
    float* br2_pw1 = br1 + c_out * Ho * Wo;
    nd_run(&cvs[2], in, H, W, br2_pw1, col);
    float* br2_dw = br2_pw1 + cvs[2].cout * H * W;
    nd_run(&cvs[3], br2_pw1, H, W, br2_dw, col);
    float* br2 = br2_dw + cvs[3].cout * Ho * Wo;
    nd_run(&cvs[4], br2_dw, Ho, Wo, br2, col);

    /* Concat: [br1, br2] */
    int sp = Ho * Wo;
    memcpy(out, br1, c_out * sp * 4);
    memcpy(out + c_out * sp, br2, c_out * sp * 4);
}

/* Regular block: split → PW → DW → PW + identity → concat → shuffle.
 * cvs[0]=PW, cvs[1]=DW, cvs[2]=PW */
static void shufflev2_regular(const NDConv cvs[3], const float* in,
                               int C, int H, int W,
                               float* out, float* tmp, float* col)
{
    int sp = H * W;
    int half = C / 2;
    const float* left;
    const float* right;
    split_channels(in, C, sp, &left, &right);

    /* Process right branch: PW → DW → PW */
    float* pw1 = tmp;
    nd_run(&cvs[0], right, H, W, pw1, col);
    float* dw = pw1 + cvs[0].cout * sp;
    nd_run(&cvs[1], pw1, H, W, dw, col);
    float* pw2 = dw + cvs[1].cout * sp;
    nd_run(&cvs[2], dw, H, W, pw2, col);

    /* Fused concat+shuffle: write directly to interleaved output.
     * Shuffle2 of [left, pw2] → out[2c]=left[c], out[2c+1]=pw2[c].
     * Use SIMD memcpy for better bandwidth. */
    int bytes = sp * 4;
    for (int ci = 0; ci < half; ci++) {
        memcpy(out + (2*ci)     * sp, left + ci * sp, bytes);
        memcpy(out + (2*ci + 1) * sp, pw2  + ci * sp, bytes);
    }
}

/* ============ Ghost module (in PAN neck) ============ */

/* Ghost module: PW(in→out/2) → DW → concat → out
 * cvs[0]=PW(in→out/2), cvs[1]=DW(out/2) */
static void ghost_module(const NDConv cvs[2], const float* in,
                         int H, int W, float* out, float* tmp, float* col)
{
    int sp = H * W;
    /* Primary: PW */
    float* primary = tmp;
    nd_run(&cvs[0], in, H, W, primary, col);
    int c_half = cvs[0].cout;
    /* Ghost: DW on primary */
    float* ghost = primary + c_half * sp;
    nd_run(&cvs[1], primary, H, W, ghost, col);
    /* Concat [primary, ghost] → out */
    memcpy(out, primary, c_half * sp * 4);
    memcpy(out + c_half * sp, ghost, c_half * sp * 4);
}

/* Ghost block: ghost1 → ghost2 → add(shortcut, ghost2)
 * shortcut = DW5×5(input) → PW
 * cvs: [ghost1_pw, ghost1_dw, ghost1_pw2, ghost1_dw2,
 *        shortcut_dw, shortcut_pw] (6 convs) */

/* Ghost module — uses tmp for intermediates (no malloc) */
static void nd_ghost_module(const NDConv cvs[2], const float* in,
                            int H, int W, float* out, float* tmp, float* col)
{
    int sp = H * W;
    int c_half = cvs[0].cout;
    float* primary = tmp;
    nd_run(&cvs[0], in, H, W, primary, col);
    float* ghost_out = tmp + c_half * sp;
    nd_run(&cvs[1], primary, H, W, ghost_out, col);
    memcpy(out, primary, c_half * sp * sizeof(float));
    memcpy(out + c_half * sp, ghost_out, c_half * sp * sizeof(float));
}

/* Ghost block: ghost1→ghost2→shortcut(DW+PW)→add.
 * Uses tmp for all intermediates — needs ~(ch_in + 96*4) * sp floats. */
static void nd_ghost_block(const NDConv* cvs, int ch_in,
                           const float* in, int H, int W,
                           float* out, float* tmp, float* col)
{
    int sp = H * W;
    /* Layout in tmp: [g1: 96*sp][g2: 96*sp][sc_dw: ch_in*sp][sc: 96*sp][ghost_scratch: 96*sp] */
    float* g1 = tmp;
    float* ghost_tmp = tmp + (96 + 96 + ch_in + 96) * sp; /* scratch for ghost module internals */
    nd_ghost_module(&cvs[0], in, H, W, g1, ghost_tmp, col);
    float* g2 = g1 + 96 * sp;
    nd_ghost_module(&cvs[2], g1, H, W, g2, ghost_tmp, col);
    float* sc_dw = g2 + 96 * sp;
    nd_run(&cvs[4], in, H, W, sc_dw, col);
    float* sc = sc_dw + ch_in * sp;
    nd_run(&cvs[5], sc_dw, H, W, sc, col);
    for (int i = 0; i < 96 * sp; i++) out[i] = g2[i] + sc[i];
}

/* ============ FORWARD PASS ============ */

int nanodet_forward(NanoDet* nd, const float* input,
                    float* det_output, int* n_anchors)
{
    NDConv* c = nd->cv; /* shorthand */
    int S = nd->input_size;
    float* ws = nd->ws;
    /* Workspace layout: [col 4M][A 1M][B 1M][tmp 50M] = 56M floats in 256MB ws */
    float* col = ws;
    float* A = ws + 4*1024*1024;
    float* B = A + 1024*1024;
    float* tmp = B + 1024*1024;
    #define SWAP() do { float* _t = A; A = B; B = _t; } while(0)

    int h = S, w = S;

    /* === BACKBONE === */

    /* Stem: cv[0] Conv(3→24, k=3, s=2) + MaxPool(k=3,s=2,p=1) */
    nd_run(&c[0], input, h, w, A, col);
    h /= 2; w /= 2; /* 160×160 */
    /* MaxPool 3×3 s=2 p=1 */
    nn2_maxpool2d(A, c[0].cout, h, w, 3, 2, 1, B);
    SWAP();
    h /= 2; w /= 2; /* 80×80 */
    int stem_ch = 24;

    /* Stage 1: stride block (cvs 1-5) → 116ch @ 80×80 */
    shufflev2_stride(&c[1], A, stem_ch, h, w, B, tmp, col);
    SWAP();
    int s1_ch = c[1+1].cout * 2; /* 58*2 = 116 */

    /* Stage 1: 3 regular blocks (cvs 6-8, 9-11, 12-14) */
    for (int i = 0; i < 3; i++) {
        shufflev2_regular(&c[6 + i*3], A, s1_ch, h, w, B, tmp, col);
        SWAP();
    }
    float* feat_s1 = (float*)malloc(s1_ch * h * w * 4); /* P3: 116ch @ 80×80 */
    memcpy(feat_s1, A, s1_ch * h * w * 4);

    /* Stage 2: stride block (cvs 15-19) → 232ch @ 40×40 */
    /* stage2 stride */
    shufflev2_stride(&c[15], A, s1_ch, h, w, B, tmp, col);
    SWAP();
    h /= 2; w /= 2; /* 40×40 */
    int s2_ch = c[15+1].cout * 2; /* 116*2 = 232 */

    /* Stage 2: 7 regular blocks (cvs 20-40) */
    for (int i = 0; i < 7; i++) {
        shufflev2_regular(&c[20 + i*3], A, s2_ch, h, w, B, tmp, col);
        SWAP();
    }
    float* feat_s2 = (float*)malloc(s2_ch * h * w * 4); /* P4: 232ch @ 40×40 */
    memcpy(feat_s2, A, s2_ch * h * w * 4);

    /* Stage 3: stride block (cvs 41-45) → 464ch @ 20×20 */
    shufflev2_stride(&c[41], A, s2_ch, h, w, B, tmp, col);
    SWAP();
    h /= 2; w /= 2; /* 20×20 */
    int s3_ch = c[41+1].cout * 2; /* 232*2 = 464 */

    /* Stage 3: 3 regular blocks (cvs 46-54) */
    for (int i = 0; i < 3; i++) {
        shufflev2_regular(&c[46 + i*3], A, s3_ch, h, w, B, tmp, col);
        SWAP();
    }
    float* feat_s3 = (float*)malloc(s3_ch * h * w * 4);
    memcpy(feat_s3, A, s3_ch * h * w * 4);

    /* === NECK (Ghost-PAN) === */

    int h2=S/4, w2=S/4, h3=S/8, w3=S/8, h4=S/16, w4=S/16, h5=S/32, w5=S/32;
    int sp2=h2*w2, sp3=h3*w3, sp4=h4*w4, sp5=h5*w5;
    /* Bump allocator on tmp — no malloc in neck */
    float* bump = tmp;
    #define BUMP(n) (bump += (n), bump - (n))

    /* Reduce: permanent until end of neck */
    float*r1=BUMP(96*sp2); nd_run(&c[55],feat_s1,h2,w2,r1,col);
    float*r2=BUMP(96*sp3); nd_run(&c[56],feat_s2,h3,w3,r2,col);
    float*r3=BUMP(96*sp4); nd_run(&c[57],feat_s3,h4,w4,r3,col);
    free(feat_s1);free(feat_s2);free(feat_s3);

    /* Pan features: permanent until head */
    float*pan_p4=BUMP(96*sp3);
    float*pan_p3=BUMP(96*sp2);
    float*pan_n4=BUMP(96*sp3);
    float*pan_n5=BUMP(96*sp4);
    float*pan_n6=BUMP(96*sp5);
    /* ghost_block scratch: reused per block */
    float*gscratch=bump; /* ~(192+96*4)*sp2 worst case */

    /* Top-down */
    { float*up1=gscratch; upsample2x(r3,96,h4,w4,up1);
      float*cat1=up1+96*sp3; nn2_concat(up1,96,r2,96,h3,w3,cat1);
      nd_ghost_block(&c[58],192,cat1,h3,w3,pan_p4,gscratch+192*sp3+96*sp3,col); }

    { float*up2=gscratch; upsample2x(pan_p4,96,h3,w3,up2);
      float*cat2=up2+96*sp2; nn2_concat(up2,96,r1,96,h2,w2,cat2);
      nd_ghost_block(&c[64],192,cat2,h2,w2,pan_p3,gscratch+192*sp2+96*sp2,col); }

    /* Bottom-up */
    { float*ds_dw=gscratch; nd_run(&c[70],pan_p3,h2,w2,ds_dw,col);
      float*ds=ds_dw+96*sp3; nd_run(&c[71],ds_dw,h3,w3,ds,col);
      float*cat3=ds+96*sp3; nn2_concat(ds,96,pan_p4,96,h3,w3,cat3);
      nd_ghost_block(&c[72],192,cat3,h3,w3,pan_n4,cat3+192*sp3,col); }

    { float*ds_dw=gscratch; nd_run(&c[78],pan_n4,h3,w3,ds_dw,col);
      float*ds=ds_dw+96*sp4; nd_run(&c[79],ds_dw,h4,w4,ds,col);
      float*cat4=ds+96*sp4; nn2_concat(ds,96,r3,96,h4,w4,cat4);
      nd_ghost_block(&c[80],192,cat4,h4,w4,pan_n5,cat4+192*sp4,col); }

    /* Extra downsample for 4th scale */
    { float*ds3_dw=gscratch; nd_run(&c[86],r3,h4,w4,ds3_dw,col);
      float*ds3=ds3_dw+96*sp5; nd_run(&c[87],ds3_dw,h5,w5,ds3,col);
      float*ds4_dw=ds3+96*sp5; nd_run(&c[88],pan_n5,h4,w4,ds4_dw,col);
      float*ds4=ds4_dw+96*sp5; nd_run(&c[89],ds4_dw,h5,w5,ds4,col);
      for(int i=0;i<96*sp5;i++) pan_n6[i]=ds3[i]+ds4[i]; }
    #undef BUMP

    /* === HEAD === */
    float*feats[4]={pan_p3,pan_n4,pan_n5,pan_n6};
    int hh[4]={h2,h3,h4,h5}, hw[4]={w2,w3,w4,w5};
    int total_anchors=0;
    for(int s=0;s<4;s++) total_anchors+=hh[s]*hw[s];
    float*head_out=det_output;
    int aoff=0;
    for(int s=0;s<4;s++){
        int ci=90+s*5, fh=hh[s], fw=hw[s], fsp=fh*fw;
        /* Reuse gscratch for head intermediates — no malloc */
        float*x0=gscratch;
        nd_run(&c[ci],feats[s],fh,fw,x0,col);
        float*x1=x0+96*fsp;
        nd_run(&c[ci+1],x0,fh,fw,x1,col);
        float*x2=x1+96*fsp;
        nd_run(&c[ci+2],x1,fh,fw,x2,col);
        float*x3=x2+96*fsp;
        nd_run(&c[ci+3],x2,fh,fw,x3,col);
        float*x4=x3+96*fsp;
        nd_run(&c[ci+4],x3,fh,fw,x4,col);
        for(int a=0;a<fsp;a++)
            for(int ch=0;ch<112;ch++)
                head_out[(aoff+a)*112+ch]=x4[ch*fsp+a];
        aoff+=fsp;
    }

    /* A, B, tmp are workspace-based — no free needed */
    *n_anchors=total_anchors;
    return 0;
    #undef SWAP
}

/* ============ Detect ============ */

int nanodet_detect(NanoDet* nd, const uint8_t* rgb, int width, int height,
                   NN2Det* out, int max_det)
{
    int S = nd->input_size;

    /* Preprocess: letterbox + normalize */
    float scale = (float)S / (width > height ? width : height);
    int nw = (int)(width * scale), nh = (int)(height * scale);
    int px = (S - nw) / 2, py = (S - nh) / 2;

    /* NanoDet normalization: (pixel - mean) / std */
    /* mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395] */
    float mean[3] = {103.53f, 116.28f, 123.675f};
    float istd[3] = {1.0f/57.375f, 1.0f/57.12f, 1.0f/58.395f};

    float* input = nd->ws + 56*1024*1024; /* safe offset past forward workspace */
    /* Fill with normalized mean (gray) */
    for (int c = 0; c < 3; c++) {
        float fill = -mean[c] * istd[c];
        float* ch = input + c * S * S;
        for (int i = 0; i < S * S; i++) ch[i] = fill;
    }
    /* Bilinear resize + normalize */
    float inv_s = 1.0f / scale;
    for (int y = 0; y < nh; y++) {
        float sy = y * inv_s;
        int iy = (int)sy; float fy = sy - iy;
        if (iy >= height-1) { iy = height-2; fy = 1.0f; }
        for (int x = 0; x < nw; x++) {
            float sx = x * inv_s;
            int ix = (int)sx; float fx = sx - ix;
            if (ix >= width-1) { ix = width-2; fx = 1.0f; }
            const uint8_t* p00 = rgb + (iy*width+ix)*3;
            const uint8_t* p01 = p00+3, *p10 = p00+width*3, *p11 = p10+3;
            float w00=(1-fx)*(1-fy), w01=fx*(1-fy), w10=(1-fx)*fy, w11=fx*fy;
            int dy = y + py, dx = x + px;
            for (int c = 0; c < 3; c++) {
                float v = p00[c]*w00 + p01[c]*w01 + p10[c]*w10 + p11[c]*w11;
                input[c*S*S + dy*S + dx] = (v - mean[c]) * istd[c];
            }
        }
    }

    /* Forward — use end of workspace for detection buffer */
    float* det_buf = nd->ws + (nd->ws_size / sizeof(float)) - 8500 * 112;
    int n_anchors = 0;
    nanodet_forward(nd, input, det_buf, &n_anchors);

    /* Decode GFL output: [n_anchors, 112] → 112 = 32 (bbox DFL×4×8) + 80 (classes) */
    int nc = nd->num_classes; /* 80 */
    int reg_max = 8; /* NanoDet uses reg_max=7+1=8 (not 16 like YOLOv8) */
    int strides[4] = {8, 16, 32, 64};
    int grid_h[4], grid_w[4];
    for (int s = 0; s < 4; s++) {
        grid_h[s] = S / strides[s];
        grid_w[s] = S / strides[s];
    }

    int n_det = 0;
    int offset = 0;
    for (int si = 0; si < 4; si++) {
        int gh = grid_h[si], gw = grid_w[si];
        int stride = strides[si];
        for (int a = 0; a < gh * gw && n_det < max_det; a++) {
            float* row = det_buf + (offset + a) * 112;
            /* Classes: row[32..112], find best */
            float* cls = row + 4 * reg_max;
            /* Sigmoid + find max */
            float best = -1e9f; int best_c = 0;
            for (int ci = 0; ci < nc; ci++) {
                float v = 1.0f / (1.0f + expf(-cls[ci]));
                if (v > best) { best = v; best_c = ci; }
            }
            if (best < 0.25f) continue;

            /* DFL decode bbox: 4 distances × reg_max bins */
            float ltrb[4];
            for (int d = 0; d < 4; d++) {
                float* logits = row + d * reg_max;
                /* softmax → weighted sum */
                float maxv = logits[0];
                for (int r = 1; r < reg_max; r++)
                    if (logits[r] > maxv) maxv = logits[r];
                float sum = 0, val = 0;
                for (int r = 0; r < reg_max; r++) {
                    float e = expf(logits[r] - maxv);
                    sum += e; val += e * r;
                }
                ltrb[d] = val / sum;
            }

            int ay = a / gw, ax = a % gw;
            float cx = (ax + 0.5f) * stride;
            float cy = (ay + 0.5f) * stride;

            NN2Det* d = &out[n_det++];
            d->x1 = (cx - ltrb[0] * stride - px) / scale;
            d->y1 = (cy - ltrb[1] * stride - py) / scale;
            d->x2 = (cx + ltrb[2] * stride - px) / scale;
            d->y2 = (cy + ltrb[3] * stride - py) / scale;
            d->score = best;
            d->cls = best_c;
        }
        offset += gh * gw;
    }

    /* NMS */
    n_det = nn2_nms(out, n_det, 0.45f);
    return n_det;
}

void nanodet_free(NanoDet* nd)
{
    if (!nd) return;
    for (int i = 0; i < 110; i++) { free(nd->cv[i].w); free(nd->cv[i].b); }
    free(nd->ws); nn2_tp_destroy(); free(nd);
}
