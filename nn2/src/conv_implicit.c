/*
 * conv_implicit.c — 3x3 stride-1 conv WITHOUT im2col buffer.
 *
 * Replaces im2col+GEMM with a single pass that loads input on-the-fly.
 * Saves 2×15MB memory traffic for the biggest bottleneck layers.
 *
 * For each N-tile of 32 output pixels (same row, consecutive columns):
 *   For each k = (c, kh, kw):
 *     Load 32 input values from input[c][oh+kh-1][ow_start+kw-1 .. ow_start+kw+30]
 *     This is a contiguous load (shifted by kw-1) — perfect for AVX-512!
 *   Multiply-accumulate with weight A[m][k]
 *   Store with fused bias+SiLU
 *
 * Threading: parallel across N-tiles (same as regular GEMM).
 */

#include "nn2_internal.h"
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifdef __AVX512F__

typedef struct {
    const float* input;    /* [Cin, H, W] */
    const float* weight;   /* [Cout, Cin*9] original */
    const float* w_packed; /* packed A or NULL */
    const float* bias;     /* [Cout] */
    float* output;         /* [Cout, H, W] */
    int Cin, Cout, H, W;
    int act;
} ImplicitConvCtx;

#include "fast_exp.h"
static inline __m512 _isilu(__m512 x) { return _mm512_silu_fast(x); }

/* Worker: process output tiles [start, end) where each tile = 32 consecutive output columns */
static void implicit_conv_worker(void* arg, int start, int end)
{
    ImplicitConvCtx* g = (ImplicitConvCtx*)arg;
    int H = g->H, W = g->W;
    int Cin = g->Cin, Cout = g->Cout;
    int spatial = H * W;

    for (int ti = start; ti < end; ti++) {
        /* Tile → (oh, ow_start): tile = oh * tiles_per_row + ow_block */
        int tiles_per_row = (W + 31) / 32;
        int oh = ti / tiles_per_row;
        int ow_block = ti % tiles_per_row;
        int ow_start = ow_block * 32;
        int nr = 32;
        if (ow_start + nr > W) nr = W - ow_start;

        /* Process MR=6 output channels at a time */
        for (int oc = 0; oc < Cout; oc += NN2_MR) {
            int mr = (oc + NN2_MR <= Cout) ? NN2_MR : Cout - oc;

            __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();

            int K = Cin * 9;
            int panel = oc / NN2_MR;
            const float* Ap = g->w_packed ? g->w_packed + panel * K * NN2_MR : NULL;

            for (int ci = 0; ci < Cin; ci++) {
                const float* src = g->input + ci * H * W;

                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh - 1 + kh;
                    if (ih < 0 || ih >= H) continue;
                    const float* srow = src + ih * W;

                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow_start - 1 + kw;

                        /* Load 32 input values: srow[iw .. iw+31] */
                        __m512 b0, b1;
                        if (iw >= 0 && iw + 32 <= W && nr == 32) {
                            b0 = _mm512_loadu_ps(srow + iw);
                            b1 = _mm512_loadu_ps(srow + iw + 16);
                        } else {
                            /* Edge case: partial load with bounds check */
                            float tmp[32] = {0};
                            for (int j = 0; j < nr; j++) {
                                int iw_j = iw + j;
                                if (iw_j >= 0 && iw_j < W) tmp[j] = srow[iw_j];
                            }
                            b0 = _mm512_loadu_ps(tmp);
                            b1 = _mm512_loadu_ps(tmp + 16);
                        }

                        int k = ci * 9 + kh * 3 + kw;
                        const float* a = Ap + k * NN2_MR; /* packed: 6 consecutive */
                        __m512 va;
                        va=_mm512_set1_ps(a[0]); c00=_mm512_fmadd_ps(va,b0,c00); c01=_mm512_fmadd_ps(va,b1,c01);
                        if(mr>1){va=_mm512_set1_ps(a[1]); c10=_mm512_fmadd_ps(va,b0,c10); c11=_mm512_fmadd_ps(va,b1,c11);}
                        if(mr>2){va=_mm512_set1_ps(a[2]); c20=_mm512_fmadd_ps(va,b0,c20); c21=_mm512_fmadd_ps(va,b1,c21);}
                        if(mr>3){va=_mm512_set1_ps(a[3]); c30=_mm512_fmadd_ps(va,b0,c30); c31=_mm512_fmadd_ps(va,b1,c31);}
                        if(mr>4){va=_mm512_set1_ps(a[4]); c40=_mm512_fmadd_ps(va,b0,c40); c41=_mm512_fmadd_ps(va,b1,c41);}
                        if(mr>5){va=_mm512_set1_ps(a[5]); c50=_mm512_fmadd_ps(va,b0,c50); c51=_mm512_fmadd_ps(va,b1,c51);}
                    }
                }
            }

            /* Fused bias + SiLU + store */
            #define ISTORE(i, lo, hi) if(i<mr) { \
                __m512 vb = _mm512_set1_ps(g->bias[oc+(i)]); \
                lo = _mm512_add_ps(lo, vb); hi = _mm512_add_ps(hi, vb); \
                if (g->act == 1) { lo = _isilu(lo); hi = _isilu(hi); } \
                _mm512_storeu_ps(g->output + (oc+(i))*spatial + oh*W + ow_start, lo); \
                if (nr > 16) _mm512_storeu_ps(g->output + (oc+(i))*spatial + oh*W + ow_start + 16, hi); \
            }
            ISTORE(0,c00,c01); ISTORE(1,c10,c11); ISTORE(2,c20,c21);
            ISTORE(3,c30,c31); ISTORE(4,c40,c41); ISTORE(5,c50,c51);
            #undef ISTORE
        }
    }
}

void nn2_conv3x3_s1_implicit(const float* input, const float* weight, const float* bias,
                              int Cin, int Cout, int H, int W, int act, float* output,
                              const float* w_packed)
{
    ImplicitConvCtx ctx = {input, weight, w_packed, bias, output, Cin, Cout, H, W, act};
    /* Tiles: each tile = one (oh, ow_block). ow_block = 32 columns per row.
     * Total = H * tiles_per_row. Tile index → (oh, ow_start) via divmod. */
    int tiles_per_row = (W + 31) / 32;
    int total_tiles = H * tiles_per_row;

    int grain = (total_tiles > 128) ? total_tiles / (nn2_tp_num_threads() * 4) : 1;
    if (grain < 1) grain = 1;
    nn2_tp_parallel_for(implicit_conv_worker, &ctx, total_tiles, grain);
}

/* ============ Implicit 3x3 stride-2 conv ============ */

typedef struct {
    const float* input;
    const float* weight;
    const float* bias;
    float* output;
    int Cin, Cout, H, W, Hout, Wout;
    int act;
} ImplicitConvS2Ctx;

static void implicit_conv_s2_worker(void* arg, int start, int end)
{
    ImplicitConvS2Ctx* g = (ImplicitConvS2Ctx*)arg;
    int H = g->H, W = g->W, Wout = g->Wout;
    int Cin = g->Cin, Cout = g->Cout;
    int out_spatial = g->Hout * g->Wout;

    __m256i vidx = _mm256_setr_epi32(0,2,4,6,8,10,12,14); /* stride-2 gather */

    for (int ti = start; ti < end; ti++) {
        int n_start = ti * 16; /* 16 output pixels per tile (AVX-512 half) */
        int oh = n_start / Wout;
        int ow_start = n_start % Wout;
        int nr = 16;
        if (ow_start + nr > Wout) nr = Wout - ow_start;

        for (int oc = 0; oc < Cout; oc += NN2_MR) {
            int mr = (oc + NN2_MR <= Cout) ? NN2_MR : Cout - oc;

            __m512 c0=_mm512_setzero_ps();
            __m512 c1=_mm512_setzero_ps();
            __m512 c2=_mm512_setzero_ps();
            __m512 c3=_mm512_setzero_ps();
            __m512 c4=_mm512_setzero_ps();
            __m512 c5=_mm512_setzero_ps();

            const float* Wp[6];
            Wp[0] = g->weight + oc*Cin*9;
            for (int i=1;i<6;i++) Wp[i] = (i<mr) ? Wp[0]+i*Cin*9 : Wp[0];

            for (int ci = 0; ci < Cin; ci++) {
                const float* src = g->input + ci*H*W;
                int woff = ci*9;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh*2 - 1 + kh;
                    if (ih < 0 || ih >= H) continue;
                    const float* srow = src + ih*W;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw_base = ow_start*2 - 1 + kw;
                        __m512 b;
                        /* Stride-2 gather: load srow[iw_base], srow[iw_base+2], ... */
                        if (iw_base >= 0 && iw_base + nr*2 <= W && nr == 16) {
                            /* Use two AVX2 gathers → combine into ZMM */
                            __m256 lo = _mm256_i32gather_ps(srow+iw_base, vidx, 4);
                            __m256 hi = _mm256_i32gather_ps(srow+iw_base+16, vidx, 4);
                            b = _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1);
                        } else {
                            float tmp[16] = {0};
                            for (int j = 0; j < nr; j++) {
                                int iw = iw_base + j*2;
                                if (iw >= 0 && iw < W) tmp[j] = srow[iw];
                            }
                            b = _mm512_loadu_ps(tmp);
                        }
                        int k = woff + kh*3 + kw;
                        __m512 va;
                        va=_mm512_set1_ps(Wp[0][k]); c0=_mm512_fmadd_ps(va,b,c0);
                        if(mr>1){va=_mm512_set1_ps(Wp[1][k]); c1=_mm512_fmadd_ps(va,b,c1);}
                        if(mr>2){va=_mm512_set1_ps(Wp[2][k]); c2=_mm512_fmadd_ps(va,b,c2);}
                        if(mr>3){va=_mm512_set1_ps(Wp[3][k]); c3=_mm512_fmadd_ps(va,b,c3);}
                        if(mr>4){va=_mm512_set1_ps(Wp[4][k]); c4=_mm512_fmadd_ps(va,b,c4);}
                        if(mr>5){va=_mm512_set1_ps(Wp[5][k]); c5=_mm512_fmadd_ps(va,b,c5);}
                    }
                }
            }

            #define IS2(i, acc) if(i<mr) { \
                __m512 vb = _mm512_set1_ps(g->bias[oc+(i)]); \
                acc = _mm512_add_ps(acc, vb); \
                if (g->act==1) acc = _isilu(acc); \
                if (nr>=16) _mm512_storeu_ps(g->output+(oc+(i))*out_spatial+oh*Wout+ow_start, acc); \
                else { __mmask16 mask = ((__mmask16)1<<nr)-1; \
                       _mm512_mask_storeu_ps(g->output+(oc+(i))*out_spatial+oh*Wout+ow_start, mask, acc); } \
            }
            IS2(0,c0);IS2(1,c1);IS2(2,c2);IS2(3,c3);IS2(4,c4);IS2(5,c5);
            #undef IS2
        }
    }
}

void nn2_conv3x3_s2_implicit(const float* input, const float* weight, const float* bias,
                              int Cin, int Cout, int H, int W, int act, float* output)
{
    int Hout = (H+1)/2, Wout = (W+1)/2;
    ImplicitConvS2Ctx ctx = {input, weight, bias, output, Cin, Cout, H, W, Hout, Wout, act};
    int tiles_per_row = (Wout + 15) / 16;
    int total_tiles = Hout * tiles_per_row;
    int grain = (total_tiles > 64) ? total_tiles / (nn2_tp_num_threads()*4) : 1;
    if (grain < 1) grain = 1;
    nn2_tp_parallel_for(implicit_conv_s2_worker, &ctx, total_tiles, grain);
}

#endif /* __AVX512F__ */
