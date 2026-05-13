/*
 * conv_tiled.c — Cross-layer fused 3x3→3x3 bottleneck.
 *
 * Instead of:  cv1(full H×W) → scratch[c×H×W] → cv2(full H×W)
 * We do:       cv1(3 rows at a time) → line_buf[c×3×W] → cv2(1 row) → output
 *
 * The line buffer (3×c×W floats) stays in L1 cache.
 * Eliminates writing/reading the full intermediate feature map.
 *
 * For cv1: 3x3 s=1 p=1, Cin→c
 * For cv2: 3x3 s=1 p=1, c→c
 * With optional residual add on cv2 output.
 */

#include "nn2_internal.h"
#include <string.h>
#include <immintrin.h>

#ifdef __AVX512F__
#include "fast_exp.h"

/* Compute one output row of a 3x3 s=1 p=1 conv with packed weights.
 * Reads 3 input rows (ih-1, ih, ih+1) or zero-pads if out of bounds.
 * Output: out[cout × W] for row oh. */
static void conv3x3_one_row(
    const float* input, int Cin, int H, int W, int oh,
    const float* w_packed, const float* bias, int Cout, int act,
    float* out_row)
{
    int K = Cin * 9;
    int panels = (Cout + NN2_MR - 1) / NN2_MR;

    /* Process NR=32 output columns at a time */
    int tiles_w = (W + 31) / 32;
    for (int tw = 0; tw < tiles_w; tw++) {
        int ow_start = tw * 32;
        int nr = (ow_start + 32 <= W) ? 32 : W - ow_start;

        for (int p = 0; p < panels; p++) {
            int oc = p * NN2_MR;
            int mr = (oc + NN2_MR <= Cout) ? NN2_MR : Cout - oc;

            __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
            __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
            __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
            __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
            __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
            __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();

            const float* Ap = w_packed + p * K * NN2_MR;

            for (int ci = 0; ci < Cin; ci++) {
                const float* src = input + ci * H * W;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = oh - 1 + kh;
                    if (ih < 0 || ih >= H) continue;
                    const float* srow = src + ih * W;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = ow_start - 1 + kw;
                        __m512 b0, b1;
                        if (iw >= 0 && iw + 32 <= W && nr == 32) {
                            b0 = _mm512_loadu_ps(srow + iw);
                            b1 = _mm512_loadu_ps(srow + iw + 16);
                        } else {
                            float tmp[32] = {0};
                            for (int j = 0; j < nr; j++)
                                if (iw+j >= 0 && iw+j < W) tmp[j] = srow[iw+j];
                            b0 = _mm512_loadu_ps(tmp);
                            b1 = _mm512_loadu_ps(tmp + 16);
                        }
                        int k = (ci * 9 + kh * 3 + kw);
                        const float* a = Ap + k * NN2_MR;
                        __m512 va;
                        va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01);
                        if(mr>1){va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11);}
                        if(mr>2){va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21);}
                        if(mr>3){va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31);}
                        if(mr>4){va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41);}
                        if(mr>5){va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51);}
                    }
                }
            }

            #define ST(i,lo,hi) if(i<mr){ \
                __m512 vb=_mm512_set1_ps(bias[oc+(i)]); \
                lo=_mm512_add_ps(lo,vb);hi=_mm512_add_ps(hi,vb); \
                if(act==1){lo=_mm512_silu_fast(lo);hi=_mm512_silu_fast(hi);} \
                _mm512_storeu_ps(out_row+(oc+(i))*W+ow_start,lo); \
                if(nr>16) _mm512_storeu_ps(out_row+(oc+(i))*W+ow_start+16,hi); \
            }
            ST(0,c00,c01);ST(1,c10,c11);ST(2,c20,c21);
            ST(3,c30,c31);ST(4,c40,c41);ST(5,c50,c51);
            #undef ST
        }
    }
}

/* Fused bottleneck: cv1(3x3) → cv2(3x3) with line-buffer streaming.
 * cv2 reads from a 3-row circular buffer of cv1 output.
 * Optional residual add from 'residual' pointer. */
void nn2_bottleneck_fused(
    const float* input, int Cin, int H, int W,
    const ConvLayer* cv1, const ConvLayer* cv2,
    const float* residual, /* NULL or [c×H×W] for shortcut */
    float* output,         /* [c×H×W] */
    float* line_buf)       /* workspace: [3×c×W] */
{
    int c = cv1->cout;  /* cv1 output channels = cv2 input channels */

    /* line_buf holds 3 rows of cv1 output in circular fashion */
    /* Row mapping: line_buf[(row % 3) * c * W ... ] */

    /* Pre-compute cv1 rows 0 and 1 into line buffer */
    conv3x3_one_row(input, Cin, H, W, 0, cv1->w_packed, cv1->b, c, cv1->act,
                    line_buf + 0 * c * W);
    if (H > 1)
        conv3x3_one_row(input, Cin, H, W, 1, cv1->w_packed, cv1->b, c, cv1->act,
                        line_buf + 1 * c * W);

    /* Stream: for each output row of cv2, compute cv1's next row then cv2 */
    for (int oh = 0; oh < H; oh++) {
        /* Compute cv1 row oh+1 (if valid) into circular buffer */
        int next_row = oh + 1;
        if (next_row < H) {
            int slot = next_row % 3;
            conv3x3_one_row(input, Cin, H, W, next_row, cv1->w_packed, cv1->b, c, cv1->act,
                            line_buf + slot * c * W);
        }

        /* Now cv2 for row oh: reads cv1 rows (oh-1, oh, oh+1) from line buffer.
         * We build a virtual 3-row "input" for cv2's row computation. */

        /* cv2 processes: for each output pixel at (oh, ow), accumulate over
         * c channels × 3 kh × 3 kw, reading from line_buf rows. */
        int K2 = c * 9;
        int panels2 = (cv2->cout + NN2_MR - 1) / NN2_MR;
        int tiles_w = (W + 31) / 32;

        for (int tw = 0; tw < tiles_w; tw++) {
            int ow_start = tw * 32;
            int nr = (ow_start + 32 <= W) ? 32 : W - ow_start;

            for (int p2 = 0; p2 < panels2; p2++) {
                int oc = p2 * NN2_MR;
                int mr = (oc + NN2_MR <= cv2->cout) ? NN2_MR : cv2->cout - oc;

                __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
                __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
                __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
                __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
                __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
                __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();

                const float* Ap2 = cv2->w_packed + p2 * K2 * NN2_MR;

                for (int ci = 0; ci < c; ci++) {
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = oh - 1 + kh;
                        if (ih < 0 || ih >= H) continue;
                        /* Read from circular line buffer */
                        int slot = ih % 3;
                        const float* srow = line_buf + slot * c * W + ci * W;
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = ow_start - 1 + kw;
                            __m512 b0, b1;
                            if (iw >= 0 && iw + 32 <= W && nr == 32) {
                                b0 = _mm512_loadu_ps(srow + iw);
                                b1 = _mm512_loadu_ps(srow + iw + 16);
                            } else {
                                float tmp[32] = {0};
                                for (int j = 0; j < nr; j++)
                                    if (iw+j >= 0 && iw+j < W) tmp[j] = srow[iw+j];
                                b0 = _mm512_loadu_ps(tmp);
                                b1 = _mm512_loadu_ps(tmp + 16);
                            }
                            int k = ci * 9 + kh * 3 + kw;
                            const float* a = Ap2 + k * NN2_MR;
                            __m512 va;
                            va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01);
                            if(mr>1){va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11);}
                            if(mr>2){va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21);}
                            if(mr>3){va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31);}
                            if(mr>4){va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41);}
                            if(mr>5){va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51);}
                        }
                    }
                }

                /* Store cv2 output + optional residual */
                int out_spatial = H * W;
                #define ST2(i,lo,hi) if(i<mr){ \
                    __m512 vb=_mm512_set1_ps(cv2->b[oc+(i)]); \
                    lo=_mm512_add_ps(lo,vb);hi=_mm512_add_ps(hi,vb); \
                    if(cv2->act==1){lo=_mm512_silu_fast(lo);hi=_mm512_silu_fast(hi);} \
                    if(residual){ \
                        lo=_mm512_add_ps(lo,_mm512_loadu_ps(residual+(oc+(i))*out_spatial+oh*W+ow_start)); \
                        if(nr>16)hi=_mm512_add_ps(hi,_mm512_loadu_ps(residual+(oc+(i))*out_spatial+oh*W+ow_start+16)); \
                    } \
                    _mm512_storeu_ps(output+(oc+(i))*out_spatial+oh*W+ow_start,lo); \
                    if(nr>16) _mm512_storeu_ps(output+(oc+(i))*out_spatial+oh*W+ow_start+16,hi); \
                }
                ST2(0,c00,c01);ST2(1,c10,c11);ST2(2,c20,c21);
                ST2(3,c30,c31);ST2(4,c40,c41);ST2(5,c50,c51);
                #undef ST2
            }
        }
    }
}

#endif /* __AVX512F__ */
