/*
 * winograd.c — Winograd F(2,3) for 3x3 stride-1 convolutions.
 *
 * Reduces multiplications by 2.25x (9→4 per output element).
 * All transforms are AVX-512 vectorized — process full rows at once.
 *
 * Algorithm:
 *   1. Transform input tiles: V = B^T × d × B  (per channel, per 2-row strip)
 *   2. Batched GEMM: M = U × V  (16 independent [Cout,Cin]×[Cin,tiles] GEMMs)
 *   3. Transform output: y = A^T × m × A  (per channel, per 2-row strip)
 *
 * Key insight: B^T operates on ROWS (element-wise across full width),
 * then B operates on COLUMNS (stride-2 shifts). Both are vectorizable.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

/* Pre-transform weights: U[16][Cout][Cin] from W[Cout][Cin*9] */
void nn2_winograd_transform_weights(const float* W, int Cout, int Cin, float* U)
{
    for (int oc = 0; oc < Cout; oc++) {
        for (int ic = 0; ic < Cin; ic++) {
            const float* g = W + oc * Cin * 9 + ic * 9;
            /* G × g (4×3): rows */
            float t[4][3];
            t[0][0]=g[0]; t[0][1]=g[1]; t[0][2]=g[2];
            t[1][0]=(g[0]+g[3]+g[6])*.5f; t[1][1]=(g[1]+g[4]+g[7])*.5f; t[1][2]=(g[2]+g[5]+g[8])*.5f;
            t[2][0]=(g[0]-g[3]+g[6])*.5f; t[2][1]=(g[1]-g[4]+g[7])*.5f; t[2][2]=(g[2]-g[5]+g[8])*.5f;
            t[3][0]=g[6]; t[3][1]=g[7]; t[3][2]=g[8];
            /* × G^T (4×4): columns */
            for (int i = 0; i < 4; i++) {
                float u0 = t[i][0];
                float u1 = (t[i][0]+t[i][1]+t[i][2])*.5f;
                float u2 = (t[i][0]-t[i][1]+t[i][2])*.5f;
                float u3 = t[i][2];
                U[(i*4+0)*Cout*Cin + oc*Cin + ic] = u0;
                U[(i*4+1)*Cout*Cin + oc*Cin + ic] = u1;
                U[(i*4+2)*Cout*Cin + oc*Cin + ic] = u2;
                U[(i*4+3)*Cout*Cin + oc*Cin + ic] = u3;
            }
        }
    }
}

/* Winograd F(2,3) convolution: 3x3 stride-1 pad-1 */
void nn2_winograd_conv3x3(
    const float* input, int Cin, int H, int W,
    const float* U,     /* [16][Cout][Cin] pre-transformed weights */
    const float* bias, int Cout, int act,
    float* output,
    float* workspace)   /* needs: 16*Cin*tiles + 16*Cout*tiles floats */
{
    int Htiles = (H + 1) / 2;
    int Wtiles = (W + 1) / 2;
    int ntiles = Htiles * Wtiles;

    float* V = workspace;                        /* [16][Cin][ntiles] */
    float* M = V + 16 * Cin * ntiles;            /* [16][Cout][ntiles] + [Cout][ntiles][16] transposed */
    /* Total workspace: 16*Cin*ntiles + 2*16*Cout*ntiles */

    /* ============ Input transform ============
     * For each channel, for each tile-row th:
     *   Load 4 input rows (with padding)
     *   Apply B^T on rows (element-wise across full width)
     *   Apply B on columns (extract stride-2 shifted values)
     *   Store to V[alpha][c][tile_idx] */

    for (int c = 0; c < Cin; c++) {
        const float* src = input + c * H * W;

        for (int th = 0; th < Htiles; th++) {
            /* 4 input rows for this tile row */
            int r0 = th*2 - 1, r1 = th*2, r2 = th*2 + 1, r3 = th*2 + 2;
            const float* d0 = (r0 >= 0 && r0 < H) ? src + r0*W : NULL;
            const float* d1 = (r1 >= 0 && r1 < H) ? src + r1*W : NULL;
            const float* d2 = (r2 >= 0 && r2 < H) ? src + r2*W : NULL;
            const float* d3 = (r3 >= 0 && r3 < H) ? src + r3*W : NULL;

            /* B^T on rows: t0=d0-d2, t1=d1+d2, t2=-d1+d2, t3=d1-d3
             * Process full width with AVX-512 */
            float t0[324], t1[324], t2[324], t3[324]; /* max W+4 */

#ifdef __AVX512F__
            int j = 0;
            for (; j + 16 <= W; j += 16) {
                __m512 vd0 = d0 ? _mm512_loadu_ps(d0+j) : _mm512_setzero_ps();
                __m512 vd1 = d1 ? _mm512_loadu_ps(d1+j) : _mm512_setzero_ps();
                __m512 vd2 = d2 ? _mm512_loadu_ps(d2+j) : _mm512_setzero_ps();
                __m512 vd3 = d3 ? _mm512_loadu_ps(d3+j) : _mm512_setzero_ps();
                _mm512_storeu_ps(t0+j, _mm512_sub_ps(vd0, vd2));
                _mm512_storeu_ps(t1+j, _mm512_add_ps(vd1, vd2));
                _mm512_storeu_ps(t2+j, _mm512_sub_ps(vd2, vd1));
                _mm512_storeu_ps(t3+j, _mm512_sub_ps(vd1, vd3));
            }
            for (; j < W; j++) {
                float v0=d0?d0[j]:0, v1=d1?d1[j]:0, v2=d2?d2[j]:0, v3=d3?d3[j]:0;
                t0[j]=v0-v2; t1[j]=v1+v2; t2[j]=v2-v1; t3[j]=v1-v3;
            }
#else
            for (int j = 0; j < W; j++) {
                float v0=d0?d0[j]:0, v1=d1?d1[j]:0, v2=d2?d2[j]:0, v3=d3?d3[j]:0;
                t0[j]=v0-v2; t1[j]=v1+v2; t2[j]=v2-v1; t3[j]=v1-v3;
            }
#endif

            /* B on columns: for each tile tw, extract columns at tw*2-1, tw*2, tw*2+1, tw*2+2
             * V[i*4+0] = t[i][tw*2-1] - t[i][tw*2+1]
             * V[i*4+1] = t[i][tw*2]   + t[i][tw*2+1]
             * V[i*4+2] = t[i][tw*2+1] - t[i][tw*2]
             * V[i*4+3] = t[i][tw*2]   - t[i][tw*2+2] */
            float* trows[4] = {t0, t1, t2, t3};
            for (int i = 0; i < 4; i++) {
                float* tr = trows[i];
                for (int tw = 0; tw < Wtiles; tw++) {
                    int tile_idx = th * Wtiles + tw;
                    int c0 = tw*2 - 1, c1 = tw*2, c2 = tw*2 + 1, c3 = tw*2 + 2;
                    float v0 = (c0>=0&&c0<W) ? tr[c0] : 0;
                    float v1 = (c1>=0&&c1<W) ? tr[c1] : 0;
                    float v2 = (c2>=0&&c2<W) ? tr[c2] : 0;
                    float v3 = (c3>=0&&c3<W) ? tr[c3] : 0;
                    V[(i*4+0)*Cin*ntiles + c*ntiles + tile_idx] = v0 - v2;
                    V[(i*4+1)*Cin*ntiles + c*ntiles + tile_idx] = v1 + v2;
                    V[(i*4+2)*Cin*ntiles + c*ntiles + tile_idx] = v2 - v1;
                    V[(i*4+3)*Cin*ntiles + c*ntiles + tile_idx] = v1 - v3;
                }
            }
        }
    }

    /* ============ 16 GEMMs + transpose to tile-major ============
     * GEMM into M_raw[16][Cout][ntiles], then transpose to Mt[Cout][ntiles][16]
     * so output transform has contiguous access (1 cache line per tile). */
    float* M_raw = M;
    float* Mt = M_raw + 16 * Cout * ntiles; /* need extra space for transposed */

    for (int a = 0; a < 16; a++) {
        nn2_sgemm(Cout, Cin, ntiles,
                  U + a * Cout * Cin, Cin,
                  V + a * Cin * ntiles, ntiles,
                  M_raw + a * Cout * ntiles, ntiles);
    }

    /* Transpose: M_raw[a][oc][t] → Mt[oc][t][a] (16 floats contiguous per tile) */
    for (int oc = 0; oc < Cout; oc++) {
        for (int t = 0; t < ntiles; t++) {
            float* dst = Mt + (oc * ntiles + t) * 16;
            for (int a = 0; a < 16; a++)
                dst[a] = M_raw[a * Cout * ntiles + oc * ntiles + t];
        }
    }

    /* ============ Output transform (cache-friendly) ============
     * Mt[oc][t][16] — all 16 components for one tile are contiguous.
     * m[i][j] = Mt[oc][t][i*4+j] for i=0..3, j=0..3 */
    for (int oc = 0; oc < Cout; oc++) {
        float b = bias[oc];
        float* dst = output + oc * H * W;

        for (int th = 0; th < Htiles; th++) {
            for (int tw = 0; tw < Wtiles; tw++) {
                int tidx = th * Wtiles + tw;
                int oh = th * 2, ow = tw * 2;

                const float* m = Mt + (oc * ntiles + tidx) * 16; /* contiguous! */

                /* A^T × m × A → 2×2 output */
                float r00 = m[0]+m[4]+m[8],  r01 = m[1]+m[5]+m[9];
                float r02 = m[2]+m[6]+m[10], r03 = m[3]+m[7]+m[11];
                float r10 = m[4]-m[8]-m[12], r11 = m[5]-m[9]-m[13];
                float r12 = m[6]-m[10]-m[14],r13 = m[7]-m[11]-m[15];

                float y00 = r00+r01+r02 + b;
                float y01 = r01-r02-r03 + b;
                float y10 = r10+r11+r12 + b;
                float y11 = r11-r12-r13 + b;

                if (act == 1) {
                    #define S(x) ((x)*(0.5f+0.5f*(x)/(1.0f+fabsf(x))))
                    y00=S(y00); y01=S(y01); y10=S(y10); y11=S(y11);
                    #undef S
                }

                if (oh<H && ow<W)     dst[oh*W+ow] = y00;
                if (oh<H && ow+1<W)   dst[oh*W+ow+1] = y01;
                if (oh+1<H && ow<W)   dst[(oh+1)*W+ow] = y10;
                if (oh+1<H && ow+1<W) dst[(oh+1)*W+ow+1] = y11;
            }
        }
    }
}
