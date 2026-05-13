/*
 * conv.c — Convolution operators.
 *
 * im2col + GEMM for 3x3 conv. Direct GEMM for 1x1 conv.
 * Fused bias + SiLU with AVX2 fast sigmoid approximation.
 */

#include "nn2_internal.h"
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ============ Parallel im2col helpers ============ */

typedef struct { const float* in; float* col; int C,H,W,Hout,Wout,spatial; } Im2colS2Ctx;

static void im2col_s2_worker(void* arg, int start, int end) {
    Im2colS2Ctx* x = (Im2colS2Ctx*)arg;
    for (int row = start; row < end; row++) {
        int c = row / 9, rem = row % 9;
        int kh = rem / 3, kw = rem % 3;
        const float* src = x->in + c * x->H * x->W;
        float* dst = x->col + row * x->spatial;
        int iw_off = kw - 1;
        for (int oh = 0; oh < x->Hout; oh++) {
            int ih = oh * 2 - 1 + kh;
            float* drow = dst + oh * x->Wout;
            if (ih < 0 || ih >= x->H) {
                memset(drow, 0, x->Wout * sizeof(float));
                continue;
            }
            const float* srow = src + ih * x->W;
            int ow_start = 0, ow_end = x->Wout;
            if (iw_off == -1) { drow[0] = 0.0f; ow_start = 1; }
            int last_iw = (ow_end - 1) * 2 + iw_off;
            if (last_iw >= x->W) { drow[ow_end - 1] = 0.0f; ow_end--; }
            int ow = ow_start;
#ifdef __AVX2__
            __m256i vidx = _mm256_setr_epi32(0,2,4,6,8,10,12,14);
            for (; ow + 8 <= ow_end; ow += 8) {
                __m256 g = _mm256_i32gather_ps(srow + ow*2 + iw_off, vidx, 4);
                _mm256_storeu_ps(drow + ow, g);
            }
#endif
            for (; ow < ow_end; ow++)
                drow[ow] = srow[ow * 2 + iw_off];
        }
    }
}

typedef struct { const float* in; float* col; int C,H,W; } Im2colS1Ctx;

static void im2col_s1_worker(void* arg, int start, int end) {
    Im2colS1Ctx* x = (Im2colS1Ctx*)arg;
    int H = x->H, W = x->W;
    int spatial = H * W;
    for (int row = start; row < end; row++) {
        int c = row / 9, rem = row % 9;
        int kh = rem / 3, kw = rem % 3;
        const float* src = x->in + c * H * W;
        float* dst = x->col + row * spatial;
        int ih_base = kh - 1;
        for (int oh = 0; oh < H; oh++) {
            int ih = oh + ih_base;
            if (ih < 0 || ih >= H) {
                memset(dst + oh * W, 0, W * sizeof(float));
                continue;
            }
            const float* srow = src + ih * W;
            float* drow = dst + oh * W;
            if (kw == 0) {
                drow[0] = 0.0f;
                memcpy(drow + 1, srow, (W - 1) * sizeof(float));
            } else if (kw == 1) {
                memcpy(drow, srow, W * sizeof(float));
            } else {
                memcpy(drow, srow + 1, (W - 1) * sizeof(float));
                drow[W - 1] = 0.0f;
            }
        }
    }
}

/* ============ im2col ============ */

void nn2_im2col(const float* in, int C, int H, int W,
                int kH, int kW, int stride, int pad,
                float* col)
{
    int Hout = (H + 2 * pad - kH) / stride + 1;
    int Wout = (W + 2 * pad - kW) / stride + 1;
    int spatial = Hout * Wout;

    /* Fast path: 3x3, stride 1, pad 1 — parallel across im2col rows */
    if (kH == 3 && kW == 3 && stride == 1 && pad == 1) {
        Im2colS1Ctx ctx = {in, col, C, H, W};
        int total_rows = C * 9;
        if (total_rows >= 36 && spatial >= 400) /* parallel only for large enough */
            nn2_tp_parallel_for(im2col_s1_worker, &ctx, total_rows, 9);
        else
            im2col_s1_worker(&ctx, 0, total_rows);
        return;
    }

    /* Fast path: 3x3, stride 2, pad 1 — parallel across im2col rows */
    if (kH == 3 && kW == 3 && stride == 2 && pad == 1) {
        Im2colS2Ctx ctx = {in, col, C, H, W, Hout, Wout, spatial};
        int total_rows = C * 9;
        if (total_rows >= 36 && spatial >= 200) /* parallel only for large enough */
            nn2_tp_parallel_for(im2col_s2_worker, &ctx, total_rows, 9);
        else
            im2col_s2_worker(&ctx, 0, total_rows);
        return;
    }

    /* Generic fallback */
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int col_row = (c * kH + kh) * kW + kw;
                float* dst = col + col_row * spatial;
                for (int oh = 0; oh < Hout; oh++) {
                    int ih = oh * stride - pad + kh;
                    for (int ow = 0; ow < Wout; ow++) {
                        int iw = ow * stride - pad + kw;
                        *dst++ = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                 ? in[c * H * W + ih * W + iw] : 0.0f;
                    }
                }
            }
        }
    }
}

/* ============ Fused bias + activation ============ */

#ifdef __AVX512F__
#include "fast_exp.h"
static inline __m512 fast_silu_avx512(__m512 x) { return _mm512_silu_fast(x); }
static inline __m512 fast_sigmoid_avx512(__m512 x) { return _mm512_sigmoid_fast(x); }
#endif

#ifdef __AVX2__
static inline __m256 fast_sigmoid_avx2(__m256 x)
{
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 z = _mm256_mul_ps(x, half);
    __m256 z2 = _mm256_mul_ps(z, z);
    __m256 v27 = _mm256_set1_ps(27.0f);
    __m256 num = _mm256_mul_ps(z, _mm256_add_ps(v27, z2));
    __m256 den = _mm256_add_ps(v27, _mm256_mul_ps(_mm256_set1_ps(9.0f), z2));
    __m256 t = _mm256_div_ps(num, den);
    return _mm256_fmadd_ps(half, t, half);
}
static inline __m256 fast_silu_avx2(__m256 x)
{
    return _mm256_mul_ps(x, fast_sigmoid_avx2(x));
}

#endif /* __AVX2__ */

static void apply_bias_act(float* out, const float* bias, int cout, int spatial,
                           int act)
{
    for (int co = 0; co < cout; co++) {
        float b = bias[co];
        float* row = out + co * spatial;
        int i = 0;

#ifdef __AVX512F__
        __m512 vb512 = _mm512_set1_ps(b);
        if (act == 1) {
            for (; i + 16 <= spatial; i += 16) {
                __m512 vx = _mm512_add_ps(_mm512_loadu_ps(row + i), vb512);
                _mm512_storeu_ps(row + i, fast_silu_avx512(vx));
            }
        } else if (act == 2) {
            for (; i + 16 <= spatial; i += 16) {
                __m512 vx = _mm512_add_ps(_mm512_loadu_ps(row + i), vb512);
                _mm512_storeu_ps(row + i, fast_sigmoid_avx512(vx));
            }
        } else {
            for (; i + 16 <= spatial; i += 16)
                _mm512_storeu_ps(row + i, _mm512_add_ps(_mm512_loadu_ps(row + i), vb512));
        }
#endif
#ifdef __AVX2__
        __m256 vb = _mm256_set1_ps(b);
        if (act == 1) {
            for (; i + 8 <= spatial; i += 8) {
                __m256 vx = _mm256_add_ps(_mm256_loadu_ps(row + i), vb);
                _mm256_storeu_ps(row + i, fast_silu_avx2(vx));
            }
        } else if (act == 2) {
            for (; i + 8 <= spatial; i += 8) {
                __m256 vx = _mm256_add_ps(_mm256_loadu_ps(row + i), vb);
                _mm256_storeu_ps(row + i, fast_sigmoid_avx2(vx));
            }
        } else {
            for (; i + 8 <= spatial; i += 8)
                _mm256_storeu_ps(row + i, _mm256_add_ps(_mm256_loadu_ps(row + i), vb));
        }
#endif
        for (; i < spatial; i++) {
            float x = row[i] + b;
            if (act == 1) {
                float s = 1.0f / (1.0f + expf(-x));
                x = x * s;
            } else if (act == 2) {
                x = 1.0f / (1.0f + expf(-x));
            }
            row[i] = x;
        }
    }
}

/* ============ Direct conv for small Cin (avoids im2col overhead) ============ */

#ifdef __AVX2__
static void direct_conv3x3_s2(const ConvLayer* L, const float* in, int H, int W,
                               float* out)
{
    int Hout = (H + 1) / 2;  /* (H + 2*1 - 3)/2 + 1 = H/2 */
    int Wout = (W + 1) / 2;
    int spatial = Hout * Wout;

    /* For each output channel, accumulate over cin×3×3 */
    for (int co = 0; co < L->cout; co++) {
        float* dst = out + co * spatial;
        const float* wt = L->w + co * L->cin * 9; /* [cin*9] */

        /* Initialize with bias */
        float b = L->b[co];
        int i = 0;
        __m256 vb = _mm256_set1_ps(b);
        for (; i + 8 <= spatial; i += 8)
            _mm256_storeu_ps(dst + i, vb);
        for (; i < spatial; i++)
            dst[i] = b;

        for (int ci = 0; ci < L->cin; ci++) {
            const float* src = in + ci * H * W;
            const float* w = wt + ci * 9;

            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    float wv = w[kh * 3 + kw];
                    if (wv == 0.0f) continue;
                    __m256 vw = _mm256_set1_ps(wv);

                    for (int oh = 0; oh < Hout; oh++) {
                        int ih = oh * 2 - 1 + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* srow = src + ih * W;
                        float* drow = dst + oh * Wout;
                        int iw_off = kw - 1;

                        int ow = 0;
                        /* Handle left boundary */
                        if (iw_off < 0) { ow = 1; } /* skip ow=0 (padding) */
                        /* Handle right boundary */
                        int ow_end = Wout;
                        if ((ow_end - 1) * 2 + iw_off >= W) ow_end--;

                        /* AVX2 gather path */
                        __m256i vidx = _mm256_setr_epi32(0,2,4,6,8,10,12,14);
                        for (; ow + 8 <= ow_end; ow += 8) {
                            int biw = ow * 2 + iw_off;
                            __m256 sv = _mm256_i32gather_ps(srow + biw, vidx, 4);
                            __m256 dv = _mm256_loadu_ps(drow + ow);
                            _mm256_storeu_ps(drow + ow, _mm256_fmadd_ps(vw, sv, dv));
                        }
                        for (; ow < ow_end; ow++) {
                            drow[ow] += wv * srow[ow * 2 + iw_off];
                        }
                    }
                }
            }
        }

        /* Apply activation (SiLU) */
        if (L->act == 1) {
            i = 0;
            for (; i + 8 <= spatial; i += 8) {
                __m256 vx = _mm256_loadu_ps(dst + i);
                _mm256_storeu_ps(dst + i, fast_silu_avx2(vx));
            }
            for (; i < spatial; i++) {
                float x = dst[i];
                dst[i] = x / (1.0f + expf(-x));
            }
        }
    }
}
#endif

/* ============ Winograd F(2,3) for 3x3 stride-1 conv ============
 *
 * Reduces multiplications by 2.25x: 9→4 per output element.
 * Input transform: V = B^T × d × B  (4×4 tile → 16 values)
 * Filter transform: U = G × g × G^T (3×3 → 16 values, pre-computed)
 * Output transform: y = A^T × m × A (4×4 → 2×2 output)
 * Core: 16 independent GEMMs of [Cout,Cin] × [Cin,tiles]
 */

/* Pre-transform filter weights for Winograd */
void nn2_prepare_winograd(ConvLayer* L)
{
    if (L->k != 3 || L->stride != 1 || L->pad != 1) return;
    /* U[16][cout][cin] */
    int co = L->cout, ci = L->cin;
    L->w_wino = (float*)malloc(16 * co * ci * sizeof(float));
    if (!L->w_wino) return;

    for (int oc = 0; oc < co; oc++) {
        for (int ic = 0; ic < ci; ic++) {
            const float* g = L->w + oc * ci * 9 + ic * 9; /* 3×3 filter */
            /* G × g (4×3) */
            float t[4][3];
            t[0][0] = g[0];       t[0][1] = g[1];       t[0][2] = g[2];
            t[1][0] = (g[0]+g[3]+g[6])*0.5f; t[1][1] = (g[1]+g[4]+g[7])*0.5f; t[1][2] = (g[2]+g[5]+g[8])*0.5f;
            t[2][0] = (g[0]-g[3]+g[6])*0.5f; t[2][1] = (g[1]-g[4]+g[7])*0.5f; t[2][2] = (g[2]-g[5]+g[8])*0.5f;
            t[3][0] = g[6];       t[3][1] = g[7];       t[3][2] = g[8];
            /* × G^T → U[4][4] */
            float U[16];
            for (int i = 0; i < 4; i++) {
                U[i*4+0] = t[i][0];
                U[i*4+1] = (t[i][0]+t[i][1]+t[i][2])*0.5f;
                U[i*4+2] = (t[i][0]-t[i][1]+t[i][2])*0.5f;
                U[i*4+3] = t[i][2];
            }
            /* Store as [16][co][ci]: component-major for GEMM batching */
            for (int a = 0; a < 16; a++)
                L->w_wino[a * co * ci + oc * ci + ic] = U[a];
        }
    }
}

static void winograd_conv3x3(const ConvLayer* L, const float* in, int H, int W,
                              float* out, float* work)
{
    int Hout = H, Wout = W; /* stride=1, pad=1 */
    int h_tiles = (Hout + 1) / 2;
    int w_tiles = (Wout + 1) / 2;
    int n_tiles = h_tiles * w_tiles;
    int ci = L->cin, co = L->cout;

    /* work layout: V[16][ci][tiles] + M[16][co][tiles] */
    float* V = work;
    float* M = V + 16 * ci * n_tiles;

    /* 1. Input transform: for each channel, extract tiles and transform */
    for (int c = 0; c < ci; c++) {
        const float* src = in + c * H * W;
        for (int th = 0; th < h_tiles; th++) {
            for (int tw = 0; tw < w_tiles; tw++) {
                int tile_idx = th * w_tiles + tw;
                int oh = th * 2, ow = tw * 2;

                /* Extract 4×4 tile with padding */
                float d[4][4];
                for (int i = 0; i < 4; i++) {
                    int ih = oh - 1 + i; /* pad=1 */
                    for (int j = 0; j < 4; j++) {
                        int iw = ow - 1 + j;
                        d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                  ? src[ih * W + iw] : 0.0f;
                    }
                }

                /* B^T × d (rows) */
                float t[4][4];
                for (int j = 0; j < 4; j++) {
                    t[0][j] = d[0][j] - d[2][j];
                    t[1][j] = d[1][j] + d[2][j];
                    t[2][j] = -d[1][j] + d[2][j];
                    t[3][j] = d[1][j] - d[3][j];
                }
                /* × B (cols) → V components */
                for (int i = 0; i < 4; i++) {
                    float v0 = t[i][0] - t[i][2];
                    float v1 = t[i][1] + t[i][2];
                    float v2 = -t[i][1] + t[i][2];
                    float v3 = t[i][1] - t[i][3];
                    V[(i*4+0) * ci * n_tiles + c * n_tiles + tile_idx] = v0;
                    V[(i*4+1) * ci * n_tiles + c * n_tiles + tile_idx] = v1;
                    V[(i*4+2) * ci * n_tiles + c * n_tiles + tile_idx] = v2;
                    V[(i*4+3) * ci * n_tiles + c * n_tiles + tile_idx] = v3;
                }
            }
        }
    }

    /* 2. Batched GEMM: M_a = U_a × V_a for each of 16 components */
    for (int a = 0; a < 16; a++) {
        const float* Ua = L->w_wino + a * co * ci;     /* [co, ci] */
        const float* Va = V + a * ci * n_tiles;          /* [ci, tiles] */
        float* Ma = M + a * co * n_tiles;                /* [co, tiles] */
        nn2_sgemm(co, ci, n_tiles, Ua, ci, Va, n_tiles, Ma, n_tiles);
    }

    /* 3. Output transform + bias + activation */
    for (int oc = 0; oc < co; oc++) {
        float bias = L->b[oc];
        for (int th = 0; th < h_tiles; th++) {
            for (int tw = 0; tw < w_tiles; tw++) {
                int tile_idx = th * w_tiles + tw;
                int oh = th * 2, ow = tw * 2;

                /* Gather M[4][4] for this tile */
                float m[4][4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        m[i][j] = M[(i*4+j) * co * n_tiles + oc * n_tiles + tile_idx];

                /* A^T × m (rows) */
                float t[2][4];
                for (int j = 0; j < 4; j++) {
                    t[0][j] = m[0][j] + m[1][j] + m[2][j];
                    t[1][j] = m[1][j] - m[2][j] - m[3][j];
                }
                /* × A (cols) → 2×2 output */
                float y00 = t[0][0] + t[0][1] + t[0][2] + bias;
                float y01 = t[0][1] - t[0][2] - t[0][3] + bias;
                float y10 = t[1][0] + t[1][1] + t[1][2] + bias;
                float y11 = t[1][1] - t[1][2] - t[1][3] + bias;

                /* Activation */
                if (L->act == 1) { /* SiLU */
                    #define SILU(x) ((x) / (1.0f + expf(-(x))))
                    y00 = SILU(y00); y01 = SILU(y01);
                    y10 = SILU(y10); y11 = SILU(y11);
                    #undef SILU
                }

                /* Write to output (handle boundary for odd H/W) */
                float* dst = out + oc * Hout * Wout;
                if (oh < Hout && ow < Wout)     dst[oh * Wout + ow] = y00;
                if (oh < Hout && ow+1 < Wout)   dst[oh * Wout + ow + 1] = y01;
                if (oh+1 < Hout && ow < Wout)   dst[(oh+1) * Wout + ow] = y10;
                if (oh+1 < Hout && ow+1 < Wout) dst[(oh+1) * Wout + ow + 1] = y11;
            }
        }
    }
}

/* ============ Conv2d ============ */

void nn2_conv2d(const ConvLayer* L, const float* in, int H, int W,
                float* out, float* col_buf)
{
    int Hout = (H + 2 * L->pad - L->k) / L->stride + 1;
    int Wout = (W + 2 * L->pad - L->k) / L->stride + 1;
    int spatial = Hout * Wout;

    if (L->k == 1 && L->stride == 1 && L->pad == 0) {
        /* INT8 path for pointwise conv (if quantized weights available) */
        if (L->w_int8_packed) {
            /* Quantize input FP32 → INT8 */
            int n_in = L->cin * spatial;
            int8_t* a_int8 = (int8_t*)col_buf; /* borrow col_buf for quantized input */
            float inv_scale = 1.0f / L->act_in_scale;
            for (int i = 0; i < n_in; i++) {
                int v = (int)(in[i] * inv_scale + 0.5f);
                a_int8[i] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
            }
            nn2_int8_gemm_fused(a_int8, L->cout, L->cin, spatial,
                                L->w_int8_packed, L->col_sums,
                                L->w_scales, L->b,
                                L->act_in_scale, L->act_out_scale,
                                L->act, out);
            return;
        }
        /* Fused GEMM+bias+activation — use packed weights if available */
        if (L->w_packed) {
            nn2_sgemm_bias_act_packed(L->cout, L->cin, spatial,
                                      L->w_packed, in,
                                      out, spatial, L->b, L->act);
        } else {
            nn2_sgemm_bias_act(L->cout, L->cin, spatial,
                               L->w, L->cin, in, spatial,
                               out, spatial, L->b, L->act);
        }
        return;
    }

    int K_gemm = L->cin * L->k * L->k;

    /* Winograd F(2,3) for 3x3 stride-1 — 2.25x fewer multiplications.
     * Wins on single-thread (fewer FLOPs). Loses on multi-thread (16 small GEMMs
     * can't parallelize as well as one large GEMM). */
#ifdef __AVX512F__
    /* Implicit im2col: skip the im2col buffer entirely.
     * Load input on-the-fly during GEMM. Saves 2× memory traffic. */
    /* Implicit im2col: best for small Cin (K=Cin*9 small → few iterations per tile).
     * For large Cin (K>200), regular im2col+GEMM wins due to tighter inner loop. */
    if (L->k == 3 && L->stride == 1 && L->pad == 1
        && L->w_packed && spatial >= 2500) {
        /* Implicit im2col: no buffer, loads input on-the-fly.
         * Wins when im2col buffer (K×spatial×4B) would thrash L2.
         * Threshold 2500 = ~50×50 minimum (avoids edge-case issues at very small sizes). */
        nn2_conv3x3_s1_implicit(in, L->w, L->b, L->cin, L->cout, H, W, L->act, out, L->w_packed);
        return;
    }
#endif
    if (L->w_wino && L->k == 3 && L->stride == 1 && L->pad == 1
        && nn2_tp_num_threads() <= 2) {
        /* workspace: 16*(Cin+Cout)*tiles floats. Use col_buf. */
        int tiles = ((Hout+1)/2) * ((Wout+1)/2);
        /* Check col_buf is large enough: need 16*(cin+cout)*tiles */
        nn2_winograd_conv3x3(in, L->cin, H, W, L->w_wino, L->b, L->cout, L->act,
                             out, col_buf);
        return;
    }

    /* Standard im2col + fused GEMM (with packed weights if available) */
    nn2_im2col(in, L->cin, H, W, L->k, L->k, L->stride, L->pad, col_buf);
    if (L->w_packed) {
        nn2_sgemm_bias_act_packed(L->cout, K_gemm, spatial,
                                  L->w_packed, col_buf,
                                  out, spatial, L->b, L->act);
    } else {
        nn2_sgemm_bias_act(L->cout, K_gemm, spatial,
                           L->w, K_gemm, col_buf, spatial,
                           out, spatial, L->b, L->act);
    }
}

