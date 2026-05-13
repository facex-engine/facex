/*
 * antispoof_ops.c — Additional ops needed for MiniFASNet that don't exist in
 * the YOLOv8-tuned nn2 core: PReLU (per-channel), global avg pool, SE block,
 * channel multiply, generic Linear (matmul + bias).
 *
 * All AVX-512 paths fall back gracefully to scalar.
 */

#include "nn2_internal.h"
#include "nn2_antispoof_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif


/* ===== PReLU: per-channel parametric ReLU =====
 * f(x) = x          if x > 0
 *        alpha[c]*x if x <= 0
 * Tensor layout: CHW row-major.
 */
typedef struct { float* x; const float* alpha; int C, HW; } PreluCtx;

static void prelu_worker(void* arg, int start, int end)
{
    PreluCtx* p = (PreluCtx*)arg;
    int HW = p->HW;
    for (int c = start; c < end; c++) {
        float a = p->alpha[c];
        float* row = p->x + c * HW;
        int i = 0;
#ifdef __AVX512F__
        __m512 va = _mm512_set1_ps(a);
        __m512 vzero = _mm512_setzero_ps();
        for (; i + 16 <= HW; i += 16) {
            __m512 v = _mm512_loadu_ps(row + i);
            __mmask16 neg = _mm512_cmplt_ps_mask(v, vzero);
            __m512 out = _mm512_mask_mul_ps(v, neg, v, va);
            _mm512_storeu_ps(row + i, out);
        }
#endif
        for (; i < HW; i++) {
            float v = row[i];
            row[i] = v > 0.0f ? v : v * a;
        }
    }
}

void nn2_prelu(float* x, const float* alpha, int C, int HW)
{
    PreluCtx ctx = { x, alpha, C, HW };
    if (C >= 8) nn2_tp_parallel_for(prelu_worker, &ctx, C, 4);
    else prelu_worker(&ctx, 0, C);
}

/* 1D PReLU after Linear: simple in-place, no spatial dim. */
void nn2_prelu_1d(float* x, const float* alpha, int N)
{
    int i = 0;
#ifdef __AVX512F__
    __m512 vzero = _mm512_setzero_ps();
    for (; i + 16 <= N; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 a = _mm512_loadu_ps(alpha + i);
        __mmask16 neg = _mm512_cmplt_ps_mask(v, vzero);
        __m512 out = _mm512_mask_mul_ps(v, neg, v, a);
        _mm512_storeu_ps(x + i, out);
    }
#endif
    for (; i < N; i++) {
        float v = x[i];
        x[i] = v > 0.0f ? v : v * alpha[i];
    }
}

/* ===== Global Average Pool (CHW → C) ===== */
void nn2_global_avg_pool(const float* x, int C, int HW, float* out)
{
    float inv = 1.0f / (float)HW;
    for (int c = 0; c < C; c++) {
        const float* row = x + c * HW;
        int i = 0;
        float sum = 0.0f;
#ifdef __AVX512F__
        __m512 acc = _mm512_setzero_ps();
        for (; i + 16 <= HW; i += 16) {
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(row + i));
        }
        sum = _mm512_reduce_add_ps(acc);
#endif
        for (; i < HW; i++) sum += row[i];
        out[c] = sum * inv;
    }
}

/* ===== Channel multiply: out[c, hw] = x[c, hw] * scale[c] (in-place) ===== */
void nn2_channel_mul(float* x, const float* scale, int C, int HW)
{
    for (int c = 0; c < C; c++) {
        float s = scale[c];
        float* row = x + c * HW;
        int i = 0;
#ifdef __AVX512F__
        __m512 vs = _mm512_set1_ps(s);
        for (; i + 16 <= HW; i += 16) {
            __m512 v = _mm512_loadu_ps(row + i);
            _mm512_storeu_ps(row + i, _mm512_mul_ps(v, vs));
        }
#endif
        for (; i < HW; i++) row[i] *= s;
    }
}

/* ===== Add (in-place out += in) ===== */
void nn2_add_inplace(float* out, const float* in, int N)
{
    int i = 0;
#ifdef __AVX512F__
    for (; i + 16 <= N; i += 16) {
        __m512 a = _mm512_loadu_ps(out + i);
        __m512 b = _mm512_loadu_ps(in + i);
        _mm512_storeu_ps(out + i, _mm512_add_ps(a, b));
    }
#endif
    for (; i < N; i++) out[i] += in[i];
}

/* ===== Linear (FC): y = W * x + b, where W is [out, in] row-major ===== */
void nn2_linear(const float* x, const float* W, const float* b,
                int in_f, int out_f, float* y)
{
    for (int o = 0; o < out_f; o++) {
        float s = b ? b[o] : 0.0f;
        const float* row = W + o * in_f;
        int i = 0;
#ifdef __AVX512F__
        __m512 acc = _mm512_setzero_ps();
        for (; i + 16 <= in_f; i += 16) {
            __m512 vw = _mm512_loadu_ps(row + i);
            __m512 vx = _mm512_loadu_ps(x + i);
            acc = _mm512_fmadd_ps(vw, vx, acc);
        }
        s += _mm512_reduce_add_ps(acc);
#endif
        for (; i < in_f; i++) s += row[i] * x[i];
        y[o] = s;
    }
}

/* ===== Softmax over a small N (3 for antispoof) ===== */
void nn2_softmax(const float* x, int N, float* y)
{
    float m = x[0];
    for (int i = 1; i < N; i++) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i] - m);
        sum += y[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < N; i++) y[i] *= inv;
}
