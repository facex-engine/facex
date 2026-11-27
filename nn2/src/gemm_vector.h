#ifndef NN2_GEMM_VECTOR_H
#define NN2_GEMM_VECTOR_H
#ifdef __AVX512F__
#include "fast_exp.h"

static inline __m512 nn2_load_columns(const float *row, int columns, int offset) {
    int count = columns - offset;
    if (count <= 0) return _mm512_setzero_ps();
    if (count >= 16) return _mm512_loadu_ps(row + offset);
    return _mm512_maskz_loadu_ps((__mmask16)((1u << count) - 1), row + offset);
}

static inline __m512 nn2_activate_vector(__m512 x, int act) {
    if (act == 1) return _mm512_silu_fast(x);
    if (act == 2) return _mm512_sigmoid_fast(x);
    if (act == 3) {
        __mmask16 negative = _mm512_cmplt_ps_mask(x, _mm512_setzero_ps());
        return _mm512_mask_mul_ps(x, negative, x, _mm512_set1_ps(0.1f));
    }
    return x;
}
#endif
#endif
