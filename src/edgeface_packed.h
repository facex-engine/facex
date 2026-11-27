#ifndef FACEX_EDGEFACE_PACKED_H
#define FACEX_EDGEFACE_PACKED_H
#include <stdint.h>

typedef struct {
    int8_t *input;
    int32_t *output;
} QuantScratch;

typedef struct {
    void *packed;
    int32_t *col_sums;
    float *w_scales;
} PackedMM;

typedef struct {
    float *data;
    int K, N;
    PackedMM quant;
    QuantScratch *scratch;
} PackedFP32;

#ifndef FACEX_FP32_ONLY
void matmul_dynamic_int8(const float *, int, int, int, const void *, const int32_t *,
                         const float *, float *, QuantScratch *);
#endif
#endif
