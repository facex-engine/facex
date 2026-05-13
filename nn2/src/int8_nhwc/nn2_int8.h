/*
 * nn2_int8.h — Full INT8 NHWC inference pipeline.
 *
 * Layout: activations are int8_t[H][W][C_pad16] (NHWC, C padded to 16).
 * VNNI GEMM: A[M,K4] (uint8) × B_packed[K4,N16] (int8 VNNI) → int32 → int8
 *
 * Key insight: NHWC makes K-dimension (channels) contiguous.
 * No transpose needed for VNNI — im2col output is naturally [spatial, K].
 *
 * Pipeline per conv:
 *   1. im2col from INT8 NHWC input → INT8 [spatial, K4] (contiguous)
 *   2. XOR trick: s8→u8 for VPDPBUSD's unsigned×signed requirement
 *   3. VNNI GEMM: u8 × s8 → s32 accumulators
 *   4. Dequant s32 → f32, apply bias + SiLU, requant → s8
 *   5. Store INT8 NHWC output
 */

#ifndef NN2_INT8_H
#define NN2_INT8_H

#include <stdint.h>
#include <stddef.h>

/* Pad channel count to multiple of 16 for AVX-512 alignment */
#define C_PAD16(c) (((c) + 15) & ~15)
/* Pad K to multiple of 4 for VNNI (4 bytes per VPDPBUSD group) */
#define K_PAD4(k) (((k) + 3) & ~3)

/* Per-layer quantization parameters */
typedef struct {
    float scale;       /* activation scale: float_val = int8_val * scale */
    float inv_scale;   /* 1.0 / scale for quantization */
} QuantScale;

/* INT8 NHWC convolution layer */
typedef struct {
    int8_t*  w_packed;    /* VNNI-packed: [Cout_pad/16][K4/4][64] */
    int32_t* col_sums;    /* [Cout] = 128 * sum(w[:,n]) for u8 offset compensation */
    float*   w_scales;    /* [Cout] per-channel weight scale */
    float*   bias;        /* [Cout] FP32 bias */
    float    act_scale;   /* input activation scale */
    float    out_scale;   /* output activation scale */
    int cout, cin, k, stride, pad;
    int act;              /* 0=none, 1=SiLU, 2=sigmoid */
    int K4;               /* K_PAD4(cin * k * k) */
    int cout_pad;         /* C_PAD16(cout) */
} I8Conv;

/* INT8 NHWC C2f block */
typedef struct {
    I8Conv cv1;
    I8Conv cv2;
    struct { I8Conv cv1, cv2; } m[3];
    int n, c, shortcut;
} I8C2f;

/* Full INT8 model */
typedef struct {
    /* Backbone */
    I8Conv bb_conv[5];
    I8C2f  bb_c2f[4];
    I8Conv sppf_cv1, sppf_cv2;
    /* Neck */
    I8C2f  nk_c2f[4];
    I8Conv nk_conv[2];
    /* Head (stays FP32 for decode accuracy) */
    /* ... head uses existing FP32 path */

    int num_classes;
    int input_size;

    /* Workspace */
    int8_t* ws_int8;     /* INT8 workspace for activations */
    float*  ws_fp32;     /* FP32 workspace for head + decode */
    size_t  ws_int8_size;
    size_t  ws_fp32_size;
} I8Model;

/* ============ VNNI GEMM ============ */

/* INT8 VNNI GEMM: C_int32[M,N] = A_u8[M,K4] × B_packed[K4,N]
 * Then: dequant + bias + SiLU + requant → int8 output.
 * A is s8 input XOR'd to u8 (+128 offset).
 * B_packed: [N/16][K4/4][64 bytes] VNNI format.
 * col_sums: [N] = 128 * sum(B[:,n]) for compensation. */
void i8_gemm_bias_act(
    const int8_t* A, int M, int K4, int N,
    const int8_t* B_packed,
    const int32_t* col_sums,
    const float* w_scales,    /* [N] per-channel */
    const float* bias,        /* [N] */
    float act_scale,          /* input scale */
    float out_scale,          /* output scale */
    int act,                  /* 0=none, 1=SiLU */
    int8_t* output,           /* [M, N_pad16] INT8 NHWC */
    int out_stride);          /* stride between output rows (= N_pad16) */

/* ============ Weight packing ============ */

/* Pack FP32 weights to INT8 VNNI format.
 * weights: [Cout, K] FP32 row-major
 * Returns: per-channel scales in w_scales[Cout] */
void i8_pack_weights(const float* weights, int Cout, int K,
                     int8_t* packed, int32_t* col_sums,
                     float* w_scales);

/* ============ INT8 NHWC Conv ============ */

/* 1x1 conv: direct GEMM, no im2col */
void i8_conv1x1(const I8Conv* L, const int8_t* input, int H, int W,
                int8_t* output);

/* 3x3 conv: im2col + GEMM */
void i8_conv3x3(const I8Conv* L, const int8_t* input, int H, int W,
                int8_t* output, int8_t* im2col_buf);

/* ============ INT8 NHWC Ops ============ */

/* Quantize FP32 CHW → INT8 NHWC */
void i8_quantize_nhwc(const float* fp32_chw, int C, int H, int W,
                       float scale, int8_t* out_nhwc, int C_pad);

/* Dequantize INT8 NHWC → FP32 CHW */
void i8_dequant_to_chw(const int8_t* int8_nhwc, int C, int H, int W,
                        int C_pad, float scale, float* fp32_chw);

/* Nearest-neighbor upsample 2× in NHWC (trivial: duplicate HW pixels) */
void i8_upsample_2x(const int8_t* in, int C_pad, int H, int W, int8_t* out);

/* Concat in NHWC: [H,W,Ca] + [H,W,Cb] → [H,W,Ca+Cb] */
void i8_concat(const int8_t* a, int Ca_pad, const int8_t* b, int Cb_pad,
               int H, int W, int8_t* out, int Cout_pad,
               float scale_a, float scale_b, float scale_out);

/* MaxPool 5×5 s=1 p=2 in INT8 NHWC */
void i8_maxpool(const int8_t* in, int C_pad, int H, int W,
                int k, int stride, int pad, int8_t* out);

#endif /* NN2_INT8_H */
