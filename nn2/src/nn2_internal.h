/*
 * nn2_internal.h — Internal types and function declarations.
 */

#ifndef NN2_INTERNAL_H
#define NN2_INTERNAL_H

#include "nn2.h"
#include <stddef.h>

/* ============ GEMM config ============ */
#define NN2_MR 6
#define NN2_NR 16

/* ============ YOLO config ============ */
#define NN2_REG_MAX     16
#define NN2_MAX_ANCHORS 8400  /* 80*80 + 40*40 + 20*20 at 640 */
#define NN2_MAX_DET     300

/* ============ Convolution layer (fused Conv+BN) ============ */
typedef struct {
    float* w;         /* [cout, cin*k*k] FP32 row-major */
    float* w_packed;  /* A-packed for GEMM: [ceil(cout/MR)][K][MR] or NULL */
    float* b;         /* [cout] fused bias */
    float* w_wino;    /* [16][cout][cin] Winograd-transformed, or NULL */
    /* INT8 quantized path */
    void*    w_int8_packed; /* packed INT8 weights (VNNI format), or NULL */
    int32_t* col_sums;     /* [cout] compensation: 128 * sum(w) */
    float*   w_scales;     /* [cout] per-channel weight scale */
    float    act_in_scale;  /* input activation scale */
    float    act_out_scale; /* output activation scale */
    int    cout, cin, k, stride, pad;
    int    act;     /* 0=none, 1=SiLU, 2=sigmoid */
} ConvLayer;

/* ============ C2f block ============ */
typedef struct {
    ConvLayer cv1;      /* 1x1: in → 2*c */
    ConvLayer cv2;      /* 1x1: (2+n)*c → out */
    struct {
        ConvLayer cv1;  /* 3x3: c → c */
        ConvLayer cv2;  /* 3x3: c → c */
    } m[3];             /* max 3 bottlenecks */
    int n;              /* actual bottleneck count */
    int c;              /* hidden channels = out/2 */
    int shortcut;       /* residual in bottleneck */
} C2fBlock;

/* ============ Full model ============ */
struct NN2 {
    /* Backbone */
    ConvLayer bb_conv[5];   /* 0: 3→16, 1: 16→32, 2: 32→64, 3: 64→128, 4: 128→256 */
    C2fBlock  bb_c2f[4];    /* 0: 32→32(n=1), 1: 64→64(n=2), 2: 128→128(n=2), 3: 256→256(n=1) */
    ConvLayer sppf_cv1;     /* 256→128 */
    ConvLayer sppf_cv2;     /* 512→256 */

    /* Neck */
    C2fBlock  nk_c2f[4];   /* 0: 384→128, 1: 192→64, 2: 192→128, 3: 384→256 */
    ConvLayer nk_conv[2];   /* 0: 64→64 s=2, 1: 128→128 s=2 */

    /* Head (3 scales) */
    struct {
        ConvLayer bbox[3];  /* conv3x3, conv3x3, conv1x1(→64) */
        ConvLayer cls[3];   /* conv3x3, conv3x3, conv1x1(→nc) */
    } head[3];

    /* DFL weight: softmax over reg_max bins → weighted sum */
    float dfl_weight[NN2_REG_MAX]; /* [0, 1, 2, ..., 15] */

    /* Config */
    int   num_classes;
    int   input_size;    /* 320 or 640 */
    float conf_thresh;
    float nms_thresh;

    /* Workspace */
    float* workspace;     /* big slab for all temporaries */
    size_t ws_size;
    int    n_threads;
    int    _fused_norm;   /* 1 if normalization fused into first conv weights */
};

/* ============ GEMM ============ */
void nn2_sgemm(int M, int K, int N,
               const float* A, int lda,
               const float* B, int ldb,
               float* C, int ldc);

/* Pre-pack weight matrix A for faster GEMM */
float* nn2_pack_weight_a(const float* A, int M, int K);

/* Fused GEMM: C = A×B + bias + activation, in one memory pass */
void nn2_sgemm_bias_act(int M, int K, int N,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float* C, int ldc,
                        const float* bias, int act);
/* Fused GEMM with pre-packed A (sequential weight access) */
void nn2_sgemm_bias_act_packed(int M, int K, int N,
                               const float* Ap, const float* B,
                               float* C, int ldc,
                               const float* bias, int act);
/* Fused GEMM + bias + act + residual add (saves 2 memory passes) */
void nn2_sgemm_bias_act_res_packed(int M, int K, int N,
                                    const float* Ap, const float* B,
                                    float* C, int ldc,
                                    const float* bias, int act,
                                    const float* residual, int ldr);

/* ============ Conv ============ */
void nn2_im2col(const float* in, int C, int H, int W,
                int kH, int kW, int stride, int pad,
                float* col);

/* Conv + bias + activation. out must be pre-allocated [cout, Hout*Wout]. */
void nn2_conv2d(const ConvLayer* layer, const float* in, int H, int W,
                float* out, float* col_buf);

/* ============ Ops ============ */
void nn2_silu_inplace(float* x, int n);
void nn2_sigmoid_inplace(float* x, int n);
void nn2_upsample_2x(const float* in, int C, int H, int W, float* out);
void nn2_concat(const float* a, int ca, const float* b, int cb,
                int H, int W, float* out);
void nn2_maxpool2d(const float* in, int C, int H, int W,
                   int k, int stride, int pad, float* out);
void nn2_dwconv2d(const float* in, const float* w, const float* bias,
                  int C, int H, int W, int k, int stride, int pad,
                  int act, float* out); /* depthwise conv: group=C */
void nn2_channel_shuffle(const float* in, int C, int H, int W,
                         int groups, float* out);

/* ============ Decode ============ */
int nn2_decode(const float* bbox_feat, const float* cls_feat,
               int num_classes, int input_size, float conf_thresh,
               const float* dfl_w, NN2Det* dets, int max_det);

int nn2_nms(NN2Det* dets, int n, float iou_thresh);

/* ============ Thread pool ============ */
typedef void (*nn2_task_fn)(void* ctx, int start, int end);
void nn2_tp_init(int n_threads);
void nn2_tp_parallel_for(nn2_task_fn fn, void* ctx, int total, int grain);
void nn2_tp_destroy(void);
int  nn2_tp_num_threads(void);
int  nn2_tp_dispatch_count(void);
void nn2_tp_reset_dispatch_count(void);

/* ============ Winograd ============ */
void nn2_prepare_winograd(ConvLayer* L);
void nn2_winograd_transform_weights(const float* W, int Cout, int Cin, float* U);
void nn2_winograd_conv3x3(const float* input, int Cin, int H, int W,
                          const float* U, const float* bias, int Cout, int act,
                          float* output, float* workspace);

/* ============ INT8 GEMM ============ */
void nn2_int8_pack_weights(const int8_t* w, int K, int Cout, void* out, int32_t* col_sums);
int  nn2_int8_packed_size(int K, int Cout);
void nn2_int8_gemm_fused(const int8_t* A, int M, int K, int N,
                          const void* B_packed, const int32_t* col_sums,
                          const float* w_scales, const float* bias,
                          float act_scale, float out_scale, int act_type,
                          float* output);

/* ============ Implicit im2col conv (no buffer, loads from input directly) ============ */
void nn2_conv3x3_s1_implicit(const float* input, const float* weight, const float* bias,
                              int Cin, int Cout, int H, int W, int act, float* output,
                              const float* w_packed);
void nn2_conv3x3_s2_implicit(const float* input, const float* weight, const float* bias,
                              int Cin, int Cout, int H, int W, int act, float* output);

/* ============ Cross-layer fused bottleneck ============ */
void nn2_bottleneck_fused(const float* input, int Cin, int H, int W,
                           const ConvLayer* cv1, const ConvLayer* cv2,
                           const float* residual, float* output, float* line_buf);

/* ============ Fused conv (no im2col buffer needed) ============ */
void nn2_conv3x3_s1_fused(const ConvLayer* L, const float* in, int H, int W, float* out);
void nn2_conv3x3_s2_fused(const ConvLayer* L, const float* in, int H, int W, float* out);

/* ============ Net forward ============ */
int nn2_forward(NN2* ctx, const float* input, float* bbox_out, float* cls_out);

#endif /* NN2_INTERNAL_H */
