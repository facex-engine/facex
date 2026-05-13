/*
 * nn2_antispoof_ops.h — Internal ops added for MiniFASNet.
 * Not exposed in the public header.
 */

#ifndef NN2_ANTISPOOF_OPS_H
#define NN2_ANTISPOOF_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/* PReLU on a (C, H*W) tensor, per-channel alpha. */
void nn2_prelu(float* x, const float* alpha, int C, int HW);

/* 1D PReLU after Linear: per-element alpha[N]. */
void nn2_prelu_1d(float* x, const float* alpha, int N);

/* Global avg pool: (C, H*W) → (C,) — mean over spatial. */
void nn2_global_avg_pool(const float* x, int C, int HW, float* out);

/* Per-channel scale: x[c, :] *= scale[c] in place. */
void nn2_channel_mul(float* x, const float* scale, int C, int HW);

/* In-place elementwise add: out[i] += in[i]. */
void nn2_add_inplace(float* out, const float* in, int N);

/* Dense matmul: y = W * x + b, W is [out_f, in_f] row-major. */
void nn2_linear(const float* x, const float* W, const float* b,
                int in_f, int out_f, float* y);

/* Softmax over N (typically 3 for antispoof). */
void nn2_softmax(const float* x, int N, float* y);

#ifdef __cplusplus
}
#endif

#endif /* NN2_ANTISPOOF_OPS_H */
