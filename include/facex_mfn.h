/*
 * facex_mfn.h — Parametric MobileFaceNet embedding engine.
 *
 * Loads EFM3 .bin files (4 size variants: Nano / Tiny / Standard / XS)
 * and runs the forward pass. Reads architecture from a binary header
 * so one engine binary handles all sizes.
 *
 * BN is folded into the preceding convolution at load time, so
 * runtime is plain Conv + Bias + (optional) PReLU.
 */
#ifndef FACEX_MFN_H
#define FACEX_MFN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle (defined in facex_mfn.c). */
typedef struct MfnEngine MfnEngine;

/*
 * Initialize from EFM3 weights file.
 *   path: path to .bin
 *   out:  caller-allocated MfnEngine, zero-initialized
 *   Returns 0 on success, negative on error.
 */
int mfn_engine_init(const char *path, MfnEngine *out);

/*
 * Forward pass.
 *   input_chw: [3 * 112 * 112] FP32, values in [-1, 1]
 *   embedding: output buffer, [embedding_dim] FP32 (L2-normalized).
 * Use mfn_embedding_dim() to size the buffer.
 */
void mfn_engine_forward(const MfnEngine *engine,
                        const float *input_chw, float *embedding);

/*
 * Embedding dim declared in the loaded .bin.
 */
int mfn_embedding_dim(const MfnEngine *engine);

/*
 * Cosine similarity between two embeddings, [-1, 1].
 * > 0.3 typically = same person.
 */
float mfn_similarity(const float *a, const float *b, int dim);

/*
 * Free engine resources.
 */
void mfn_engine_free(MfnEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_MFN_H */
