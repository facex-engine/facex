/*
 * facex_v2.h — Parametric EdgeFace engine API.
 *
 * Loads EFX2 .bin files (binary header describes architecture) and runs
 * the forward pass dynamically. Supports the 4 FaceX size variants
 * (Nano/Tiny/Standard/XS) with one engine binary.
 *
 * For the legacy EFXS (XS-only) format, use the v1 facex.h API.
 */
#ifndef FACEX_V2_H
#define FACEX_V2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle (defined in edgeface_engine_v2.c as V2_Engine). */
typedef struct V2_Engine V2_Engine;

/*
 * Initialize from EFX2 weights file.
 *   path: path to .bin
 *   out: zero-initialized V2_Engine struct (caller-allocated)
 *   Returns 0 on success, negative on error.
 */
int v2_engine_init(const char *path, V2_Engine *out);

/*
 * Forward pass on a 3x112x112 CHW float32 input (values in [-1,1]).
 *   input_chw: [3 * 112 * 112] floats
 *   embedding: output buffer, [embedding_dim] floats (L2-normalized).
 *
 * Use v2_embedding_dim(engine) to size the buffer.
 */
void v2_engine_forward(const V2_Engine *engine, const float *input_chw, float *embedding);

/*
 * Embedding dimension declared in the loaded .bin header.
 */
int v2_embedding_dim(const V2_Engine *engine);

/*
 * Free engine resources (raw buffer, packed weights, workspace).
 */
void v2_engine_free(V2_Engine *engine);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_V2_H */
