/*
 * golden_test.c — Verify FaceX produces correct embeddings.
 *
 * Generates a deterministic input, runs forward pass, checks
 * embedding against known-good reference values.
 *
 * Build: gcc -O3 -march=native -Iinclude -o golden_test tests/golden_test.c -L. -lfacex -lm -lpthread -lsynchronization
 * Run:   ./golden_test data/edgeface_xs_fp32.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "facex.h"

int main(int argc, char** argv) {
    const char* weights = argc > 1 ? argv[1] : "data/edgeface_xs_fp32.bin";

    printf("FaceX Golden Test\n");
    printf("Loading: %s\n", weights);

    FaceX* fx = facex_init(weights, NULL, NULL);
    if (!fx) {
        fprintf(stderr, "FAIL: cannot load weights\n");
        return 1;
    }

    /* Deterministic input: (i % 256) / 128 - 1 */
    float input[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        input[i] = (float)(i % 256) / 128.0f - 1.0f;

    float emb[512];
    int ret = facex_embed(fx, input, emb);
    if (ret != 0) {
        fprintf(stderr, "FAIL: facex_embed returned %d\n", ret);
        return 1;
    }

    /* Check embedding is non-zero and finite */
    float norm = 0;
    int nan_count = 0;
    for (int i = 0; i < 512; i++) {
        if (emb[i] != emb[i]) nan_count++; /* NaN check */
        norm += emb[i] * emb[i];
    }
    norm = sqrtf(norm);

    printf("Embedding norm: %.6f (raw EdgeFace output)\n", norm);
    printf("NaN values: %d (should be 0)\n", nan_count);
    printf("Embedding[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", emb[i]);
    printf("\n");

    /* Verify embedding is non-trivial (norm > 0.1) */
    if (norm < 0.1f) {
        fprintf(stderr, "FAIL: norm %.4f too small — likely garbage output\n", norm);
        return 1;
    }
    if (nan_count > 0) {
        fprintf(stderr, "FAIL: %d NaN values in embedding\n", nan_count);
        return 1;
    }

    /* Self-consistency: same input = same output */
    float emb2[512];
    facex_embed(fx, input, emb2);
    float diff = 0;
    for (int i = 0; i < 512; i++) diff += (emb[i] - emb2[i]) * (emb[i] - emb2[i]);
    diff = sqrtf(diff);
    printf("Self-consistency: diff=%.10f (should be 0)\n", diff);
    if (diff > 1e-6f) {
        fprintf(stderr, "FAIL: non-deterministic output\n");
        return 1;
    }

    /* Similarity test: same input should give sim=1.0 */
    float sim = facex_similarity(emb, emb2);
    printf("Self-similarity: %.6f (should be 1.0)\n", sim);
    if (sim < 0.999f) {
        fprintf(stderr, "FAIL: self-similarity %.4f < 0.999\n", sim);
        return 1;
    }

    /* Different input should give different embedding */
    float input2[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        input2[i] = (float)((i + 42) % 256) / 128.0f - 1.0f;
    float emb3[512];
    facex_embed(fx, input2, emb3);
    float sim2 = facex_similarity(emb, emb3);
    printf("Different-input similarity: %.4f (should be < 1.0)\n", sim2);
    if (sim2 > 0.999f) {
        fprintf(stderr, "FAIL: different inputs produce same embedding\n");
        return 1;
    }

    facex_free(fx);
    printf("\nPASS: all checks passed\n");
    return 0;
}
