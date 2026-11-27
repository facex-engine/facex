/*
 * example_mfn.c — Minimal MobileFaceNet usage.
 *
 * Loads a FaceX EFM3 .bin, computes two embeddings on dummy inputs,
 * reports the cosine similarity. Replace the dummy inputs with real
 * 112x112 RGB face data (values normalized to [-1, 1], CHW layout) to
 * get actual face-verification scores.
 *
 * Build:  make mfn-example            (or)
 *         gcc -O3 -march=native -mfma -Iinclude \
 *             -o example_mfn examples/example_mfn.c src/facex_mfn.c -lm
 *
 * Run:    ./example_mfn weights/facex_standard.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/facex_mfn.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s weights.bin\n", argv[0]);
        return 1;
    }

    MfnEngine *engine = mfn_open(argv[1]);
    if (!engine) {
        fprintf(stderr, "Failed to load %s\n", argv[1]);
        return 1;
    }
    int D = mfn_embedding_dim(engine);
    fprintf(stderr, "Loaded %s, embedding_dim=%d\n", argv[1], D);

    /* Two dummy 112x112 RGB inputs, normalized to [-1, 1] (CHW). */
    static float img_a[3 * 112 * 112];
    static float img_b[3 * 112 * 112];
    for (int i = 0; i < 3 * 112 * 112; i++) {
        img_a[i] = (float)(i % 256) / 128.0f - 1.0f;
        img_b[i] = (float)((i + 17) % 256) / 128.0f - 1.0f;
    }

    float *emb_a = (float*)malloc(D * sizeof(float));
    float *emb_b = (float*)malloc(D * sizeof(float));

    if (!emb_a || !emb_b) {
        free(emb_a);
        free(emb_b);
        mfn_close(engine);
        return 1;
    }
    mfn_engine_forward(engine, img_a, emb_a);
    mfn_engine_forward(engine, img_b, emb_b);

    float sim = mfn_similarity(emb_a, emb_b, D);
    fprintf(stdout, "cosine similarity = %.4f\n", sim);

    free(emb_a); free(emb_b);
    mfn_close(engine);
    return 0;
}
