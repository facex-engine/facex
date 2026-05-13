/*
 * FaceX example — compute face embedding and compare two faces.
 *
 * Build: gcc -O3 -Iinclude -o example examples/example.c -L. -lfacex -lm -lpthread
 * Run:   ./example data/edgeface_xs_fp32.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include "facex.h"

int main(int argc, char** argv) {
    const char* weights = argc > 1 ? argv[1] : "data/edgeface_xs_fp32.bin";

    printf("FaceX %s\n", facex_version());

    /* Initialize engine */
    FaceX* fx = facex_init(weights, NULL, NULL);
    if (!fx) {
        fprintf(stderr, "Failed to load weights: %s\n", weights);
        return 1;
    }
    printf("Engine ready.\n");

    /* Create two dummy faces (in real use: load 112x112 RGB image) */
    float face1[112 * 112 * 3];
    float face2[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++) {
        face1[i] = (float)(i % 256) / 128.0f - 1.0f;
        face2[i] = (float)((i + 42) % 256) / 128.0f - 1.0f;
    }

    /* Compute embeddings */
    float emb1[512], emb2[512];
    facex_embed(fx, face1, emb1);
    facex_embed(fx, face2, emb2);

    /* Compare */
    float sim = facex_similarity(emb1, emb2);
    printf("Similarity: %.4f\n", sim);
    printf("Same person: %s\n", sim > 0.3f ? "YES" : "NO");

    facex_free(fx);
    return 0;
}
