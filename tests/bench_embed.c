/*
 * bench_embed.c — Benchmark FaceX embedding speed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "facex.h"
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart * 1000.0;
}
#endif

int main(int argc, char** argv) {
    const char* weights = argc > 1 ? argv[1] : "data/edgeface_xs_fp32.bin";
    int loops = argc > 2 ? atoi(argv[2]) : 100;

    printf("FaceX Embedding Benchmark\n");
    FaceX* fx = facex_init(weights, NULL, NULL);
    if (!fx) { fprintf(stderr, "Cannot load weights\n"); return 1; }

    float input[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        input[i] = (float)(i % 256) / 128.0f - 1.0f;
    float emb[512];

    /* Warmup */
    for (int i = 0; i < 5; i++) facex_embed(fx, input, emb);

    /* Benchmark */
    double best = 1e9, total = 0;
    for (int i = 0; i < loops; i++) {
        double t0 = now_ms();
        facex_embed(fx, input, emb);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        total += dt;
    }

    printf("  loops:  %d\n", loops);
    printf("  best:   %.2f ms\n", best);
    printf("  avg:    %.2f ms\n", total / loops);
    printf("  fps:    %.0f\n", 1000.0 / (total / loops));

    facex_free(fx);
    return 0;
}
