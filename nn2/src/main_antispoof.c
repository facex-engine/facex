/*
 * main_antispoof.c — CLI for the MiniFASNet port:
 *   nn2_antispoof_test <v2|v1se> <weights.bin> <80x80_bgr.bin> [iters]
 *
 * Reads a raw 80×80 BGR uint8 image (HWC, 19200 bytes) and runs N forward
 * passes, prints predicted probabilities + average latency.
 */

#include "nn2_antispoof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double now_sec(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <v2|v1se> <weights.bin> <80x80_bgr.bin> [iters]\n", argv[0]);
        return 1;
    }
    AntiSpoofVariant var = (strcmp(argv[1], "v2") == 0) ? NN2_AS_V2 : NN2_AS_V1SE;
    int iters = (argc > 4) ? atoi(argv[4]) : 100;

    AntiSpoofModel* m = nn2_antispoof_load(argv[2], var, 0);
    if (!m) { fprintf(stderr, "load failed\n"); return 2; }

    unsigned char bgr[80 * 80 * 3];
    FILE* f = fopen(argv[3], "rb");
    if (!f || fread(bgr, 1, sizeof(bgr), f) != sizeof(bgr)) {
        fprintf(stderr, "cannot read 19200 bytes from %s\n", argv[3]);
        if (f) fclose(f);
        return 3;
    }
    fclose(f);

    float probs[3];
    /* Warm-up */
    for (int i = 0; i < 5; i++) nn2_antispoof_predict(m, bgr, probs);

    /* Timed run */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) nn2_antispoof_predict(m, bgr, probs);
    double t1 = now_sec();
    double ms = (t1 - t0) * 1000.0 / iters;

    printf("variant=%s probs=[%.6f, %.6f, %.6f]  P(live)=%.4f  %.3f ms/iter  (%d iters)\n",
           argv[1], probs[0], probs[1], probs[2], probs[1], ms, iters);

    nn2_antispoof_free(m);
    return 0;
}
