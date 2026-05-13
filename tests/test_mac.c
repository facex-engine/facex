/*
 * test_mac.c — Apple Silicon smoke test.
 *
 * Validates the AArch64/scalar build of FaceX:
 *   1. Engine loads from data/edgeface_xs_fp32.bin (+ optional weights/yunet_fp32.bin).
 *   2. facex_embed produces finite, deterministic output.
 *   3. facex_similarity self-sim == 1.0; different-input sim < 0.999.
 *   4. Reports median embed latency over 50 iterations.
 *   5. (If detector weights present) runs end-to-end on tests/test_face_160.raw.
 *
 * Build: see Makefile target `mac-test`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "facex.h"

#ifdef FACEX_HAVE_SME
extern int facex_has_sme(void);
extern int facex_has_sme2(void);
#endif
#ifdef FACEX_HAVE_ACCELERATE
extern int facex_accelerate_enabled(void);
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

#define EMBED_ITERS 50

int main(int argc, char** argv) {
    const char* embed_w  = argc > 1 ? argv[1] : "data/edgeface_xs_fp32.bin";
    const char* detect_w = argc > 2 ? argv[2] : "weights/yunet_fp32.bin";

    printf("FaceX %s — macOS / arm64 smoke test\n", facex_version());
    printf("Embedder weights: %s\n", embed_w);

    /* Report compile-time + runtime backend status so the same binary
     * tells the user (and the test runner) which path it'll exercise. */
    printf("Backends compiled in:");
#ifdef FACEX_HAVE_ACCELERATE
    printf(" Accelerate");
#endif
#ifdef FACEX_HAVE_SME
    printf(" SME");
#endif
#ifdef FACEX_HAVE_COREML
    printf(" CoreML");
#endif
    printf(" NEON\n");

    printf("Backends active at runtime:");
#ifdef FACEX_HAVE_ACCELERATE
    if (facex_accelerate_enabled()) printf(" Accelerate(AMX)");
#endif
#ifdef FACEX_HAVE_SME
    if (facex_has_sme())  printf(" SME");
    if (facex_has_sme2()) printf(" SME2");
#endif
    printf(" NEON\n");

    /* Try to also load detector. */
    FILE* dw = fopen(detect_w, "rb");
    int have_detector = dw != NULL;
    if (dw) fclose(dw);

    FaceX* fx = facex_init(embed_w, have_detector ? detect_w : NULL, NULL);
    if (!fx) {
        fprintf(stderr, "FAIL: facex_init returned NULL\n");
        return 1;
    }
    printf("Engine ready (detector: %s)\n", have_detector ? "yes" : "no — embed-only");

    /* === 1. Embedding sanity =================================== */
    float input[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        input[i] = (float)(i % 256) / 128.0f - 1.0f;

    float emb[512], emb2[512];
    if (facex_embed(fx, input, emb) != 0) {
        fprintf(stderr, "FAIL: facex_embed returned non-zero\n");
        return 1;
    }
    int nan_count = 0;
    double sumsq = 0;
    for (int i = 0; i < 512; i++) {
        if (emb[i] != emb[i]) nan_count++;
        sumsq += (double)emb[i] * emb[i];
    }
    if (nan_count > 0) {
        fprintf(stderr, "FAIL: %d NaN values in embedding\n", nan_count);
        return 1;
    }
    if (sumsq < 0.01) {
        fprintf(stderr, "FAIL: embedding norm² = %.6f, output looks dead\n", sumsq);
        return 1;
    }
    printf("[ok] embed: 512-dim finite, ||emb||² = %.4f\n", sumsq);

    /* === 2. Determinism ======================================== */
    facex_embed(fx, input, emb2);
    double diff = 0;
    for (int i = 0; i < 512; i++) diff += (emb[i]-emb2[i])*(emb[i]-emb2[i]);
    diff = sqrt(diff);
    if (diff > 1e-6) {
        fprintf(stderr, "FAIL: non-deterministic, diff=%.3e\n", diff);
        return 1;
    }
    printf("[ok] determinism: same input → identical output (diff=%.1e)\n", diff);

    /* === 3. Self / cross similarity ============================ */
    float self_sim = facex_similarity(emb, emb2);
    if (self_sim < 0.9999f) {
        fprintf(stderr, "FAIL: self-similarity %.6f < 0.9999\n", self_sim);
        return 1;
    }
    float input2[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        input2[i] = (float)((i + 42) % 256) / 128.0f - 1.0f;
    float emb3[512];
    facex_embed(fx, input2, emb3);
    float cross_sim = facex_similarity(emb, emb3);
    if (cross_sim > 0.999f) {
        fprintf(stderr, "FAIL: different inputs gave sim=%.4f (>0.999)\n", cross_sim);
        return 1;
    }
    printf("[ok] similarity: self=%.4f  cross=%.4f\n", self_sim, cross_sim);

    /* === 4. Embed-only latency ================================= */
    /* Warmup */
    for (int i = 0; i < 5; i++) facex_embed(fx, input, emb);

    double samples[EMBED_ITERS];
    for (int i = 0; i < EMBED_ITERS; i++) {
        double t0 = now_ms();
        facex_embed(fx, input, emb);
        samples[i] = now_ms() - t0;
    }
    qsort(samples, EMBED_ITERS, sizeof(double), cmp_double);
    double median = samples[EMBED_ITERS / 2];
    double p99    = samples[(int)(EMBED_ITERS * 0.99)];
    double minv   = samples[0];
    printf("[ok] embed latency: min=%.2f ms  median=%.2f ms  p99=%.2f ms  (n=%d)\n",
           minv, median, p99, EMBED_ITERS);

    /* === 5. (Optional) end-to-end with detector ================ */
    if (have_detector) {
        FILE* f = fopen("tests/test_face_160.raw", "rb");
        if (!f) {
            printf("[skip] tests/test_face_160.raw not present — skipping e2e\n");
        } else {
            uint8_t img[160 * 160 * 3];
            size_t n = fread(img, 1, sizeof(img), f);
            fclose(f);
            if (n != sizeof(img)) {
                fprintf(stderr, "FAIL: short read on test_face_160.raw (%zu bytes)\n", n);
                return 1;
            }
            FaceXResult results[10];
            facex_set_score_threshold(fx, 0.5f);
            double t0 = now_ms();
            int nfaces = facex_detect(fx, img, 160, 160, results, 10);
            double dt = now_ms() - t0;
            if (nfaces < 0) {
                fprintf(stderr, "FAIL: facex_detect returned %d\n", nfaces);
                return 1;
            }
            printf("[ok] e2e: detected %d face(s) in %.2f ms\n", nfaces, dt);
            for (int i = 0; i < nfaces; i++) {
                printf("       #%d  bbox=[%.1f,%.1f → %.1f,%.1f]  score=%.3f\n",
                       i, results[i].x1, results[i].y1,
                       results[i].x2, results[i].y2, results[i].score);
            }
        }
    }

    facex_free(fx);
    printf("\nPASS: macOS arm64 smoke test\n");
    return 0;
}
