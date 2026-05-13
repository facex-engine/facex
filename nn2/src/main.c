/*
 * main.c — CLI entry point for nn2 YOLO inference.
 *
 * Usage: nn2 <weights.bin> <image.raw> <width> <height>
 * Image format: raw RGB uint8, HWC layout.
 *
 * For real use, integrate stb_image.h or feed from video decoder.
 */

#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <weights.bin> <image.raw> <width> <height> [threads]\n", argv[0]);
        fprintf(stderr, "  image.raw: raw RGB uint8, HWC layout, width*height*3 bytes\n");
        return 1;
    }

    const char* weights_path = argv[1];
    const char* image_path = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int n_threads = (argc > 5) ? atoi(argv[5]) : 0;

    /* Load image */
    FILE* f = fopen(image_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", image_path); return 1; }
    size_t img_size = (size_t)width * height * 3;
    uint8_t* img = (uint8_t*)malloc(img_size);
    fread(img, 1, img_size, f);
    fclose(f);

    /* Init */
    NN2* ctx = nn2_init(weights_path, n_threads);
    if (!ctx) { fprintf(stderr, "failed to init nn2\n"); free(img); return 1; }

    /* Warmup */
    NN2Det dets[300];
    nn2_detect(ctx, img, width, height, dets, 300);

    /* Benchmark */
    int N_RUNS = 50;
    double t0 = get_time_ms();
    int n_det = 0;
    for (int i = 0; i < N_RUNS; i++)
        n_det = nn2_detect(ctx, img, width, height, dets, 300);
    double t1 = get_time_ms();

    printf("nn2 v%s | %dx%d → %d input | %d threads\n",
           nn2_version(), width, height, ctx->input_size,
           nn2_tp_num_threads());
    printf("avg %.1f ms (%.1f FPS) over %d runs | %d dispatches/inference\n",
           (t1 - t0) / N_RUNS, N_RUNS / ((t1 - t0) / 1000.0), N_RUNS,
           nn2_tp_dispatch_count() / N_RUNS);
    nn2_tp_reset_dispatch_count();
    printf("%d detections:\n", n_det);
    for (int i = 0; i < n_det; i++) {
        printf("  [%d] cls=%d score=%.3f box=(%.1f, %.1f, %.1f, %.1f)\n",
               i, dets[i].cls, dets[i].score,
               dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }

    nn2_free(ctx);
    free(img);
    return 0;
}
