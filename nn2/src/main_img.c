/*
 * main_img.c — nn2 CLI with real image loading via stb_image.
 * Usage: nn2_img <weights.bin> <image.jpg/png> [threads]
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "nn2.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart * 1000.0;
}
#endif

static const char* coco_names[] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <weights.bin> <image.jpg/png> [threads]\n", argv[0]);
        return 1;
    }

    int n_threads = (argc > 3) ? atoi(argv[3]) : 0;

    /* Load image */
    int w, h, channels;
    uint8_t* img = stbi_load(argv[2], &w, &h, &channels, 3);
    if (!img) { fprintf(stderr, "Failed to load %s\n", argv[2]); return 1; }
    printf("Image: %s (%dx%d)\n", argv[2], w, h);

    /* Init */
    NN2* ctx = nn2_init(argv[1], n_threads);
    if (!ctx) { stbi_image_free(img); return 1; }

    /* Detect */
    NN2Det dets[300];
    double t0 = get_time_ms();
    int n = nn2_detect(ctx, img, w, h, dets, 300);
    double ms = get_time_ms() - t0;

    printf("\n%d detections in %.1f ms:\n", n, ms);
    for (int i = 0; i < n; i++) {
        const char* name = (dets[i].cls >= 0 && dets[i].cls < 80) ? coco_names[dets[i].cls] : "?";
        printf("  [%d] %s (%.1f%%) @ (%.0f,%.0f)-(%.0f,%.0f)\n",
               i, name, dets[i].score * 100,
               dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }

    /* Benchmark */
    nn2_detect(ctx, img, w, h, dets, 300); /* warmup */
    int N = 50;
    t0 = get_time_ms();
    for (int i = 0; i < N; i++)
        nn2_detect(ctx, img, w, h, dets, 300);
    ms = (get_time_ms() - t0) / N;
    printf("\nBenchmark: %.1f ms (%.0f FPS) over %d runs\n", ms, 1000.0/ms, N);

    nn2_free(ctx);
    stbi_image_free(img);
    return 0;
}
