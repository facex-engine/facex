/*
 * main_annotate.c — Detect objects and save annotated image with bboxes.
 * Usage: nn2_annotate <weights.bin> <input.jpg> <output.jpg> [threads]
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "nn2.h"
#include "draw.h"
#include <stdio.h>
#include <stdlib.h>

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
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <weights.bin> <input.jpg> <output.jpg> [threads]\n", argv[0]);
        return 1;
    }

    int threads = (argc > 4) ? atoi(argv[4]) : 0;

    /* Load image */
    int w, h, channels;
    uint8_t* img = stbi_load(argv[2], &w, &h, &channels, 3);
    if (!img) { fprintf(stderr, "Failed to load %s\n", argv[2]); return 1; }
    printf("Input: %s (%dx%d)\n", argv[2], w, h);

    /* Init + detect */
    NN2* ctx = nn2_init(argv[1], threads);
    if (!ctx) { stbi_image_free(img); return 1; }

    NN2Det dets[300];
    int n = nn2_detect(ctx, img, w, h, dets, 300);
    printf("%d detections:\n", n);
    for (int i = 0; i < n; i++) {
        const char* name = (dets[i].cls >= 0 && dets[i].cls < 80) ? coco_names[dets[i].cls] : "?";
        printf("  [%d] %s %.0f%% (%.0f,%.0f)-(%.0f,%.0f)\n",
               i, name, dets[i].score*100,
               dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }

    /* Draw bboxes */
    nn2_draw_detections(img, w, h, dets, n);

    /* Save */
    const char* out_path = argv[3];
    int ok = 0;
    if (strstr(out_path, ".png"))
        ok = stbi_write_png(out_path, w, h, 3, img, w * 3);
    else
        ok = stbi_write_jpg(out_path, w, h, 3, img, 95);

    if (ok) printf("Saved: %s\n", out_path);
    else fprintf(stderr, "Failed to save %s\n", out_path);

    nn2_free(ctx);
    stbi_image_free(img);
    return 0;
}
