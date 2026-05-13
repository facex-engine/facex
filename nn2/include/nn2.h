/*
 * nn2 — YOLO inference engine. Pure C, zero dependencies.
 * AVX2/AVX-512 SIMD, lock-free threading, fused Conv+BN+SiLU.
 *
 * Usage:
 *   NN2* ctx = nn2_init("yolov8n.bin", 0);
 *   NN2Det dets[300];
 *   int n = nn2_detect(ctx, rgb, w, h, dets, 300);
 *   nn2_free(ctx);
 */

#ifndef NN2_H
#define NN2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NN2 NN2;

typedef struct {
    float x1, y1, x2, y2;   /* bbox in input pixel coords */
    float score;             /* confidence */
    int   cls;               /* class index */
} NN2Det;

/* Init engine. n_threads=0 → auto-detect. */
NN2* nn2_init(const char* weights_path, int n_threads);

/* Detect objects. rgb_hwc: width*height*3 uint8, HWC, BGR or RGB.
 * Returns number of detections (0..max_det). */
int nn2_detect(NN2* ctx, const uint8_t* rgb_hwc, int width, int height,
               NN2Det* out, int max_det);

void nn2_free(NN2* ctx);

/* Config */
void nn2_set_conf_threshold(NN2* ctx, float threshold);  /* default 0.25 */
void nn2_set_nms_threshold(NN2* ctx, float threshold);    /* default 0.45 */
void nn2_set_input_size(NN2* ctx, int size);              /* 320 or 640, default from weights */

const char* nn2_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NN2_H */
