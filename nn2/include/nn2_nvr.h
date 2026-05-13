/*
 * nn2_nvr.h — Multi-camera NVR pipeline.
 *
 * Manages N camera feeds, motion-gated detection, event callbacks.
 * Single nn2 model shared across all cameras.
 *
 * Usage:
 *   NN2NVR* nvr = nn2_nvr_create("weights/yolov8n_320.bin", 6);
 *   nn2_nvr_add_camera(nvr, 0, "Camera-1");
 *   nn2_nvr_add_camera(nvr, 1, "Camera-2");
 *   nn2_nvr_set_callback(nvr, on_detection);
 *   // In frame loop:
 *   nn2_nvr_push_frame(nvr, cam_id, rgb, w, h);
 *   nn2_nvr_free(nvr);
 */

#ifndef NN2_NVR_H
#define NN2_NVR_H

#include "nn2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NN2_NVR_MAX_CAMERAS 64

typedef struct {
    int cam_id;
    int det_count;
    NN2Det dets[50];
    double timestamp_ms;
    double inference_ms;
} NN2NVREvent;

typedef void (*nn2_nvr_callback)(const NN2NVREvent* event, void* userdata);

typedef struct NN2NVR NN2NVR;

/* Create NVR pipeline with shared model */
NN2NVR* nn2_nvr_create(const char* weights_path, int n_threads);

/* Add a camera feed */
int nn2_nvr_add_camera(NN2NVR* nvr, int cam_id, const char* name);

/* Set detection callback */
void nn2_nvr_set_callback(NN2NVR* nvr, nn2_nvr_callback cb, void* userdata);

/* Configure */
void nn2_nvr_set_skip_frames(NN2NVR* nvr, int skip);       /* default 5 */
void nn2_nvr_set_motion_threshold(NN2NVR* nvr, float thresh); /* 0=disabled */
void nn2_nvr_set_conf_threshold(NN2NVR* nvr, float thresh);

/* Push a frame from camera. Returns 1 if detection was run, 0 if skipped. */
int nn2_nvr_push_frame(NN2NVR* nvr, int cam_id,
                        const uint8_t* rgb_hwc, int width, int height);

/* Get stats */
void nn2_nvr_stats(const NN2NVR* nvr, int* total_frames, int* total_detections,
                   double* avg_inference_ms);

void nn2_nvr_free(NN2NVR* nvr);

#ifdef __cplusplus
}
#endif

#endif /* NN2_NVR_H */
