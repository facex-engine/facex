/*
 * nvr.c — Multi-camera NVR pipeline with motion gating.
 *
 * Architecture:
 *   - Single NN2 model shared across all cameras
 *   - Round-robin frame processing with skip-N
 *   - Simple motion detection: frame difference > threshold
 *   - Callback on detection events
 */

#include "nn2_nvr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double get_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart * 1000.0;
}
#else
#include <time.h>
static double get_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

typedef struct {
    int id;
    char name[64];
    int frame_count;
    int det_count;
    /* Previous frame for motion detection (downsampled grayscale) */
    uint8_t* prev_gray;  /* 80×60 downsampled */
    int prev_w, prev_h;
    int active;
} CameraState;

struct NN2NVR {
    NN2* model;
    CameraState cameras[NN2_NVR_MAX_CAMERAS];
    int n_cameras;
    int skip_frames;       /* process every N-th frame */
    float motion_thresh;   /* 0 = disabled */
    float conf_thresh;
    nn2_nvr_callback callback;
    void* cb_userdata;
    /* Stats */
    int total_frames;
    int total_inferences;
    int total_detections;
    double total_inference_ms;
};

NN2NVR* nn2_nvr_create(const char* weights_path, int n_threads)
{
    NN2NVR* nvr = (NN2NVR*)calloc(1, sizeof(NN2NVR));
    if (!nvr) return NULL;

    nvr->model = nn2_init(weights_path, n_threads);
    if (!nvr->model) { free(nvr); return NULL; }

    nvr->skip_frames = 5;
    nvr->motion_thresh = 0.0f; /* disabled by default */
    nvr->conf_thresh = 0.25f;
    nn2_set_conf_threshold(nvr->model, nvr->conf_thresh);

    return nvr;
}

int nn2_nvr_add_camera(NN2NVR* nvr, int cam_id, const char* name)
{
    if (nvr->n_cameras >= NN2_NVR_MAX_CAMERAS) return -1;

    CameraState* cam = &nvr->cameras[nvr->n_cameras++];
    cam->id = cam_id;
    strncpy(cam->name, name ? name : "cam", sizeof(cam->name) - 1);
    cam->active = 1;
    cam->prev_gray = NULL;
    cam->prev_w = 80;
    cam->prev_h = 60;

    return 0;
}

void nn2_nvr_set_callback(NN2NVR* nvr, nn2_nvr_callback cb, void* userdata)
{
    nvr->callback = cb;
    nvr->cb_userdata = userdata;
}

void nn2_nvr_set_skip_frames(NN2NVR* nvr, int skip)
{
    nvr->skip_frames = (skip > 0) ? skip : 1;
}

void nn2_nvr_set_motion_threshold(NN2NVR* nvr, float thresh)
{
    nvr->motion_thresh = thresh;
}

void nn2_nvr_set_conf_threshold(NN2NVR* nvr, float thresh)
{
    nvr->conf_thresh = thresh;
    nn2_set_conf_threshold(nvr->model, thresh);
}

/* Simple motion detection: compare downsampled grayscale frames */
static int detect_motion(CameraState* cam, const uint8_t* rgb, int w, int h,
                         float threshold)
{
    if (threshold <= 0.0f) return 1; /* motion detection disabled */

    int dw = cam->prev_w, dh = cam->prev_h;
    int pixels = dw * dh;

    /* Allocate prev frame if needed */
    if (!cam->prev_gray) {
        cam->prev_gray = (uint8_t*)calloc(pixels, 1);
    }

    /* Downsample current frame to grayscale */
    uint8_t* curr = (uint8_t*)malloc(pixels);
    float sx = (float)w / dw, sy = (float)h / dh;
    for (int y = 0; y < dh; y++) {
        int iy = (int)(y * sy);
        if (iy >= h) iy = h - 1;
        for (int x = 0; x < dw; x++) {
            int ix = (int)(x * sx);
            if (ix >= w) ix = w - 1;
            const uint8_t* p = rgb + (iy * w + ix) * 3;
            curr[y * dw + x] = (uint8_t)((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
        }
    }

    /* Frame difference */
    int diff_sum = 0;
    for (int i = 0; i < pixels; i++)
        diff_sum += abs((int)curr[i] - (int)cam->prev_gray[i]);

    float avg_diff = (float)diff_sum / pixels;

    /* Update prev */
    memcpy(cam->prev_gray, curr, pixels);
    free(curr);

    return (avg_diff > threshold * 255.0f);
}

static CameraState* find_camera(NN2NVR* nvr, int cam_id)
{
    for (int i = 0; i < nvr->n_cameras; i++)
        if (nvr->cameras[i].id == cam_id) return &nvr->cameras[i];
    return NULL;
}

int nn2_nvr_push_frame(NN2NVR* nvr, int cam_id,
                        const uint8_t* rgb_hwc, int width, int height)
{
    CameraState* cam = find_camera(nvr, cam_id);
    if (!cam || !cam->active) return 0;

    nvr->total_frames++;
    cam->frame_count++;

    /* Skip frames */
    if ((cam->frame_count % nvr->skip_frames) != 0)
        return 0;

    /* Motion check */
    if (!detect_motion(cam, rgb_hwc, width, height, nvr->motion_thresh))
        return 0;

    /* Run detection */
    NN2Det dets[50];
    double t0 = get_ms();
    int n = nn2_detect(nvr->model, rgb_hwc, width, height, dets, 50);
    double ms = get_ms() - t0;

    nvr->total_inferences++;
    nvr->total_inference_ms += ms;
    nvr->total_detections += n;
    cam->det_count += n;

    /* Fire callback */
    if (n > 0 && nvr->callback) {
        NN2NVREvent event;
        event.cam_id = cam_id;
        event.det_count = n;
        memcpy(event.dets, dets, n * sizeof(NN2Det));
        event.timestamp_ms = get_ms();
        event.inference_ms = ms;
        nvr->callback(&event, nvr->cb_userdata);
    }

    return 1;
}

void nn2_nvr_stats(const NN2NVR* nvr, int* total_frames, int* total_detections,
                   double* avg_inference_ms)
{
    if (total_frames) *total_frames = nvr->total_frames;
    if (total_detections) *total_detections = nvr->total_detections;
    if (avg_inference_ms) {
        *avg_inference_ms = (nvr->total_inferences > 0)
            ? nvr->total_inference_ms / nvr->total_inferences : 0;
    }
}

void nn2_nvr_free(NN2NVR* nvr)
{
    if (!nvr) return;
    for (int i = 0; i < nvr->n_cameras; i++)
        free(nvr->cameras[i].prev_gray);
    nn2_free(nvr->model);
    free(nvr);
}
