/*
 * main_nvr.c — NVR multi-camera simulation.
 *
 * Simulates N cameras pushing frames at 30fps.
 * Shows how many cameras can be processed in real-time.
 */

#include "nn2_nvr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double get_ms(void) {
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

static void on_detection(const NN2NVREvent* event, void* userdata)
{
    (void)userdata;
    printf("  [CAM %d] %d objects detected (%.1fms):",
           event->cam_id, event->det_count, event->inference_ms);
    for (int i = 0; i < event->det_count && i < 3; i++) {
        const char* name = (event->dets[i].cls >= 0 && event->dets[i].cls < 80)
                           ? coco_names[event->dets[i].cls] : "?";
        printf(" %s(%.0f%%)", name, event->dets[i].score * 100);
    }
    if (event->det_count > 3) printf(" +%d more", event->det_count - 3);
    printf("\n");
}

int main(int argc, char** argv)
{
    int n_cameras = 8;
    int n_threads = 6;
    int sim_seconds = 5;
    const char* weights = "weights/yolov8n_320.bin";

    if (argc > 1) weights = argv[1];
    if (argc > 2) n_cameras = atoi(argv[2]);
    if (argc > 3) n_threads = atoi(argv[3]);

    printf("nn2 NVR Simulation\n");
    printf("  Model: %s\n", weights);
    printf("  Cameras: %d\n", n_cameras);
    printf("  Threads: %d\n", n_threads);
    printf("  Skip: 5 (process every 5th frame)\n");
    printf("  Simulating %d seconds at 30fps per camera\n\n", sim_seconds);

    NN2NVR* nvr = nn2_nvr_create(weights, n_threads);
    if (!nvr) { fprintf(stderr, "Failed to create NVR\n"); return 1; }

    nn2_nvr_set_callback(nvr, on_detection, NULL);
    nn2_nvr_set_skip_frames(nvr, 5);
    nn2_nvr_set_conf_threshold(nvr, 0.25f);

    for (int i = 0; i < n_cameras; i++) {
        char name[32];
        snprintf(name, sizeof(name), "Camera-%d", i);
        nn2_nvr_add_camera(nvr, i, name);
    }

    /* Create fake camera frames (each camera gets a slightly different image) */
    uint8_t** frames = (uint8_t**)malloc(n_cameras * sizeof(uint8_t*));
    int fw = 640, fh = 480;
    for (int c = 0; c < n_cameras; c++) {
        frames[c] = (uint8_t*)malloc(fw * fh * 3);
        /* Fill with camera-specific pattern */
        for (int i = 0; i < fw * fh * 3; i++)
            frames[c][i] = (uint8_t)((i + c * 17) % 256);
    }

    /* Simulate: 30fps × n_cameras for sim_seconds */
    int total_frames = 30 * sim_seconds;
    double t0 = get_ms();

    for (int f = 0; f < total_frames; f++) {
        /* Each camera pushes one frame per cycle */
        for (int c = 0; c < n_cameras; c++) {
            /* Slightly modify frame to simulate motion */
            if (f > 0) {
                int offset = (f * 7 + c * 13) % (fw * fh * 3);
                frames[c][offset] = (uint8_t)(frames[c][offset] + 50);
            }
            nn2_nvr_push_frame(nvr, c, frames[c], fw, fh);
        }
    }

    double elapsed = get_ms() - t0;

    /* Results */
    int tot_frames, tot_dets;
    double avg_ms;
    nn2_nvr_stats(nvr, &tot_frames, &tot_dets, &avg_ms);

    printf("\n=== NVR Results ===\n");
    printf("  Cameras: %d\n", n_cameras);
    printf("  Total frames pushed: %d\n", tot_frames);
    printf("  Inferences run: %d (skip=%d)\n",
           tot_frames / n_cameras / 5 * n_cameras, 5);
    printf("  Avg inference: %.1f ms\n", avg_ms);
    printf("  Total time: %.0f ms\n", elapsed);
    printf("  Throughput: %.0f frames/sec (all cameras)\n",
           tot_frames / (elapsed / 1000.0));
    printf("  Real-time capacity: %.0f cameras @ 30fps skip-5\n",
           1000.0 / avg_ms * 5.0 / 6.0); /* 6 inferences/sec per camera */
    printf("  Total detections: %d\n", tot_dets);

    /* Cleanup */
    for (int c = 0; c < n_cameras; c++) free(frames[c]);
    free(frames);
    nn2_nvr_free(nvr);
    return 0;
}
