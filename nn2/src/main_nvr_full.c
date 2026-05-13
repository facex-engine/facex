/*
 * main_nvr_full.c — Realistic NVR deployment simulation.
 *
 * 8 cameras × 30fps × 30 seconds with detection + tracking + counting.
 * Measures real-world throughput, latency, and capacity.
 */

#include "nn2.h"
#include "nn2_track.h"
#include "nn2_count.h"
#include "nn2_json.h"
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
#endif

#define MAX_CAMS 32

typedef struct {
    int id;
    char name[32];
    NN2Tracker* tracker;
    NN2Counter* counter;
    int frames_pushed;
    int frames_processed;
    int total_dets;
    double total_inference_ms;
} CamState;

int main(int argc, char** argv)
{
    int n_cams = 8;
    int fps = 30;
    int sim_seconds = 30;
    int skip = 5;
    int threads = 6;
    const char* weights = "weights/yolov8n_320.bin";

    if (argc > 1) weights = argv[1];
    if (argc > 2) n_cams = atoi(argv[2]);
    if (argc > 3) threads = atoi(argv[3]);
    if (argc > 4) sim_seconds = atoi(argv[4]);
    if (n_cams > MAX_CAMS) n_cams = MAX_CAMS;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║        nn2 NVR Deployment Simulation         ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Cameras:    %-3d @ 640x480 30fps             ║\n", n_cams);
    printf("║  Model:      YOLOv8n @ 320x320               ║\n");
    printf("║  Skip:       %d (%.1f detections/sec/cam)     ║\n", skip, (float)fps/skip);
    printf("║  Duration:   %d seconds                       ║\n", sim_seconds);
    printf("║  Threads:    %d                               ║\n", threads);
    printf("║  Pipeline:   Detect → Track → Count → JSON   ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    /* Init model */
    NN2* model = nn2_init(weights, threads);
    if (!model) return 1;

    /* Init cameras with trackers and counters */
    CamState cams[MAX_CAMS];
    for (int i = 0; i < n_cams; i++) {
        memset(&cams[i], 0, sizeof(CamState));
        cams[i].id = i;
        snprintf(cams[i].name, sizeof(cams[i].name), "Cam-%d", i);
        cams[i].tracker = nn2_tracker_create(15, 2);
        /* Horizontal counting line at y=240 (middle of 480px frame) */
        cams[i].counter = nn2_counter_create(0, 240, 640, 240);
    }

    /* Open JSON log */
    FILE* jlog = fopen("nvr_events.jsonl", "w");

    /* Create camera frames (each slightly different) */
    int fw = 640, fh = 480;
    uint8_t** frames = (uint8_t**)malloc(n_cams * sizeof(uint8_t*));
    for (int c = 0; c < n_cams; c++) {
        frames[c] = (uint8_t*)malloc(fw * fh * 3);
        for (int j = 0; j < fw * fh * 3; j++)
            frames[c][j] = (uint8_t)((j * 7 + c * 31) % 200 + 30);
    }

    /* Simulation */
    int total_frames = fps * sim_seconds;
    double start = get_ms();
    double last_report = start;
    int global_dets = 0;
    int global_inferences = 0;

    printf("Running %d seconds of %d cameras...\n\n", sim_seconds, n_cams);

    for (int f = 0; f < total_frames; f++) {
        /* Simulate motion: modify some pixels each frame */
        for (int c = 0; c < n_cams; c++) {
            int offset = ((f + c * 17) * 137) % (fw * fh * 3 - 100);
            for (int j = 0; j < 50; j++)
                frames[c][offset + j] = (uint8_t)((frames[c][offset + j] + 37 + f) % 256);
        }

        /* Process each camera */
        for (int c = 0; c < n_cams; c++) {
            cams[c].frames_pushed++;

            /* Skip frames */
            if ((cams[c].frames_pushed % skip) != 0) continue;

            /* Detect */
            NN2Det dets[50];
            double t0 = get_ms();
            int n = nn2_detect(model, frames[c], fw, fh, dets, 50);
            double ms = get_ms() - t0;

            cams[c].frames_processed++;
            cams[c].total_dets += n;
            cams[c].total_inference_ms += ms;
            global_dets += n;
            global_inferences++;

            /* Track */
            NN2Track tracks[50];
            int nt = nn2_tracker_update(cams[c].tracker, dets, n, tracks, 50);

            /* Count */
            nn2_counter_update(cams[c].counter, tracks, nt);

            /* Log detections */
            if (n > 0 && jlog) {
                fprintf(jlog, "{\"camera\":%d,\"frame\":%d,\"time\":%.1f,"
                        "\"inference_ms\":%.1f,\"detections\":%d,"
                        "\"tracks\":%d,\"count_in\":%d,\"count_out\":%d}\n",
                        c, f, (get_ms() - start) / 1000.0,
                        ms, n, nt,
                        cams[c].counter->count_in,
                        cams[c].counter->count_out);
            }
        }

        /* Progress report every 5 seconds */
        double now = get_ms();
        if (now - last_report > 5000) {
            double elapsed = (now - start) / 1000.0;
            int sim_sec = f / fps;
            double det_rate = global_inferences / elapsed;
            printf("  [%2ds/%ds] %d inferences (%.0f/sec), %d detections, "
                   "avg %.1fms\n",
                   sim_sec, sim_seconds, global_inferences, det_rate,
                   global_dets,
                   global_inferences > 0 ?
                   (now - start) / global_inferences : 0);
            last_report = now;
        }
    }

    double elapsed = (get_ms() - start) / 1000.0;
    if (jlog) fclose(jlog);

    /* Results */
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║            Simulation Results                 ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Duration:       %.1f sec (simulated %ds)    \n", elapsed, sim_seconds);
    printf("║  Total frames:   %d (%d per camera)          \n", total_frames * n_cams, total_frames);
    printf("║  Inferences:     %d                          \n", global_inferences);
    printf("║  Detections:     %d                          \n", global_dets);
    printf("║  Inference rate: %.0f/sec                    \n", global_inferences / elapsed);
    printf("║                                              \n");

    double total_inf_ms = 0;
    for (int c = 0; c < n_cams; c++) total_inf_ms += cams[c].total_inference_ms;
    double avg_inf = global_inferences > 0 ? total_inf_ms / global_inferences : 0;

    printf("║  Avg inference:  %.1f ms                     \n", avg_inf);
    printf("║  Throughput:     %.0f fps (all cameras)      \n",
           total_frames * n_cams / elapsed);
    printf("║                                              \n");

    /* Can we keep up? */
    double needed_rate = (double)n_cams * fps / skip; /* inferences needed per second */
    double actual_rate = global_inferences / elapsed;
    int realtime = actual_rate >= needed_rate * 0.95;
    printf("║  Needed rate:    %.0f inferences/sec         \n", needed_rate);
    printf("║  Actual rate:    %.0f inferences/sec         \n", actual_rate);
    printf("║  Real-time:      %s                          \n", realtime ? "YES" : "NO — OVERLOADED");
    printf("║  Headroom:       %.0f%%                       \n",
           (actual_rate / needed_rate - 1.0) * 100);
    printf("║                                              \n");

    /* Per-camera stats */
    printf("║  Per-camera:                                 \n");
    for (int c = 0; c < n_cams; c++) {
        printf("║    %s: %d dets, %d tracks, In:%d Out:%d  \n",
               cams[c].name, cams[c].total_dets,
               nn2_tracker_total_ids(cams[c].tracker),
               cams[c].counter->count_in,
               cams[c].counter->count_out);
    }
    printf("╚══════════════════════════════════════════════╝\n");
    printf("\nJSON log: nvr_events.jsonl\n");

    /* Cleanup */
    for (int c = 0; c < n_cams; c++) {
        free(frames[c]);
        nn2_tracker_free(cams[c].tracker);
        nn2_counter_free(cams[c].counter);
    }
    free(frames);
    nn2_free(model);
    return 0;
}
