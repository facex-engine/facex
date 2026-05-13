/*
 * main_pipeline.c — Full NVR analytics pipeline demo.
 *
 * Detection → Tracking → Line Counting → JSON output.
 * Processes raw video frames and shows real-time analytics.
 */

#include "nn2.h"
#include "nn2_track.h"
#include "nn2_count.h"
#include "nn2_json.h"
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

int main(int argc, char** argv)
{
    const char* weights = (argc > 1) ? argv[1] : "weights/yolov8n_320.bin";
    int threads = (argc > 2) ? atoi(argv[2]) : 6;
    int n_frames = (argc > 3) ? atoi(argv[3]) : 150; /* 5 sec @ 30fps */

    printf("=== nn2 Full Analytics Pipeline ===\n\n");
    printf("Model: %s\n", weights);
    printf("Frames: %d (simulated 30fps)\n", n_frames);
    printf("Pipeline: Detection -> Tracking -> Counting -> JSON\n\n");

    /* Init */
    NN2* model = nn2_init(weights, threads);
    if (!model) return 1;

    NN2Tracker* tracker = nn2_tracker_create(15, 2);
    /* Counting line: horizontal at y=240 (middle of 480px frame) */
    NN2Counter* counter = nn2_counter_create(0, 240, 640, 240);

    FILE* json_out = fopen("pipeline_events.jsonl", "w");

    int fw = 640, fh = 480;
    uint8_t* frame = (uint8_t*)malloc(fw * fh * 3);

    /* Stats */
    int total_dets = 0, total_tracks = 0;
    double total_det_ms = 0, total_trk_ms = 0;
    double start = get_ms();

    printf("Processing %d frames...\n\n", n_frames);

    for (int f = 0; f < n_frames; f++) {
        /* Generate frame with varying content */
        memset(frame, 140, fw * fh * 3); /* gray background */

        /* Add moving colored rectangles (simulate objects) */
        int obj_y = 50 + (f * 3) % 380;
        int obj_x = 100 + (f * 5) % 440;
        for (int y = obj_y; y < obj_y + 80 && y < fh; y++)
            for (int x = obj_x; x < obj_x + 60 && x < fw; x++) {
                frame[(y*fw+x)*3+0] = 200;
                frame[(y*fw+x)*3+1] = 100;
                frame[(y*fw+x)*3+2] = 80;
            }

        /* Second object moving opposite */
        int obj2_y = 400 - (f * 2) % 350;
        int obj2_x = 400 + (f * 3) % 200;
        for (int y = obj2_y; y < obj2_y + 70 && y < fh; y++)
            for (int x = obj2_x; x < obj2_x + 90 && x < fw; x++) {
                if (y >= 0 && x >= 0) {
                    frame[(y*fw+x)*3+0] = 60;
                    frame[(y*fw+x)*3+1] = 60;
                    frame[(y*fw+x)*3+2] = 180;
                }
            }

        /* Skip frames (process every 5th) */
        if ((f % 5) != 0) continue;

        /* 1. Detection */
        NN2Det dets[50];
        double t0 = get_ms();
        int n_det = nn2_detect(model, frame, fw, fh, dets, 50);
        double det_ms = get_ms() - t0;
        total_det_ms += det_ms;
        total_dets += n_det;

        /* 2. Tracking */
        NN2Track tracks[50];
        double t1 = get_ms();
        int n_trk = nn2_tracker_update(tracker, dets, n_det, tracks, 50);
        double trk_ms = get_ms() - t1;
        total_trk_ms += trk_ms;
        total_tracks += n_trk;

        /* 3. Counting */
        nn2_counter_update(counter, tracks, n_trk);

        /* 4. JSON output */
        if (n_det > 0 && json_out) {
            fprintf(json_out, "{\"frame\":%d,\"inference_ms\":%.1f,\"detections\":[", f, det_ms);
            for (int i = 0; i < n_det; i++) {
                const char* name = (dets[i].cls >= 0 && dets[i].cls < 80) ? coco_names[dets[i].cls] : "?";
                fprintf(json_out, "%s{\"class\":\"%s\",\"score\":%.3f,\"bbox\":[%.0f,%.0f,%.0f,%.0f]}",
                        i?",":"", name, dets[i].score, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
            }
            fprintf(json_out, "],\"tracks\":[");
            for (int i = 0; i < n_trk; i++) {
                fprintf(json_out, "%s{\"id\":%d,\"class\":%d,\"bbox\":[%.0f,%.0f,%.0f,%.0f]}",
                        i?",":"", tracks[i].id, tracks[i].cls,
                        tracks[i].x1, tracks[i].y1, tracks[i].x2, tracks[i].y2);
            }
            fprintf(json_out, "],\"count\":{\"in\":%d,\"out\":%d}}\n",
                    counter->count_in, counter->count_out);
        }

        /* Console output every 10 processed frames */
        int processed = f / 5 + 1;
        if (processed % 10 == 0 || f == n_frames - 1) {
            printf("  Frame %3d: %d dets, %d tracks, In:%d Out:%d (%.1fms det + %.2fms trk)\n",
                   f, n_det, n_trk, counter->count_in, counter->count_out, det_ms, trk_ms);
        }
    }

    double elapsed = get_ms() - start;
    int processed = n_frames / 5;

    printf("\n=== Pipeline Results ===\n");
    printf("  Frames: %d total, %d processed (skip-5)\n", n_frames, processed);
    printf("  Avg detection: %.1f ms\n", processed > 0 ? total_det_ms / processed : 0);
    printf("  Avg tracking:  %.2f ms\n", processed > 0 ? total_trk_ms / processed : 0);
    printf("  Total detections: %d\n", total_dets);
    printf("  Unique track IDs: %d\n", nn2_tracker_total_ids(tracker));
    printf("  Line crossings — In: %d, Out: %d\n", counter->count_in, counter->count_out);
    printf("  Pipeline FPS: %.0f (end-to-end)\n", n_frames / (elapsed / 1000.0));
    printf("  JSON output: pipeline_events.jsonl\n");

    if (json_out) fclose(json_out);
    free(frame);
    nn2_counter_free(counter);
    nn2_tracker_free(tracker);
    nn2_free(model);
    return 0;
}
