/*
 * main_video.c — Video detection via ffmpeg pipe.
 *
 * Usage:
 *   nn2_video <weights.bin> <video.mp4> [--skip N] [--threads T] [--json out.jsonl]
 *   nn2_video <weights.bin> rtsp://camera/stream [--skip 5]
 *
 * Spawns ffmpeg to decode video → raw RGB frames → nn2 detection → JSON output.
 * Works with any format ffmpeg supports: mp4, avi, mkv, rtsp, http, etc.
 */

#include "nn2.h"
#include "nn2_json.h"
#include "nn2_nvr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define popen _popen
#define pclose _pclose
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

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <weights.bin> <video_or_rtsp> [options]\n"
        "\n"
        "Options:\n"
        "  --skip N       Process every Nth frame (default: 5)\n"
        "  --threads T    Number of threads (default: auto)\n"
        "  --json FILE    Write detections as JSONL to file\n"
        "  --width W      Input width for ffmpeg (default: 640)\n"
        "  --height H     Input height for ffmpeg (default: 480)\n"
        "  --conf F       Confidence threshold (default: 0.25)\n"
        "  --max-frames N Stop after N frames (0=unlimited)\n"
        "\n"
        "Examples:\n"
        "  %s weights/yolov8n_320.bin video.mp4 --skip 3 --json dets.jsonl\n"
        "  %s weights/yolov8n_320.bin rtsp://192.168.1.100:554/stream --skip 5\n"
        , prog, prog, prog);
}

int main(int argc, char** argv)
{
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* weights = argv[1];
    const char* source = argv[2];
    int skip = 5;
    int threads = 0;
    int vw = 640, vh = 480;
    float conf = 0.25f;
    int max_frames = 0;
    const char* json_path = NULL;

    /* Parse options */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--skip") && i+1 < argc) skip = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i+1 < argc) json_path = argv[++i];
        else if (!strcmp(argv[i], "--width") && i+1 < argc) vw = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i+1 < argc) vh = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--conf") && i+1 < argc) conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-frames") && i+1 < argc) max_frames = atoi(argv[++i]);
    }

    /* Init model */
    NN2* ctx = nn2_init(weights, threads);
    if (!ctx) return 1;
    nn2_set_conf_threshold(ctx, conf);

    /* Open JSON output */
    FILE* json_out = NULL;
    if (json_path) {
        json_out = fopen(json_path, "w");
        if (!json_out) fprintf(stderr, "Warning: cannot open %s for writing\n", json_path);
    }

    fprintf(stderr, "nn2_video: source=%s %dx%d skip=%d conf=%.2f\n", source, vw, vh, skip, conf);

    FILE* pipe;
    int is_raw = 0;

    /* Check if source is a .raw file or stdin "-" → read raw RGB directly */
    if (strcmp(source, "-") == 0) {
        /* Raw frames from stdin */
        pipe = stdin;
        is_raw = 1;
        fprintf(stderr, "nn2_video: reading raw RGB frames from stdin\n");
#ifdef _WIN32
        _setmode(_fileno(stdin), 0x8000); /* _O_BINARY */
#endif
    } else if (strstr(source, ".raw")) {
        pipe = fopen(source, "rb");
        is_raw = 1;
        fprintf(stderr, "nn2_video: reading raw RGB frames from %s\n", source);
    } else {
        /* Use ffmpeg to decode */
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "ffmpeg -hide_banner -loglevel error"
            " -i \"%s\""
            " -f rawvideo -pix_fmt rgb24"
            " -s %dx%d"
            " -",
            source, vw, vh);
        fprintf(stderr, "nn2_video: ffmpeg cmd: %s\n", cmd);
        pipe = popen(cmd, "rb");
        if (!pipe) {
            fprintf(stderr, "Failed to start ffmpeg. Is ffmpeg in PATH?\n");
            nn2_free(ctx);
            return 1;
        }
    }

    if (!pipe) {
        fprintf(stderr, "Failed to open source\n");
        nn2_free(ctx);
        return 1;
    }

    /* Read frames */
    int frame_size = vw * vh * 3;
    uint8_t* frame = (uint8_t*)malloc(frame_size);
    NN2Det dets[100];
    int frame_num = 0;
    int total_dets = 0;
    double total_ms = 0;
    int total_inferences = 0;
    double start_time = get_ms();

    fprintf(stderr, "nn2_video: processing...\n");

    while (fread(frame, 1, frame_size, pipe) == (size_t)frame_size) {
        frame_num++;
        if (max_frames > 0 && frame_num > max_frames) break;

        /* Skip frames */
        if ((frame_num % skip) != 0) continue;

        /* Detect */
        double t0 = get_ms();
        int n = nn2_detect(ctx, frame, vw, vh, dets, 100);
        double ms = get_ms() - t0;
        total_ms += ms;
        total_inferences++;
        total_dets += n;

        /* Output detections */
        if (n > 0) {
            if (json_out) {
                /* Write as JSONL */
                NN2NVREvent event;
                event.cam_id = 0;
                event.det_count = n;
                memcpy(event.dets, dets, n * sizeof(NN2Det));
                event.timestamp_ms = get_ms() - start_time;
                event.inference_ms = ms;
                nn2_event_to_file(&event, json_out);
            }

            /* Print to stderr */
            fprintf(stderr, "\r  frame %d: %d dets (%.1fms) ", frame_num, n, ms);
            for (int i = 0; i < n && i < 3; i++) {
                const char* name = (dets[i].cls >= 0 && dets[i].cls < 80)
                                   ? nn2_coco_names[dets[i].cls] : "?";
                fprintf(stderr, "%s(%.0f%%) ", name, dets[i].score * 100);
            }
        }

        /* Stats every 100 inferences */
        if (total_inferences % 100 == 0) {
            double elapsed = get_ms() - start_time;
            fprintf(stderr, "\n  [%d frames, %d inferences, avg %.1fms, %.1f fps pipeline]\n",
                    frame_num, total_inferences, total_ms / total_inferences,
                    frame_num / (elapsed / 1000.0));
        }
    }

    double elapsed = get_ms() - start_time;

    fprintf(stderr, "\n\n=== nn2_video Results ===\n");
    fprintf(stderr, "  Frames: %d\n", frame_num);
    fprintf(stderr, "  Inferences: %d (skip=%d)\n", total_inferences, skip);
    fprintf(stderr, "  Avg inference: %.1f ms\n",
            total_inferences > 0 ? total_ms / total_inferences : 0);
    fprintf(stderr, "  Total detections: %d\n", total_dets);
    fprintf(stderr, "  Pipeline FPS: %.1f\n", frame_num / (elapsed / 1000.0));
    fprintf(stderr, "  Elapsed: %.1f sec\n", elapsed / 1000.0);
    if (json_path) fprintf(stderr, "  JSON output: %s\n", json_path);

    if (json_out) fclose(json_out);
    free(frame);
    if (!is_raw) pclose(pipe);
    else if (pipe != stdin) fclose(pipe);
    nn2_free(ctx);
    return 0;
}
