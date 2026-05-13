/*
 * main_server.c — HTTP API server for nn2 detection.
 *
 * Usage: nn2_server <weights.bin> [port] [threads]
 *
 * Endpoints:
 *   POST /detect   — upload JPEG/PNG body → JSON detections
 *   GET  /health   — {"status":"ok","model":"yolov8n","input_size":320}
 *   GET  /stats    — {"requests":N,"avg_ms":12.5,"total_detections":M}
 *
 * Example:
 *   curl -X POST http://localhost:8080/detect --data-binary @photo.jpg
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "nn2.h"
#include "http_server.h"
#include <stdio.h>
#include <stdlib.h>

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

typedef struct {
    NN2* model;
    int requests;
    int total_dets;
    double total_ms;
} ServerCtx;

static void handle_request(const HttpRequest* req, socket_t client, void* userdata)
{
    ServerCtx* ctx = (ServerCtx*)userdata;

    if (strcmp(req->path, "/health") == 0 && strcmp(req->method, "GET") == 0) {
        char json[256];
        snprintf(json, sizeof(json),
            "{\"status\":\"ok\",\"engine\":\"nn2\",\"model\":\"yolov8n\"}");
        http_send_json(client, 200, json);
        return;
    }

    if (strcmp(req->path, "/stats") == 0 && strcmp(req->method, "GET") == 0) {
        char json[256];
        double avg = ctx->requests > 0 ? ctx->total_ms / ctx->requests : 0;
        snprintf(json, sizeof(json),
            "{\"requests\":%d,\"avg_ms\":%.1f,\"total_detections\":%d}",
            ctx->requests, avg, ctx->total_dets);
        http_send_json(client, 200, json);
        return;
    }

    if (strcmp(req->path, "/detect") == 0 && strcmp(req->method, "POST") == 0) {
        if (!req->body || req->content_length <= 0) {
            http_send_json(client, 400, "{\"error\":\"no image data\"}");
            return;
        }

        /* Decode image from memory */
        int w, h, ch;
        uint8_t* img = stbi_load_from_memory(
            (const uint8_t*)req->body, req->content_length, &w, &h, &ch, 3);
        if (!img) {
            http_send_json(client, 400, "{\"error\":\"invalid image\"}");
            return;
        }

        /* Detect */
        NN2Det dets[100];
        double t0 = get_ms();
        int n = nn2_detect(ctx->model, img, w, h, dets, 100);
        double ms = get_ms() - t0;
        stbi_image_free(img);

        ctx->requests++;
        ctx->total_dets += n;
        ctx->total_ms += ms;

        /* Build JSON response */
        char* json = (char*)malloc(n * 256 + 256);
        int pos = sprintf(json,
            "{\"inference_ms\":%.1f,\"image\":{\"width\":%d,\"height\":%d},\"detections\":[",
            ms, w, h);

        for (int i = 0; i < n; i++) {
            const char* name = (dets[i].cls >= 0 && dets[i].cls < 80) ? coco_names[dets[i].cls] : "unknown";
            pos += sprintf(json + pos,
                "%s{\"class\":\"%s\",\"class_id\":%d,\"confidence\":%.3f,"
                "\"bbox\":{\"x1\":%.1f,\"y1\":%.1f,\"x2\":%.1f,\"y2\":%.1f}}",
                i > 0 ? "," : "", name, dets[i].cls, dets[i].score,
                dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
        }
        pos += sprintf(json + pos, "]}");

        http_send_json(client, 200, json);
        free(json);

        fprintf(stderr, "  POST /detect %dx%d → %d dets (%.1fms)\n", w, h, n, ms);
        return;
    }

    http_send_json(client, 404, "{\"error\":\"not found\",\"endpoints\":[\"/detect\",\"/health\",\"/stats\"]}");
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weights.bin> [port] [threads]\n", argv[0]);
        fprintf(stderr, "\nEndpoints:\n");
        fprintf(stderr, "  POST /detect  — upload JPEG/PNG, get JSON detections\n");
        fprintf(stderr, "  GET  /health  — health check\n");
        fprintf(stderr, "  GET  /stats   — inference statistics\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s weights/yolov8n_320.bin 8080 6\n", argv[0]);
        fprintf(stderr, "  curl -X POST http://localhost:8080/detect --data-binary @photo.jpg\n");
        return 1;
    }

    int port = (argc > 2) ? atoi(argv[2]) : 8080;
    int threads = (argc > 3) ? atoi(argv[3]) : 0;

    ServerCtx ctx = {0};
    ctx.model = nn2_init(argv[1], threads);
    if (!ctx.model) return 1;

    fprintf(stderr, "nn2 HTTP Detection Server\n");
    fprintf(stderr, "  Model: %s\n", argv[1]);
    fprintf(stderr, "  Port: %d\n", port);
    fprintf(stderr, "  Usage: curl -X POST http://localhost:%d/detect --data-binary @image.jpg\n\n", port);

    http_listen(port, handle_request, &ctx);

    nn2_free(ctx.model);
    return 0;
}
