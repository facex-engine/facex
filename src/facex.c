/*
 * facex.c — Unified API: detect + align + embed.
 * See include/facex.h for documentation.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/facex.h"
#include "../include/detect.h"

#define FACEX_VERSION "2.0.0"

/* Include the embedder engine */
#ifndef FACEX_LIB
#define FACEX_LIB
#endif
#include "edgeface_engine.c"

/* External: alignment */
extern int align_face(const uint8_t* src_rgb, int src_w, int src_h,
                       const float kps[10], float* out_f32);

struct FaceX {
    Weights embed_weights;
    Detect* detector;
    int ready;
};

FaceX* facex_init(const char* embed_weights, const char* detect_weights, const char* license_key) {
    if (license_key) g_license_key = license_key;

    FaceX* fx = (FaceX*)calloc(1, sizeof(FaceX));
    if (!fx) return NULL;

    if (detect_weights) {
        fx->detector = detect_init(detect_weights);
        if (!fx->detector) {
            free(fx);
            return NULL;
        }
    }
    if (embed_weights && engine_init(embed_weights, &fx->embed_weights) != 0) {
        if (fx->detector) detect_free(fx->detector);
        free(fx);
        return NULL;
    }

    fx->ready = 1;
    return fx;
}

int facex_detect(FaceX* fx, const uint8_t* rgb_hwc, int width, int height,
                 FaceXResult* out, int max_faces) {
    return facex_detect_prepared(fx, rgb_hwc, width, height, rgb_hwc,
                                 width, height, width, height, out, max_faces);
}

static float source_coordinate(float value, int padding, int resized, int original) {
    double mapped = ((double)value - padding) * original / resized;
    return (float)fmax(0, fmin(original, mapped));
}

int facex_detect_prepared(FaceX* fx, const uint8_t* rgb_hwc, int width, int height,
                          const uint8_t* canvas, int canvas_w, int canvas_h,
                          int resized_w, int resized_h, FaceXResult* out, int max_faces) {
    if (!fx || !fx->ready || !fx->detector || !rgb_hwc || !canvas || !out ||
        width <= 0 || height <= 0 || max_faces <= 0 ||
        resized_w <= 0 || resized_w > canvas_w ||
        resized_h <= 0 || resized_h > canvas_h) return -1;

    DetectFace det_faces[32];
    int n_det = max_faces < 32 ? max_faces : 32;
    int n = detect_run(fx->detector, canvas, canvas_w, canvas_h, det_faces, n_det);
    int dx = (canvas_w - resized_w) / 2, dy = (canvas_h - resized_h) / 2;
    for (int i = 0; i < n; i++) {
        DetectFace* face = &det_faces[i];
        out[i].x1 = source_coordinate(face->x1, dx, resized_w, width);
        out[i].y1 = source_coordinate(face->y1, dy, resized_h, height);
        out[i].x2 = source_coordinate(face->x2, dx, resized_w, width);
        out[i].y2 = source_coordinate(face->y2, dy, resized_h, height);
        out[i].score = face->score;
        for (int k = 0; k < 10; k += 2) {
            out[i].kps[k] = source_coordinate(face->kps[k], dx, resized_w, width);
            out[i].kps[k+1] = source_coordinate(face->kps[k+1], dy, resized_h, height);
        }
        /* Align face to 112×112 using keypoints */
        float aligned[112 * 112 * 3];
        if (!fx->embed_weights.raw || !align_face(rgb_hwc, width, height, out[i].kps, aligned)) {
            memset(out[i].embedding, 0, sizeof(out[i].embedding));
            continue;
        }

        /* Compute embedding */
        facex_embed(fx, aligned, out[i].embedding);
    }

    return n;
}

int facex_embed(FaceX* fx, const float* rgb_hwc, float embedding[512]) {
    if (!fx || !fx->ready) return -1;

    float input_chw[3 * 112 * 112];
    for (int h = 0; h < 112; h++)
        for (int w = 0; w < 112; w++)
            for (int c = 0; c < 3; c++)
                input_chw[c * 112 * 112 + h * 112 + w] = rgb_hwc[(h * 112 + w) * 3 + c];

    edgeface_forward(input_chw, &fx->embed_weights, embedding);
    return 0;
}

float facex_similarity(const float emb1[512], const float emb2[512]) {
    double dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < 512; i++) {
        double a = emb1[i], b = emb2[i];
        dot += a * b;
        n1 += a * a;
        n2 += b * b;
    }
    return n1 > 0 && n2 > 0 ? (float)(dot / sqrt(n1 * n2)) : 0.0f;
}

void facex_free(FaceX* fx) {
    if (!fx) return;
    if (fx->detector) detect_free(fx->detector);
    if (fx->embed_weights.raw) free(fx->embed_weights.raw);
    if (fx->embed_weights.tensors) free(fx->embed_weights.tensors);
    free(fx);
}

const char* facex_version(void) { return FACEX_VERSION; }

void facex_set_score_threshold(FaceX* fx, float t) {
    if (fx && fx->detector) detect_set_score_threshold(fx->detector, t);
}

void facex_set_nms_threshold(FaceX* fx, float t) {
    if (fx && fx->detector) detect_set_nms_threshold(fx->detector, t);
}

void facex_set_min_box_side(FaceX* fx, int pixels) {
    if (fx) detect_set_min_box_side(fx->detector, pixels);
}
