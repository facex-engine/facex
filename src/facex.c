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

    /* Init embedder */
    if (embed_weights) {
        if (engine_init(embed_weights, &fx->embed_weights) != 0) {
            free(fx);
            return NULL;
        }
    }

    /* Init detector (optional) */
    if (detect_weights) {
        fx->detector = detect_init(detect_weights);
        if (!fx->detector) {
            free(fx->embed_weights.raw);
            free(fx);
            return NULL;
        }
    }

    fx->ready = 1;
    return fx;
}

int facex_detect(FaceX* fx, const uint8_t* rgb_hwc, int width, int height,
                 FaceXResult* out, int max_faces) {
    if (!fx || !fx->ready || !fx->detector) return -1;
    if (!rgb_hwc || !out || max_faces <= 0) return -1;

    /* Step 1: Detect faces */
    DetectFace det_faces[32];
    int n_det = max_faces < 32 ? max_faces : 32;
    int n = detect_run(fx->detector, rgb_hwc, width, height, det_faces, n_det);
    if (n <= 0) return n;

    /* Step 2: For each face, align + embed */
    for (int i = 0; i < n; i++) {
        out[i].x1 = det_faces[i].x1;
        out[i].y1 = det_faces[i].y1;
        out[i].x2 = det_faces[i].x2;
        out[i].y2 = det_faces[i].y2;
        out[i].score = det_faces[i].score;
        memcpy(out[i].kps, det_faces[i].kps, 10 * sizeof(float));

        /* Align face to 112×112 using keypoints */
        float aligned[112 * 112 * 3];
        if (!align_face(rgb_hwc, width, height, det_faces[i].kps, aligned)) {
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
