/*
 * FaceX — Face detection + recognition library.
 *
 * Detect faces, compute embeddings, compare — all in one API.
 * 3ms embedding, 4ms detection, zero dependencies.
 *
 * Usage:
 *   FaceX* fx = facex_init("embed.bin", "detect.bin", NULL);
 *   FaceXResult results[10];
 *   int n = facex_detect(fx, rgb, width, height, results, 10);
 *   float sim = facex_similarity(results[0].embedding, ref_embedding);
 *   facex_free(fx);
 *
 * https://github.com/facex-engine/facex
 * License: Apache 2.0
 */

#ifndef FACEX_H
#define FACEX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct FaceX FaceX;

/* Detection + embedding result for one face */
typedef struct {
    float x1, y1, x2, y2;      /* bbox in input pixel coords */
    float score;                /* detection confidence [0,1] */
    float kps[10];              /* 5 keypoints (x,y) pairs: left_eye, right_eye, nose, left_mouth, right_mouth */
    float embedding[512];       /* 512-dim L2-normalized embedding */
} FaceXResult;

/*
 * Initialize engine with both detector and embedder weights.
 *   embed_weights:  path to embedding model weights (.bin)
 *   detect_weights: path to detector weights (.bin), or NULL to skip detection
 *   license_key:    NULL for plain weights, or key for AES-256 encrypted
 *   Returns: engine handle, or NULL on error
 */
FaceX* facex_init(const char* embed_weights, const char* detect_weights, const char* license_key);

/*
 * Detect faces and compute embeddings in one call.
 *   fx:        engine handle
 *   rgb_hwc:   width*height*3 uint8 array, HWC layout, values 0..255
 *   width:     image width
 *   height:    image height
 *   out:       output buffer for results
 *   max_faces: capacity of out buffer
 *   Returns: number of faces found (0..max_faces), or -1 on error
 *
 * Each result contains bbox, keypoints, confidence, and embedding.
 * Internally: detect → align (affine warp to 112×112) → embed.
 */
int facex_detect(FaceX* fx, const uint8_t* rgb_hwc, int width, int height,
                 FaceXResult* out, int max_faces);

/* Detect on a prepared RGB canvas and align on the original RGB image.
 * The resized image occupies resized_w by resized_h pixels, centered with
 * integer floor padding on its left and top. Results use original coordinates.
 * The detector returns up to 32 faces per call.
 */
int facex_detect_prepared(FaceX* fx, const uint8_t* rgb_hwc, int width, int height,
                          const uint8_t* canvas, int canvas_w, int canvas_h,
                          int resized_w, int resized_h, FaceXResult* out, int max_faces);

/* Set embeddings to zero to skip recognition. A negative threshold uses the configured default. */
int facex_detect_prepared_ex(FaceX* fx, const uint8_t* rgb_hwc, int width, int height,
                          const uint8_t* canvas, int canvas_w, int canvas_h,
                          int resized_w, int resized_h, FaceXResult* out, int max_faces, int embeddings, float threshold);

int facex_embed_landmarks(FaceX* fx, const uint8_t* rgb, int w, int h,
                          const float kps[10], float embedding[512]);


/*
 * Compute embedding only (no detection). Input must be pre-aligned 112×112.
 *   rgb_hwc: 112*112*3 float32 array, HWC layout, values in [-1, 1]
 *   embedding: output 512 floats
 */
int facex_embed(FaceX* fx, const float* rgb_hwc, float embedding[512]);

/*
 * Cosine similarity between two embeddings.
 *   Returns: value in [-1, 1], >0.3 = same person
 */
float facex_similarity(const float emb1[512], const float emb2[512]);

/*
 * Free engine resources.
 */
void facex_free(FaceX* fx);

/* Version string */
const char* facex_version(void);

/* Set detection score threshold (default 0.5) */
void facex_set_score_threshold(FaceX* fx, float threshold);

/* Set NMS IoU threshold (default 0.4) */
void facex_set_nms_threshold(FaceX* fx, float threshold);

/* Minimum predicted width and height in detector pixels (default 16). */
void facex_set_min_box_side(FaceX* fx, int pixels);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_H */
