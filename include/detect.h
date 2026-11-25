/*
 * FaceX Detect — Tiny INT8 face detector.
 *
 * YuNet architecture (53K params), ~28 KB WASM engine, ~52 KB INT8 weights.
 * Outputs bbox + 5 keypoints (ArcFace ordering) for downstream alignment.
 *
 * Usage:
 *   Detect* det = detect_init("det_500m_int8.bin");
 *   DetectFace faces[10];
 *   int n = detect_run(det, rgb_160x160, 160, 160, faces, 10);
 *   detect_free(det);
 *
 * https://github.com/facex-engine/facex
 * License: CC BY-NC-SA 4.0
 */

#ifndef FACEX_DETECT_H
#define FACEX_DETECT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct Detect Detect;

/*
 * One detected face.
 *   x1, y1, x2, y2: bbox corners in input pixel coords (top-left origin).
 *   score:          detection confidence in [0, 1].
 *   kps[10]:        5 (x, y) keypoints in ArcFace order:
 *                     [0,1] left eye, [2,3] right eye, [4,5] nose tip,
 *                     [6,7] left mouth corner, [8,9] right mouth corner.
 *                   Coordinates in input pixel space.
 */
typedef struct {
    float x1, y1, x2, y2;
    float score;
    float kps[10];
} DetectFace;

/*
 * Initialize detector engine.
 *   weights_path: path to INT8 weight file (.bin).
 *   Returns: engine handle, or NULL on error.
 */
Detect* detect_init(const char* weights_path);

/*
 * Run detection on a single RGB image.
 *   det:        engine handle.
 *   rgb_hwc:    width*height*3 uint8 array, HWC layout, values 0..255.
 *               Caller is responsible for letterboxing / scaling input
 *               so that the longer side is at most `width` px (typically 160).
 *   width:      image width (recommended 160).
 *   height:     image height (recommended 160).
 *   out:        output buffer for up to `max_faces` DetectFace entries.
 *   max_faces:  capacity of `out`.
 *   Returns: number of faces written (0..max_faces), or -1 on error.
 */
int detect_run(Detect* det,
               const uint8_t* rgb_hwc,
               int width, int height,
               DetectFace* out, int max_faces);

/*
 * Free engine resources.
 */
void detect_free(Detect* det);

/*
 * Get version string.
 */
const char* detect_version(void);

/* Detection confidence threshold (default 0.5).
 * Set before/between detect_run() calls. */
void detect_set_score_threshold(Detect* det, float threshold);

/* NMS IoU threshold (default 0.4). */
void detect_set_nms_threshold(Detect* det, float threshold);

/* Minimum predicted width and height in detector pixels (default 16). */
void detect_set_min_box_side(Detect* det, int pixels);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_DETECT_H */
