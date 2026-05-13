/*
 * nn2_antispoof.h — Public API for the MiniFASNet anti-spoof port.
 *
 * Provides a 2-model ensemble (MiniFASNetV2 @ scale 2.7 + MiniFASNetV1SE @ scale 4.0)
 * that mirrors Silent-Face-Anti-Spoofing's predictor. Input is an RGB face crop
 * of arbitrary size; the API crops with the given face bbox + scale to 80×80
 * BGR [0,255] internally (matching ToTensor() that has /255 commented out in
 * the original repo).
 *
 * Class output (3-way softmax):
 *   prob[0] = P(spoof_type_a)
 *   prob[1] = P(LIVE)
 *   prob[2] = P(spoof_type_b)
 */

#ifndef NN2_ANTISPOOF_H
#define NN2_ANTISPOOF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AntiSpoofModel AntiSpoofModel;

/* Model variants we support. */
typedef enum {
    NN2_AS_V2 = 0,       /* MiniFASNetV2  — no SE, 3-class */
    NN2_AS_V1SE = 1,     /* MiniFASNetV1SE — SE blocks, 3-class */
} AntiSpoofVariant;

/* Load weights from .bin (produced by tools/export_minifasnet.py).
 * Returns NULL on failure. n_threads<=0 → auto. */
AntiSpoofModel* nn2_antispoof_load(const char* path, AntiSpoofVariant variant, int n_threads);

void nn2_antispoof_free(AntiSpoofModel* m);

/* Run inference on an already-prepared 80×80 BGR uint8 image (HWC),
 * in [0,255] range. Writes 3 probabilities (sum=1) to prob_out[3].
 * Returns 0 on success.
 *
 * Note: caller is responsible for the crop matching MiniFASNet's expected
 * framing — bbox center, dimensions = (bbox_w * scale, bbox_h * scale)
 * clamped to image bounds, then bilinear resize to 80×80. */
int nn2_antispoof_predict(AntiSpoofModel* m, const unsigned char* bgr_80x80,
                          float prob_out[3]);

/* Convenience: take RGB HWC uint8 image of arbitrary size + bbox [x,y,w,h],
 * crop with model's expected scale, then predict. */
int nn2_antispoof_predict_rgb(AntiSpoofModel* m, const unsigned char* rgb,
                              int src_w, int src_h,
                              int bbox_x, int bbox_y, int bbox_w, int bbox_h,
                              float scale, float prob_out[3]);

#ifdef __cplusplus
}
#endif

#endif /* NN2_ANTISPOOF_H */
