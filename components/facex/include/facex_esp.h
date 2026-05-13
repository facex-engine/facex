/*
 * facex_esp.h — ESP-IDF face-detection wrapper.
 *
 * Sits between the application's camera capture loop and whichever
 * inference backend was selected via Kconfig. The API is intentionally
 * small (init, detect, free) so the camera example doesn't need to
 * know which model is running underneath.
 *
 * Backends:
 *   CONFIG_FACEX_BACKEND_STUB    — synthetic faces (default)
 *   CONFIG_FACEX_BACKEND_NATIVE  — real FaceX engine (slow, eats PSRAM)
 *   CONFIG_FACEX_BACKEND_ESPNN   — reserved, sprint C5 (not yet built)
 *
 * Threading: facex_esp_detect() is NOT reentrant. Call it from a
 * single task — typically the camera RX task.
 */

#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors FaceXResult from include/facex.h, trimmed for what the
 * MCU example actually needs. Keypoints are kept so the application
 * can do alignment if it adds an embedder later. */
typedef struct {
    float x1, y1, x2, y2;   /* bbox in input pixel coords */
    float score;            /* detection confidence [0,1] */
    float kps[10];          /* 5 keypoints (x,y) — left_eye, right_eye, nose, l_mouth, r_mouth */
} FaceXEspResult;

typedef struct {
    /* Input frame dimensions. */
    int input_w;
    int input_h;

    /* Detection score threshold (default 0.5 if 0). */
    float score_threshold;
} FaceXEspConfig;

esp_err_t facex_esp_init(const FaceXEspConfig* cfg);

/* Run detection on one RGB888 (HWC, uint8) frame.
 *   rgb         : input image, input_w * input_h * 3 bytes.
 *   out         : output buffer, max_faces entries.
 *   max_faces   : capacity of out.
 *   out_count   : (output) number of faces written, may be 0.
 * Returns ESP_OK on success, even if out_count == 0. */
esp_err_t facex_esp_detect(const uint8_t* rgb,
                           FaceXEspResult* out, int max_faces, int* out_count);

void facex_esp_free(void);

/* Returns a short string identifying the active backend ("stub",
 * "native", "espnn"). Owned by the library — do not free. */
const char* facex_esp_backend_name(void);

#ifdef __cplusplus
}
#endif
