/*
 * facex_esp.c — ESP-IDF face-detection wrapper implementation.
 *
 * See include/facex_esp.h for the API contract. Backends live behind
 * Kconfig switches; only one is compiled in.
 */

#include "facex_esp.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char* TAG = "facex";

static FaceXEspConfig g_cfg;
static int g_initialized;

#if defined(CONFIG_FACEX_BACKEND_NATIVE)
/* Pulled in by components/facex/CMakeLists.txt when this backend is
 * selected. The native engine is currently too large to fit comfortably
 * on ESP32-P4; this path is a feasibility scaffold, not production. */
#include "../include/facex.h"
static FaceX* g_native = NULL;
#endif

esp_err_t facex_esp_init(const FaceXEspConfig* cfg) {
    if (!cfg) return ESP_ERR_INVALID_ARG;
    if (g_initialized) return ESP_ERR_INVALID_STATE;

    g_cfg = *cfg;
    if (g_cfg.score_threshold <= 0.0f) g_cfg.score_threshold = 0.5f;

#if defined(CONFIG_FACEX_BACKEND_STUB)
    ESP_LOGI(TAG, "backend: stub (%dx%d, threshold=%.2f)",
             g_cfg.input_w, g_cfg.input_h, g_cfg.score_threshold);

#elif defined(CONFIG_FACEX_BACKEND_NATIVE)
    /* The application is responsible for providing weights at the
     * documented path. On ESP-IDF this typically means an SD card
     * mount + a fopen-able file, or an embedded binary blob. */
    extern const char* facex_esp_native_weights_path(void); /* user-provided */
    const char* w = facex_esp_native_weights_path();
    if (!w) {
        ESP_LOGE(TAG, "native backend selected but no weights path provided");
        return ESP_ERR_INVALID_STATE;
    }
    g_native = facex_init(w, NULL, NULL);
    if (!g_native) {
        ESP_LOGE(TAG, "facex_init failed for %s", w);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "backend: native (%dx%d -> 112x112 embed)",
             g_cfg.input_w, g_cfg.input_h);

#elif defined(CONFIG_FACEX_BACKEND_ESPNN)
#  error "FACEX_BACKEND_ESPNN is reserved (sprint C5). Pick stub or native."

#else
#  error "No FACEX backend selected. Run idf.py menuconfig -> FaceX -> Inference backend."
#endif

    g_initialized = 1;
    return ESP_OK;
}

esp_err_t facex_esp_detect(const uint8_t* rgb,
                           FaceXEspResult* out, int max_faces, int* out_count) {
    if (!g_initialized) return ESP_ERR_INVALID_STATE;
    if (!rgb || !out || !out_count || max_faces <= 0) return ESP_ERR_INVALID_ARG;
    *out_count = 0;

#if defined(CONFIG_FACEX_BACKEND_STUB)
    /* Synthetic deterministic "face" centred in the frame, modulated
     * by frame number so the bbox visibly drifts — useful for verifying
     * the camera + UI plumbing. Score breathes between 0.45 and 0.95
     * so the application's threshold-based filter gets exercised. */
    static uint32_t frame_no = 0;
    frame_no++;
    int W = g_cfg.input_w, H = g_cfg.input_h;
    float t = (float)(frame_no % 200) / 200.0f;        /* 0..1 */
    float jitter_x = sinf(t * 6.2831853f) * (W * 0.05f);
    float jitter_y = cosf(t * 6.2831853f) * (H * 0.05f);
    float cx = W * 0.5f + jitter_x;
    float cy = H * 0.5f + jitter_y;
    float r  = W * 0.18f;

    out[0].x1 = cx - r;  out[0].y1 = cy - r;
    out[0].x2 = cx + r;  out[0].y2 = cy + r;
    out[0].score = 0.7f + 0.25f * sinf(t * 6.2831853f);
    /* 5 keypoints in ArcFace order, roughly proportional to the bbox. */
    out[0].kps[0] = cx - r * 0.4f; out[0].kps[1] = cy - r * 0.2f; /* L eye */
    out[0].kps[2] = cx + r * 0.4f; out[0].kps[3] = cy - r * 0.2f; /* R eye */
    out[0].kps[4] = cx;            out[0].kps[5] = cy + r * 0.05f; /* nose */
    out[0].kps[6] = cx - r * 0.3f; out[0].kps[7] = cy + r * 0.5f; /* L mouth */
    out[0].kps[8] = cx + r * 0.3f; out[0].kps[9] = cy + r * 0.5f; /* R mouth */
    *out_count = (out[0].score >= g_cfg.score_threshold) ? 1 : 0;
    return ESP_OK;

#elif defined(CONFIG_FACEX_BACKEND_NATIVE)
    /* Native path. Note: the stock FaceX engine expects 160x160 RGB for
     * the detector; if input_w/h differ the caller has to pre-letterbox.
     * For now we trust the caller and forward straight through. */
    extern int facex_detect(FaceX*, const uint8_t*, int, int,
                            void* /* FaceXResult* */, int);
    /* Use a small temporary buffer matching FaceXResult layout. The
     * native FaceXResult is larger (includes embedding); we copy only
     * the bbox/kps/score subset we care about. */
    typedef struct {
        float x1, y1, x2, y2, score;
        float kps[10];
        float embedding[512];
    } NativeRes;
    NativeRes tmp[8];
    int n = facex_detect(g_native, rgb, g_cfg.input_w, g_cfg.input_h,
                         (void*)tmp, 8);
    if (n < 0) return ESP_FAIL;
    if (n > max_faces) n = max_faces;
    for (int i = 0; i < n; i++) {
        out[i].x1 = tmp[i].x1; out[i].y1 = tmp[i].y1;
        out[i].x2 = tmp[i].x2; out[i].y2 = tmp[i].y2;
        out[i].score = tmp[i].score;
        memcpy(out[i].kps, tmp[i].kps, sizeof(out[i].kps));
    }
    *out_count = n;
    return ESP_OK;
#endif
}

void facex_esp_free(void) {
#if defined(CONFIG_FACEX_BACKEND_NATIVE)
    if (g_native) {
        extern void facex_free(FaceX*);
        facex_free(g_native);
        g_native = NULL;
    }
#endif
    g_initialized = 0;
}

const char* facex_esp_backend_name(void) {
#if defined(CONFIG_FACEX_BACKEND_STUB)
    return "stub";
#elif defined(CONFIG_FACEX_BACKEND_NATIVE)
    return "native";
#elif defined(CONFIG_FACEX_BACKEND_ESPNN)
    return "espnn";
#else
    return "unknown";
#endif
}
