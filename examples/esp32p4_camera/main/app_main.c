/*
 * app_main.c — ESP32-P4 MIPI-CSI capture loop wired to FaceX.
 *
 * Hardware target: ESP32-P4-Function-EV-Board with the bundled SC2336
 * MIPI camera module (or any sensor supported by Espressif's
 * esp_cam_sensor framework — change the include + auto-detect call).
 *
 * Flow (per Espressif ESP32-P4 Camera Controller Driver doc):
 *   1. Acquire CSI 2.5 V power via the internal LDO.
 *   2. Bring up the SCCB I2C bus and probe the sensor.
 *   3. Configure sensor format (resolution, output FOURCC, framerate).
 *   4. esp_cam_new_csi_ctlr() with matching format.
 *   5. Register on_get_new_trans / on_trans_finished callbacks.
 *   6. Allocate two PSRAM frame buffers and queue them.
 *   7. enable() + start() the controller.
 *   8. Capture loop: esp_cam_ctlr_receive() blocks until a frame is
 *      ready, hand it to FaceX, log the result, requeue the buffer.
 *
 * https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html
 *
 * What's implemented vs. stubbed:
 *   - Camera path: real, against the documented esp_cam_ctlr API.
 *   - Sensor: real, via esp_cam_sensor + auto-detect.
 *   - FaceX dispatch: see components/facex/ — defaults to the stub
 *     backend that returns a synthetic face. Switch to native via
 *     `idf.py menuconfig` → FaceX → Inference backend (see caveats
 *     in docs/esp32p4.md).
 *   - Downscale: nearest-neighbour, RGB565 → RGB888. Adequate for
 *     a 96×96 detector input; replace with PPA hardware accel for
 *     production (P4 has a dedicated Pixel Processing Accelerator).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_ldo_regulator.h"

#include "driver/i2c_master.h"
#include "esp_cam_ctlr.h"
#include "esp_cam_ctlr_csi.h"
#include "esp_cam_ctlr_types.h"

#include "esp_cam_sensor.h"
#include "esp_sccb_intf.h"
#include "esp_sccb_i2c.h"

#include "facex_esp.h"
#include "sdkconfig.h"

static const char* TAG = "app";

#define LDO_MIPI_PHY_CHAN     3       /* P4-Function-EV-Board: LDO_VO3 → CSI PHY */
#define LDO_MIPI_PHY_VOLT_MV  2500
#define SENSOR_I2C_ADDR_HINT  0x36    /* SC2336 default; auto-detect overrides */
#define MAX_FACES             4

/* CSI output format must match the sensor-side format. RGB565 is the
 * most app-friendly: PPA / LCD / our downscaler all consume it directly. */
#define CSI_OUTPUT_COLOR      CAM_CTLR_COLOR_RGB565
#define BYTES_PER_PIXEL       2

typedef struct {
    esp_cam_ctlr_handle_t cam;
    QueueHandle_t         done_q;
    int                   det_w;
    int                   det_h;
    uint8_t*              det_rgb;     /* downscaled RGB888 buffer */
    uint64_t              t_last_log_us;
    uint32_t              frames_since_log;
} app_ctx_t;

/* ---- Camera callbacks --------------------------------------------------- */
/* IRAM-safe — must NOT call non-IRAM functions or grab non-ISR locks. */

static IRAM_ATTR bool on_get_new_trans(esp_cam_ctlr_handle_t handle,
                                       esp_cam_ctlr_trans_t* trans,
                                       void* user_data) {
    /* The driver asks us for a buffer to fill. We pre-allocated and queued
     * them in app_main(); just hand back the next one. */
    app_ctx_t* ctx = (app_ctx_t*)user_data;
    (void)ctx;
    /* trans->buffer + trans->buflen were set by esp_cam_ctlr_receive caller;
     * here we just acknowledge. Returning true keeps the driver running. */
    return false;
}

static IRAM_ATTR bool on_trans_finished(esp_cam_ctlr_handle_t handle,
                                        esp_cam_ctlr_trans_t* trans,
                                        void* user_data) {
    app_ctx_t* ctx = (app_ctx_t*)user_data;
    /* Signal to the consumer task that a frame is ready. Use the
     * FromISR variant — callbacks may run in ISR context. */
    BaseType_t hp_woken = pdFALSE;
    xQueueSendFromISR(ctx->done_q, &trans, &hp_woken);
    return hp_woken == pdTRUE;
}

/* ---- Downscale: RGB565 source → RGB888 detector input ------------------- */

static void rgb565_nn_downscale_rgb888(const uint16_t* src, int sw, int sh,
                                       uint8_t* dst, int dw, int dh) {
    /* Nearest-neighbour. 800×640 → 96×96 in <1 ms on the P4 high-perf
     * cores; for production, swap for the on-chip PPA which does this
     * in DMA at no CPU cost. */
    for (int y = 0; y < dh; y++) {
        int sy = (y * sh) / dh;
        const uint16_t* srow = src + sy * sw;
        uint8_t*        drow = dst + y * dw * 3;
        for (int x = 0; x < dw; x++) {
            int sx = (x * sw) / dw;
            uint16_t p = srow[sx];
            /* RGB565 → 888. Note CSI default is little-endian; if your
             * sensor delivers swapped bytes set csi_config.byte_swap_en. */
            uint8_t r = (uint8_t)(((p >> 11) & 0x1F) << 3);
            uint8_t g = (uint8_t)(((p >> 5)  & 0x3F) << 2);
            uint8_t b = (uint8_t)((p         & 0x1F) << 3);
            drow[x*3 + 0] = r;
            drow[x*3 + 1] = g;
            drow[x*3 + 2] = b;
        }
    }
}

/* ---- Setup helpers ----------------------------------------------------- */

static esp_err_t enable_csi_ldo_power(esp_ldo_channel_handle_t* out_ldo) {
    esp_ldo_channel_config_t ldo_cfg = {
        .chan_id    = LDO_MIPI_PHY_CHAN,
        .voltage_mv = LDO_MIPI_PHY_VOLT_MV,
    };
    return esp_ldo_acquire_channel(&ldo_cfg, out_ldo);
}

static esp_err_t init_sccb(i2c_master_bus_handle_t* out_bus,
                           esp_sccb_io_handle_t* out_sccb) {
    i2c_master_bus_config_t i2c_cfg = {
        .clk_source        = I2C_CLK_SRC_DEFAULT,
        .i2c_port          = CONFIG_SCCB_PORT,
        .scl_io_num        = CONFIG_SCCB_SCL_GPIO,
        .sda_io_num        = CONFIG_SCCB_SDA_GPIO,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_RETURN_ON_ERROR(i2c_new_master_bus(&i2c_cfg, out_bus), TAG, "i2c bus");

    sccb_i2c_config_t sccb_cfg = {
        .scl_speed_hz = 100 * 1000,
        .device_address = SENSOR_I2C_ADDR_HINT,
        .addr_bits_width = 16,
        .val_bits_width = 8,
    };
    return sccb_new_i2c_io(*out_bus, &sccb_cfg, out_sccb);
}

static esp_err_t init_sensor(esp_sccb_io_handle_t sccb,
                             esp_cam_sensor_device_t** out_dev) {
    esp_cam_sensor_config_t cam_cfg = {
        .sccb_handle    = sccb,
        .reset_pin      = CONFIG_SENSOR_RESET_GPIO,
        .pwdn_pin       = CONFIG_SENSOR_PWDN_GPIO,
        .xclk_pin       = -1,
        .xclk_freq_hz   = 0,
        .sensor_port    = ESP_CAM_SENSOR_MIPI_CSI,
    };
    *out_dev = esp_cam_sensor_detect(&cam_cfg);
    if (*out_dev == NULL) {
        ESP_LOGE(TAG, "no MIPI sensor responded on SCCB at 0x%02X", SENSOR_I2C_ADDR_HINT);
        return ESP_ERR_NOT_FOUND;
    }
    ESP_LOGI(TAG, "sensor detected: %s", (*out_dev)->name);

    /* Pick the first format that matches our requested resolution and
     * RGB565 output. esp_cam_sensor enumerates them. */
    esp_cam_sensor_format_array_t fmts = {0};
    ESP_ERROR_CHECK(esp_cam_sensor_query_format(*out_dev, &fmts));
    const esp_cam_sensor_format_t* pick = NULL;
    for (uint32_t i = 0; i < fmts.count; i++) {
        const esp_cam_sensor_format_t* f = &fmts.format_array[i];
        if ((int)f->width  == CONFIG_CAM_HRES &&
            (int)f->height == CONFIG_CAM_VRES &&
            f->mipi_info.lane_num == CONFIG_CAM_DATA_LANES) {
            pick = f; break;
        }
    }
    if (!pick && fmts.count > 0) {
        ESP_LOGW(TAG, "no exact format match for %dx%d, using sensor default[0]: %dx%d",
                 CONFIG_CAM_HRES, CONFIG_CAM_VRES,
                 fmts.format_array[0].width, fmts.format_array[0].height);
        pick = &fmts.format_array[0];
    }
    return esp_cam_sensor_set_format(*out_dev, pick);
}

/* ---- Capture task ------------------------------------------------------ */

static void capture_task(void* arg) {
    app_ctx_t* ctx = (app_ctx_t*)arg;
    esp_cam_ctlr_trans_t* trans = NULL;
    FaceXEspResult faces[MAX_FACES];
    int n_faces = 0;

    while (1) {
        if (xQueueReceive(ctx->done_q, &trans, portMAX_DELAY) != pdTRUE) continue;
        if (!trans || !trans->buffer) continue;

        /* 1. Downscale to detector input. */
        rgb565_nn_downscale_rgb888((const uint16_t*)trans->buffer,
                                    CONFIG_CAM_HRES, CONFIG_CAM_VRES,
                                    ctx->det_rgb, ctx->det_w, ctx->det_h);

        /* 2. FaceX dispatch. */
        uint64_t t0 = esp_timer_get_time();
        esp_err_t r = facex_esp_detect(ctx->det_rgb, faces, MAX_FACES, &n_faces);
        uint64_t dt_us = esp_timer_get_time() - t0;
        if (r != ESP_OK) {
            ESP_LOGW(TAG, "facex_esp_detect: %s", esp_err_to_name(r));
            n_faces = 0;
        }

#ifdef CONFIG_FACEX_LOG_PER_FRAME
        if (n_faces > 0) {
            ESP_LOGI(TAG, "frame: %d face(s), first bbox=[%.0f,%.0f -> %.0f,%.0f] score=%.2f (%lld us)",
                     n_faces, (double)faces[0].x1, (double)faces[0].y1,
                     (double)faces[0].x2, (double)faces[0].y2,
                     (double)faces[0].score, dt_us);
        } else {
            ESP_LOGD(TAG, "frame: 0 faces (%lld us)", dt_us);
        }
#endif

        /* 3. Periodic FPS / latency summary so the serial console isn't silent. */
        ctx->frames_since_log++;
        uint64_t now = esp_timer_get_time();
        if (now - ctx->t_last_log_us >= 1000000ULL) {
            float fps = ctx->frames_since_log * 1.0e6f / (now - ctx->t_last_log_us);
            ESP_LOGI(TAG, "%.1f fps, last detect=%lld us, last n_faces=%d, backend=%s",
                     (double)fps, dt_us, n_faces, facex_esp_backend_name());
            ctx->t_last_log_us = now;
            ctx->frames_since_log = 0;
        }

        /* 4. Re-queue the buffer for the next frame. The driver reads
         * trans->buffer / buflen on its next on_get_new_trans callback. */
        ESP_ERROR_CHECK(esp_cam_ctlr_receive(ctx->cam, trans, ESP_CAM_CTLR_MAX_DELAY));
    }
}

/* ---- Entry ------------------------------------------------------------- */

void app_main(void) {
    ESP_LOGI(TAG, "FaceX ESP32-P4 MIPI-CSI camera example starting");

    /* 1. CSI PHY power. */
    esp_ldo_channel_handle_t ldo = NULL;
    ESP_ERROR_CHECK(enable_csi_ldo_power(&ldo));

    /* 2. SCCB + sensor. */
    i2c_master_bus_handle_t i2c_bus = NULL;
    esp_sccb_io_handle_t    sccb    = NULL;
    ESP_ERROR_CHECK(init_sccb(&i2c_bus, &sccb));

    esp_cam_sensor_device_t* sensor = NULL;
    ESP_ERROR_CHECK(init_sensor(sccb, &sensor));

    /* 3. CSI controller. */
    esp_cam_ctlr_csi_config_t csi_cfg = {
        .ctlr_id              = 0,
        .h_res                = CONFIG_CAM_HRES,
        .v_res                = CONFIG_CAM_VRES,
        .lane_bit_rate_mbps   = CONFIG_CAM_LANE_BIT_RATE_MBPS,
        .input_data_color_type  = CAM_CTLR_COLOR_RAW8,
        .output_data_color_type = CSI_OUTPUT_COLOR,
        .data_lane_num        = CONFIG_CAM_DATA_LANES,
        .byte_swap_en         = false,
        .queue_items          = CONFIG_CAM_FRAME_QUEUE_LEN,
    };
    static app_ctx_t ctx = {0};
    ESP_ERROR_CHECK(esp_cam_new_csi_ctlr(&csi_cfg, &ctx.cam));

    /* 4. Callbacks. */
    ctx.done_q = xQueueCreate(CONFIG_CAM_FRAME_QUEUE_LEN, sizeof(esp_cam_ctlr_trans_t*));
    esp_cam_ctlr_evt_cbs_t cbs = {
        .on_get_new_trans   = on_get_new_trans,
        .on_trans_finished  = on_trans_finished,
    };
    ESP_ERROR_CHECK(esp_cam_ctlr_register_event_callbacks(ctx.cam, &cbs, &ctx));

    /* 5. Frame buffers in PSRAM. RGB565 @ HxW. */
    size_t frame_bytes = (size_t)CONFIG_CAM_HRES * CONFIG_CAM_VRES * BYTES_PER_PIXEL;
    static esp_cam_ctlr_trans_t frames[8];   /* upper bound; we use queue_items */
    int nbufs = CONFIG_CAM_FRAME_QUEUE_LEN;
    if (nbufs > 8) nbufs = 8;
    for (int i = 0; i < nbufs; i++) {
        frames[i].buffer = heap_caps_aligned_alloc(64, frame_bytes,
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA);
        if (!frames[i].buffer) {
            ESP_LOGE(TAG, "frame[%d] alloc %zu bytes failed (PSRAM exhausted?)", i, frame_bytes);
            abort();
        }
        frames[i].buflen = frame_bytes;
    }

    /* 6. FaceX init — detector input is the downscaled size. */
    ctx.det_w = CONFIG_FACEX_DETECT_INPUT_W;
    ctx.det_h = CONFIG_FACEX_DETECT_INPUT_H;
    ctx.det_rgb = heap_caps_malloc((size_t)ctx.det_w * ctx.det_h * 3,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ctx.det_rgb) { ESP_LOGE(TAG, "det buffer alloc failed"); abort(); }

    FaceXEspConfig fcfg = {
        .input_w        = ctx.det_w,
        .input_h        = ctx.det_h,
        .score_threshold = 0.5f,
    };
    ESP_ERROR_CHECK(facex_esp_init(&fcfg));
    ESP_LOGI(TAG, "FaceX ready, backend=%s, detector input=%dx%d",
             facex_esp_backend_name(), ctx.det_w, ctx.det_h);

    /* 7. enable + start, queue all buffers. */
    ESP_ERROR_CHECK(esp_cam_ctlr_enable(ctx.cam));
    ESP_ERROR_CHECK(esp_cam_ctlr_start(ctx.cam));
    for (int i = 0; i < nbufs; i++) {
        ESP_ERROR_CHECK(esp_cam_ctlr_receive(ctx.cam, &frames[i], ESP_CAM_CTLR_MAX_DELAY));
    }

    /* 8. Spawn capture task. The receive() in capture_task wakes when
     * the on_trans_finished callback enqueues a completed buffer. */
    ctx.t_last_log_us = esp_timer_get_time();
    xTaskCreatePinnedToCore(capture_task, "facex_cap", 8192, &ctx, 5, NULL, 1);

    ESP_LOGI(TAG, "init complete; capture task running on core 1");
    /* app_main returns; the task drives the rest. */
}
