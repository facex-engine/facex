/*
 * motion_gate.c — Fast motion detection via frame differencing.
 *
 * Compares current frame vs previous frame using Sum of Absolute Differences.
 * If fewer than threshold pixels changed significantly → no motion → skip inference.
 *
 * Each phase samples one pixel per 4x4 cell.
 *
 * Returns: motion score (0.0 = identical, 1.0 = completely different)
 *          and bounding box of motion region for ROI-crop.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>

/* Compute frame difference score + motion ROI.
 * prev, curr: RGB HWC uint8, width*height*3 bytes.
 * pixel_thresh: per-pixel change threshold (e.g., 25 = ~10% of 255)
 * score_thresh: fraction of changed pixels to trigger motion (e.g., 0.005 = 0.5%)
 */
MotionResult nn2_motion_detect_phase(const uint8_t* prev, const uint8_t* curr,
                                int width, int height,
                                int pixel_thresh, float score_thresh, unsigned phase)
{
    MotionResult r = {0};
    if (width <= 0 || height <= 0) return r;
    int changed = 0;

    /* Track motion bounding box */
    int min_x = width, min_y = height, max_x = 0, max_y = 0;

    const int stride = 4;
    int x_start = phase % 4, y_start = (phase / 4) % 4;
    if (x_start >= width || y_start >= height) return r;
    for (int y = y_start; y < height; y += stride) {
        for (int x = x_start; x < width; x += stride) {
            size_t off = ((size_t)y * width + x) * 3;
            int d = abs((int)prev[off] - (int)curr[off])
                  + abs((int)prev[off+1] - (int)curr[off+1])
                  + abs((int)prev[off+2] - (int)curr[off+2]);
            if (d > pixel_thresh * 3) {
                changed++;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    int sampled_w = 1 + (width - 1 - x_start) / stride;
    int sampled_h = 1 + (height - 1 - y_start) / stride;
    size_t checked = (size_t)sampled_w * sampled_h;
    r.score = (float)changed / (float)checked;
    r.has_motion = (r.score > score_thresh) ? 1 : 0;

    if (r.has_motion) {
        /* Expand ROI by 20% padding, clamp to image bounds */
        int rw = max_x - min_x, rh = max_y - min_y;
        int pad_x = rw / 5 + 16, pad_y = rh / 5 + 16;
        r.roi_x1 = (min_x - pad_x > 0) ? min_x - pad_x : 0;
        r.roi_y1 = (min_y - pad_y > 0) ? min_y - pad_y : 0;
        r.roi_x2 = (max_x + pad_x < width) ? max_x + pad_x : width;
        r.roi_y2 = (max_y + pad_y < height) ? max_y + pad_y : height;

        /* Snap to multiple of 32 for YOLO input alignment */
        r.roi_x1 = (r.roi_x1 / 32) * 32;
        r.roi_y1 = (r.roi_y1 / 32) * 32;
        r.roi_x2 = ((r.roi_x2 + 31) / 32) * 32;
        r.roi_y2 = ((r.roi_y2 + 31) / 32) * 32;
        if (r.roi_x2 > width) r.roi_x2 = width;
        if (r.roi_y2 > height) r.roi_y2 = height;
    }

    return r;
}

MotionResult nn2_motion_detect(const uint8_t* prev, const uint8_t* curr,
                               int width, int height, int pixel_thresh, float score_thresh)
{
    return nn2_motion_detect_phase(prev, curr, width, height, pixel_thresh, score_thresh, 0);
}

/* ============ ROI Crop ============ */

/* Crop RGB HWC image to ROI region */
void nn2_crop_roi(const uint8_t* src, int src_w, int src_h,
                   int x1, int y1, int x2, int y2,
                   uint8_t* dst, int* dst_w, int* dst_h)
{
    *dst_w = x2 - x1;
    *dst_h = y2 - y1;
    for (int y = y1; y < y2; y++) {
        memcpy(dst + (y - y1) * (*dst_w) * 3,
               src + y * src_w * 3 + x1 * 3,
               (*dst_w) * 3);
    }
}
