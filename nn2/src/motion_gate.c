/*
 * motion_gate.c — Fast motion detection via frame differencing.
 *
 * Compares current frame vs previous frame using Sum of Absolute Differences.
 * If fewer than threshold pixels changed significantly → no motion → skip inference.
 *
 * AVX-512 SAD: processes 64 bytes (pixels) per cycle.
 * For 320×320×3 = 307,200 bytes: ~4800 iterations = ~2μs at 2.7GHz.
 *
 * Returns: motion score (0.0 = identical, 1.0 = completely different)
 *          and bounding box of motion region for ROI-crop.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>

#ifdef __AVX512BW__
#include <immintrin.h>
#endif

/* Motion detection result */
typedef struct {
    float score;        /* 0.0-1.0: fraction of changed pixels */
    int   has_motion;   /* 1 if score > threshold */
    int   roi_x1, roi_y1, roi_x2, roi_y2;  /* bounding box of motion region */
} MotionResult;

/* Compute frame difference score + motion ROI.
 * prev, curr: RGB HWC uint8, width*height*3 bytes.
 * pixel_thresh: per-pixel change threshold (e.g., 25 = ~10% of 255)
 * score_thresh: fraction of changed pixels to trigger motion (e.g., 0.005 = 0.5%)
 */
MotionResult nn2_motion_detect(const uint8_t* prev, const uint8_t* curr,
                                int width, int height,
                                int pixel_thresh, float score_thresh)
{
    MotionResult r = {0};
    int total_pixels = width * height;
    int changed = 0;

    /* Track motion bounding box */
    int min_x = width, min_y = height, max_x = 0, max_y = 0;

    /* Downsampled check: compare every 4th pixel for speed */
    int stride = 4;

#ifdef __AVX512BW__
    /* AVX-512 path: process 16 pixels at a time (48 bytes RGB) */
    /* Simplified: compare grayscale approximation (R+G+B)/3 per pixel */
    for (int y = 0; y < height; y += stride) {
        const uint8_t* p = prev + y * width * 3;
        const uint8_t* c = curr + y * width * 3;
        for (int x = 0; x + 16 <= width; x += 16) {
            /* Quick SAD on raw bytes (3 channels interleaved) */
            int off = x * 3;
            __m512i vp = _mm512_loadu_si512((__m512i*)(p + off));
            __m512i vc = _mm512_loadu_si512((__m512i*)(c + off));
            /* Absolute difference per byte */
            __m512i vd = _mm512_abs_epi8(_mm512_sub_epi8(
                _mm512_max_epu8(vp, vc), _mm512_min_epu8(vp, vc)));
            /* Actually use SAD: sum of abs diff in groups of 8 bytes */
            /* Simpler: check if any byte > threshold using mask */
            __mmask64 mask = _mm512_cmpgt_epu8_mask(vd, _mm512_set1_epi8((char)pixel_thresh));
            if (mask) {
                /* Count changed pixels (each pixel = 3 bytes, any channel > thresh counts) */
                int bits = __builtin_popcountll(mask);
                changed += bits / 3 + 1; /* approximate */
                /* Update ROI */
                if (x < min_x) min_x = x;
                if (x + 16 > max_x) max_x = x + 16;
                if (y < min_y) min_y = y;
                if (y + stride > max_y) max_y = y + stride;
            }
        }
    }
#else
    /* Scalar fallback */
    for (int y = 0; y < height; y += stride) {
        for (int x = 0; x < width; x += stride) {
            int off = (y * width + x) * 3;
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
#endif

    int checked = (total_pixels / (stride * stride)) + 1;
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
