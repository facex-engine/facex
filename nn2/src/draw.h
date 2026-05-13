/*
 * draw.h — Draw bounding boxes and labels on RGB images.
 * Minimal implementation, no fonts — uses simple block characters.
 */

#ifndef NN2_DRAW_H
#define NN2_DRAW_H

#include "nn2.h"
#include <string.h>

/* Colors for different classes (RGB) */
static const uint8_t nn2_colors[][3] = {
    {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255},
    {255,128,0},{128,0,255},{0,128,255},{255,0,128},{128,255,0},{0,255,128},
    {200,100,100},{100,200,100},{100,100,200},{200,200,100},{200,100,200},{100,200,200},
};

/* Draw a single pixel (bounds checked) */
static inline void draw_pixel(uint8_t* img, int w, int h, int x, int y,
                                uint8_t r, uint8_t g, uint8_t b)
{
    if (x >= 0 && x < w && y >= 0 && y < h) {
        uint8_t* p = img + (y * w + x) * 3;
        p[0] = r; p[1] = g; p[2] = b;
    }
}

/* Draw thick horizontal line */
static inline void draw_hline(uint8_t* img, int w, int h,
                               int x1, int x2, int y, int thick,
                               uint8_t r, uint8_t g, uint8_t b)
{
    for (int t = 0; t < thick; t++)
        for (int x = x1; x <= x2; x++)
            draw_pixel(img, w, h, x, y + t, r, g, b);
}

/* Draw thick vertical line */
static inline void draw_vline(uint8_t* img, int w, int h,
                               int x, int y1, int y2, int thick,
                               uint8_t r, uint8_t g, uint8_t b)
{
    for (int t = 0; t < thick; t++)
        for (int y = y1; y <= y2; y++)
            draw_pixel(img, w, h, x + t, y, r, g, b);
}

/* Draw rectangle */
static inline void draw_rect(uint8_t* img, int w, int h,
                               int x1, int y1, int x2, int y2, int thick,
                               uint8_t r, uint8_t g, uint8_t b)
{
    draw_hline(img, w, h, x1, x2, y1, thick, r, g, b);
    draw_hline(img, w, h, x1, x2, y2 - thick + 1, thick, r, g, b);
    draw_vline(img, w, h, x1, y1, y2, thick, r, g, b);
    draw_vline(img, w, h, x2 - thick + 1, y1, y2, thick, r, g, b);
}

/* Draw filled rectangle (for label background) */
static inline void draw_filled_rect(uint8_t* img, int w, int h,
                                     int x1, int y1, int x2, int y2,
                                     uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = y1; y <= y2; y++)
        for (int x = x1; x <= x2; x++)
            draw_pixel(img, w, h, x, y, r, g, b);
}

/* Draw all detections on image */
static inline void nn2_draw_detections(uint8_t* img, int w, int h,
                                        const NN2Det* dets, int n)
{
    for (int i = 0; i < n; i++) {
        int x1 = (int)dets[i].x1, y1 = (int)dets[i].y1;
        int x2 = (int)dets[i].x2, y2 = (int)dets[i].y2;
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
        if (x2 >= w) x2 = w-1; if (y2 >= h) y2 = h-1;

        int ci = dets[i].cls % 18;
        uint8_t r = nn2_colors[ci][0], g = nn2_colors[ci][1], b = nn2_colors[ci][2];

        /* Draw box (2px thick) */
        draw_rect(img, w, h, x1, y1, x2, y2, 2, r, g, b);

        /* Draw label background */
        int lh = 14; /* label height */
        draw_filled_rect(img, w, h, x1, y1 - lh - 2, x1 + 80, y1 - 2, r, g, b);
    }
}

#endif /* NN2_DRAW_H */
