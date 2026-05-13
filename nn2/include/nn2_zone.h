/*
 * nn2_zone.h — Zone intrusion detection.
 *
 * Define polygonal zones on camera view. When a tracked object's
 * centroid enters a zone, trigger an alert. Supports:
 *   - Multiple zones per camera
 *   - Per-zone class filters (e.g., only "person" triggers)
 *   - Dwell time (alert only after N frames inside zone)
 *   - Entry/exit events
 *
 * Usage:
 *   NN2Zone* z = nn2_zone_create("Restricted Area", 4,
 *       (float[]){100,100, 400,100, 400,400, 100,400});
 *   nn2_zone_set_classes(z, 1, (int[]){0}); // only person (class 0)
 *   nn2_zone_set_dwell(z, 3); // alert after 3 frames inside
 *   // Per frame:
 *   NN2ZoneEvent events[10];
 *   int n = nn2_zone_check(z, tracks, n_tracks, events, 10);
 */

#ifndef NN2_ZONE_H
#define NN2_ZONE_H

#include "nn2_track.h"
#include <stdlib.h>
#include <string.h>

#define NN2_ZONE_MAX_POINTS 16
#define NN2_ZONE_MAX_IDS 512

typedef enum {
    NN2_ZONE_ENTER = 1,
    NN2_ZONE_INSIDE = 2,
    NN2_ZONE_EXIT = 3
} NN2ZoneEventType;

typedef struct {
    int track_id;
    int cls;
    NN2ZoneEventType type;
    int dwell_frames; /* how many frames inside */
    float cx, cy;     /* centroid position */
} NN2ZoneEvent;

typedef struct {
    char name[64];
    float poly[NN2_ZONE_MAX_POINTS * 2]; /* x,y pairs */
    int n_points;
    int filter_classes[80]; /* 1 = trigger on this class */
    int n_filter_classes;   /* 0 = all classes trigger */
    int dwell_threshold;    /* frames before alert (0 = immediate) */

    /* Track state */
    int8_t inside[NN2_ZONE_MAX_IDS];      /* 0=outside, 1=inside */
    int16_t dwell_count[NN2_ZONE_MAX_IDS]; /* frames inside */
    int total_intrusions;
} NN2Zone;

/* Point-in-polygon test (ray casting algorithm) */
static inline int nn2_point_in_poly(float px, float py,
                                     const float* poly, int n_points)
{
    int inside = 0;
    for (int i = 0, j = n_points - 1; i < n_points; j = i++) {
        float xi = poly[i*2], yi = poly[i*2+1];
        float xj = poly[j*2], yj = poly[j*2+1];
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
            inside = !inside;
    }
    return inside;
}

static inline NN2Zone* nn2_zone_create(const char* name, int n_points, const float* poly)
{
    NN2Zone* z = (NN2Zone*)calloc(1, sizeof(NN2Zone));
    strncpy(z->name, name ? name : "zone", sizeof(z->name) - 1);
    z->n_points = (n_points > NN2_ZONE_MAX_POINTS) ? NN2_ZONE_MAX_POINTS : n_points;
    memcpy(z->poly, poly, z->n_points * 2 * sizeof(float));
    return z;
}

static inline void nn2_zone_set_classes(NN2Zone* z, int n, const int* classes)
{
    z->n_filter_classes = n;
    for (int i = 0; i < n && i < 80; i++)
        z->filter_classes[classes[i]] = 1;
}

static inline void nn2_zone_set_dwell(NN2Zone* z, int frames) { z->dwell_threshold = frames; }

/* Check tracked objects against zone. Returns events. */
static inline int nn2_zone_check(NN2Zone* z, const NN2Track* tracks, int n_tracks,
                                  NN2ZoneEvent* events, int max_events)
{
    int n_events = 0;

    /* Mark all current tracks */
    int8_t seen[NN2_ZONE_MAX_IDS] = {0};

    for (int i = 0; i < n_tracks && n_events < max_events; i++) {
        int id = tracks[i].id;
        if (id <= 0 || id >= NN2_ZONE_MAX_IDS) continue;
        seen[id] = 1;

        /* Class filter */
        if (z->n_filter_classes > 0 && !z->filter_classes[tracks[i].cls])
            continue;

        /* Centroid */
        float cx = (tracks[i].x1 + tracks[i].x2) * 0.5f;
        float cy = (tracks[i].y1 + tracks[i].y2) * 0.5f;
        int in_zone = nn2_point_in_poly(cx, cy, z->poly, z->n_points);

        int was_inside = z->inside[id];

        if (in_zone && !was_inside) {
            /* ENTER */
            z->inside[id] = 1;
            z->dwell_count[id] = 1;
            if (z->dwell_threshold <= 1) {
                events[n_events++] = (NN2ZoneEvent){id, tracks[i].cls,
                    NN2_ZONE_ENTER, 1, cx, cy};
                z->total_intrusions++;
            }
        } else if (in_zone && was_inside) {
            /* STILL INSIDE */
            z->dwell_count[id]++;
            if (z->dwell_count[id] == z->dwell_threshold) {
                /* Dwell threshold reached — first alert */
                events[n_events++] = (NN2ZoneEvent){id, tracks[i].cls,
                    NN2_ZONE_ENTER, z->dwell_count[id], cx, cy};
                z->total_intrusions++;
            } else if (z->dwell_count[id] > z->dwell_threshold) {
                events[n_events++] = (NN2ZoneEvent){id, tracks[i].cls,
                    NN2_ZONE_INSIDE, z->dwell_count[id], cx, cy};
            }
        } else if (!in_zone && was_inside) {
            /* EXIT */
            z->inside[id] = 0;
            events[n_events++] = (NN2ZoneEvent){id, tracks[i].cls,
                NN2_ZONE_EXIT, z->dwell_count[id], cx, cy};
            z->dwell_count[id] = 0;
        }
    }

    /* Check for tracks that disappeared while inside */
    for (int id = 1; id < NN2_ZONE_MAX_IDS; id++) {
        if (z->inside[id] && !seen[id]) {
            z->inside[id] = 0;
            z->dwell_count[id] = 0;
        }
    }

    return n_events;
}

static inline void nn2_zone_free(NN2Zone* z) { free(z); }

#endif /* NN2_ZONE_H */
