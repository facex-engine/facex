/*
 * nn2_count.h — Line-crossing people counter.
 *
 * Counts objects crossing a virtual line. Tracks centroid of each
 * tracked object and increments in/out counters when it crosses.
 *
 * Usage:
 *   NN2Counter* cnt = nn2_counter_create(320, 240, 0, 240, 640, 240);
 *   // horizontal line at y=240 from x=0 to x=640
 *   nn2_counter_update(cnt, tracks, n_tracks);
 *   printf("In: %d, Out: %d\n", cnt->count_in, cnt->count_out);
 */

#ifndef NN2_COUNT_H
#define NN2_COUNT_H

#include "nn2_track.h"
#include <stdlib.h>

#define NN2_COUNT_MAX_IDS 1024

typedef struct {
    /* Line definition: (x1,y1) to (x2,y2) */
    float lx1, ly1, lx2, ly2;

    /* Counters */
    int count_in;   /* crossed from positive → negative side */
    int count_out;  /* crossed from negative → positive side */

    /* Track state: last side for each track ID */
    int8_t last_side[NN2_COUNT_MAX_IDS]; /* 0=unknown, 1=positive, -1=negative */
} NN2Counter;

static inline NN2Counter* nn2_counter_create(float x1, float y1, float x2, float y2)
{
    NN2Counter* c = (NN2Counter*)calloc(1, sizeof(NN2Counter));
    c->lx1 = x1; c->ly1 = y1; c->lx2 = x2; c->ly2 = y2;
    return c;
}

/* Determine which side of the line a point is on.
 * Returns +1 or -1 based on cross product sign. */
static inline int nn2_counter_side(const NN2Counter* c, float px, float py)
{
    float cross = (c->lx2 - c->lx1) * (py - c->ly1)
                - (c->ly2 - c->ly1) * (px - c->lx1);
    return (cross > 0) ? 1 : -1;
}

/* Update counter with tracked objects */
static inline void nn2_counter_update(NN2Counter* c, const NN2Track* tracks, int n)
{
    for (int i = 0; i < n; i++) {
        int id = tracks[i].id;
        if (id <= 0 || id >= NN2_COUNT_MAX_IDS) continue;

        /* Centroid */
        float cx = (tracks[i].x1 + tracks[i].x2) * 0.5f;
        float cy = (tracks[i].y1 + tracks[i].y2) * 0.5f;
        int side = nn2_counter_side(c, cx, cy);

        int8_t prev = c->last_side[id];
        if (prev != 0 && prev != side) {
            /* Crossed! */
            if (prev == 1 && side == -1) c->count_in++;
            else if (prev == -1 && side == 1) c->count_out++;
        }
        c->last_side[id] = (int8_t)side;
    }
}

static inline void nn2_counter_free(NN2Counter* c) { free(c); }

#endif /* NN2_COUNT_H */
