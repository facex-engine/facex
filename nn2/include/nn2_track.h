/*
 * nn2_track.h — Simple object tracker (SORT algorithm).
 *
 * Kalman filter for motion prediction + IoU matching via Hungarian algorithm.
 * Assigns persistent IDs to detections across frames.
 *
 * Usage:
 *   NN2Tracker* tr = nn2_tracker_create(30, 3);  // max_age=30, min_hits=3
 *   // Each frame:
 *   NN2Track tracks[100];
 *   int n = nn2_tracker_update(tr, dets, n_dets, tracks, 100);
 *   for (int i = 0; i < n; i++)
 *       printf("ID %d: %s at (%.0f,%.0f)\n", tracks[i].id, ...);
 *   nn2_tracker_free(tr);
 */

#ifndef NN2_TRACK_H
#define NN2_TRACK_H

#include "nn2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;                 /* persistent track ID */
    int cls;                /* class from last detection */
    float score;            /* confidence from last detection */
    float x1, y1, x2, y2;  /* predicted bbox */
    int age;                /* frames since creation */
    int hits;               /* total matched detections */
    int time_since_update;  /* frames since last matched */
} NN2Track;

typedef struct NN2Tracker NN2Tracker;

/* Create tracker.
 * max_age: delete track after this many frames without match.
 * min_hits: only output tracks with >= this many hits. */
NN2Tracker* nn2_tracker_create(int max_age, int min_hits);

/* Update tracker with new detections. Returns active tracks. */
int nn2_tracker_update(NN2Tracker* tr, const NN2Det* dets, int n_dets,
                        NN2Track* out_tracks, int max_tracks);

/* Get total unique IDs assigned so far */
int nn2_tracker_total_ids(const NN2Tracker* tr);

void nn2_tracker_free(NN2Tracker* tr);

#ifdef __cplusplus
}
#endif

#endif /* NN2_TRACK_H */
