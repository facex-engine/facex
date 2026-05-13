/*
 * track.c — SORT tracker (Simple Online and Realtime Tracking).
 *
 * Kalman filter: constant velocity model on bbox [cx, cy, s, r, dx, dy, ds].
 * Matching: IoU between predicted and detected bboxes, greedy assignment.
 * (Full Hungarian is overkill for <50 objects — greedy is 100x simpler and fast enough.)
 */

#include "nn2_track.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TRACKS 128

/* Kalman state: [cx, cy, s, r, vx, vy, vs] where s=area, r=aspect ratio */
typedef struct {
    float x[7];    /* state: cx, cy, s, r, vx, vy, vs */
    float P[7];    /* diagonal of covariance (simplified) */
    int id;
    int cls;
    float score;
    int age;
    int hits;
    int time_since_update;
    int active;
} KalmanTrack;

struct NN2Tracker {
    KalmanTrack tracks[MAX_TRACKS];
    int n_tracks;
    int next_id;
    int max_age;
    int min_hits;
};

/* Convert bbox to [cx, cy, s, r] */
static void bbox_to_z(float x1, float y1, float x2, float y2, float z[4])
{
    float w = x2 - x1, h = y2 - y1;
    z[0] = x1 + w * 0.5f;  /* cx */
    z[1] = y1 + h * 0.5f;  /* cy */
    z[2] = w * h;            /* s (area) */
    z[3] = w / (h + 1e-6f); /* r (aspect ratio) */
}

/* Convert [cx, cy, s, r] back to bbox */
static void z_to_bbox(const float z[4], float* x1, float* y1, float* x2, float* y2)
{
    float w = sqrtf(z[2] * z[3]);
    float h = z[2] / (w + 1e-6f);
    *x1 = z[0] - w * 0.5f;
    *y1 = z[1] - h * 0.5f;
    *x2 = z[0] + w * 0.5f;
    *y2 = z[1] + h * 0.5f;
}

/* Kalman predict */
static void kf_predict(KalmanTrack* t)
{
    t->x[0] += t->x[4]; /* cx += vx */
    t->x[1] += t->x[5]; /* cy += vy */
    t->x[2] += t->x[6]; /* s += vs */
    if (t->x[2] < 0) t->x[2] = 0;
    /* Increase uncertainty */
    for (int i = 0; i < 7; i++) t->P[i] += 10.0f;
    t->age++;
    t->time_since_update++;
}

/* Kalman update with measurement z[4] = [cx, cy, s, r] */
static void kf_update(KalmanTrack* t, const float z[4])
{
    /* Simplified Kalman gain: K = P / (P + R) */
    float R = 10.0f; /* measurement noise */
    for (int i = 0; i < 4; i++) {
        float K = t->P[i] / (t->P[i] + R);
        t->x[i] += K * (z[i] - t->x[i]);
        t->P[i] *= (1.0f - K);
    }
    /* Update velocities: exponential moving average of residuals */
    float alpha = 0.3f;
    t->x[4] = (1-alpha) * t->x[4] + alpha * (z[0] - t->x[0]);
    t->x[5] = (1-alpha) * t->x[5] + alpha * (z[1] - t->x[1]);
    t->x[6] = (1-alpha) * t->x[6] + alpha * (z[2] - t->x[2]);

    t->time_since_update = 0;
    t->hits++;
}

/* IoU between two bboxes */
static float iou(float ax1, float ay1, float ax2, float ay2,
                  float bx1, float by1, float bx2, float by2)
{
    float ix1 = ax1 > bx1 ? ax1 : bx1;
    float iy1 = ay1 > by1 ? ay1 : by1;
    float ix2 = ax2 < bx2 ? ax2 : bx2;
    float iy2 = ay2 < by2 ? ay2 : by2;
    float inter = (ix2-ix1 > 0 ? ix2-ix1 : 0) * (iy2-iy1 > 0 ? iy2-iy1 : 0);
    float a_area = (ax2-ax1) * (ay2-ay1);
    float b_area = (bx2-bx1) * (by2-by1);
    return inter / (a_area + b_area - inter + 1e-6f);
}

NN2Tracker* nn2_tracker_create(int max_age, int min_hits)
{
    NN2Tracker* tr = (NN2Tracker*)calloc(1, sizeof(NN2Tracker));
    tr->max_age = max_age > 0 ? max_age : 30;
    tr->min_hits = min_hits > 0 ? min_hits : 3;
    tr->next_id = 1;
    return tr;
}

int nn2_tracker_update(NN2Tracker* tr, const NN2Det* dets, int n_dets,
                        NN2Track* out_tracks, int max_tracks)
{
    /* 1. Predict all existing tracks */
    for (int i = 0; i < tr->n_tracks; i++)
        if (tr->tracks[i].active)
            kf_predict(&tr->tracks[i]);

    /* 2. Greedy IoU matching: for each detection, find best matching track */
    int* det_matched = (int*)calloc(n_dets, sizeof(int));     /* -1 = unmatched */
    int* trk_matched = (int*)calloc(tr->n_tracks, sizeof(int));
    for (int i = 0; i < n_dets; i++) det_matched[i] = -1;

    float iou_thresh = 0.1f; /* lowered from 0.3 for fast-moving objects */

    for (int d = 0; d < n_dets; d++) {
        float best_iou = iou_thresh;
        int best_t = -1;

        for (int t = 0; t < tr->n_tracks; t++) {
            if (!tr->tracks[t].active || trk_matched[t]) continue;

            float tx1, ty1, tx2, ty2;
            z_to_bbox(tr->tracks[t].x, &tx1, &ty1, &tx2, &ty2);

            float v = iou(dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2,
                          tx1, ty1, tx2, ty2);
            if (v > best_iou) { best_iou = v; best_t = t; }
        }

        if (best_t >= 0) {
            det_matched[d] = best_t;
            trk_matched[best_t] = 1;
        }
    }

    /* 3. Update matched tracks */
    for (int d = 0; d < n_dets; d++) {
        if (det_matched[d] >= 0) {
            KalmanTrack* t = &tr->tracks[det_matched[d]];
            float z[4];
            bbox_to_z(dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2, z);
            kf_update(t, z);
            t->cls = dets[d].cls;
            t->score = dets[d].score;
        }
    }

    /* 4. Create new tracks for unmatched detections */
    for (int d = 0; d < n_dets; d++) {
        if (det_matched[d] >= 0) continue;
        if (tr->n_tracks >= MAX_TRACKS) break;

        KalmanTrack* t = &tr->tracks[tr->n_tracks++];
        memset(t, 0, sizeof(*t));
        float z[4];
        bbox_to_z(dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2, z);
        for (int i = 0; i < 4; i++) t->x[i] = z[i];
        for (int i = 0; i < 7; i++) t->P[i] = 100.0f;
        t->id = tr->next_id++;
        t->cls = dets[d].cls;
        t->score = dets[d].score;
        t->active = 1;
        t->hits = 1;
    }

    /* 5. Deactivate old tracks */
    for (int t = 0; t < tr->n_tracks; t++) {
        if (tr->tracks[t].active && tr->tracks[t].time_since_update > tr->max_age)
            tr->tracks[t].active = 0;
    }

    /* 6. Output active tracks with enough hits */
    int n_out = 0;
    for (int t = 0; t < tr->n_tracks && n_out < max_tracks; t++) {
        KalmanTrack* tk = &tr->tracks[t];
        if (!tk->active) continue;
        if (tk->hits < tr->min_hits && tk->time_since_update > 0) continue;

        NN2Track* out = &out_tracks[n_out++];
        out->id = tk->id;
        out->cls = tk->cls;
        out->score = tk->score;
        z_to_bbox(tk->x, &out->x1, &out->y1, &out->x2, &out->y2);
        out->age = tk->age;
        out->hits = tk->hits;
        out->time_since_update = tk->time_since_update;
    }

    free(det_matched);
    free(trk_matched);
    return n_out;
}

int nn2_tracker_total_ids(const NN2Tracker* tr) { return tr->next_id - 1; }

void nn2_tracker_free(NN2Tracker* tr) { free(tr); }
