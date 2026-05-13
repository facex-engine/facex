/*
 * decode.c — YOLO detection head decoding + NMS.
 * Optimized: early threshold on raw logits, precomputed total_anchors,
 * fast DFL with AVX-512.
 */

#include "nn2_internal.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

/* DFL: softmax over reg_max=16 bins → weighted sum.
 * AVX-512: process all 16 bins in one pass (1 register). */
#include "fast_exp.h"

static inline float dfl_one(const float* logits, int reg_max, const float* w)
{
#ifdef __AVX512F__
    if (reg_max == 16) {
        __m512 v = _mm512_loadu_ps(logits);
        float maxv = _mm512_reduce_max_ps(v);
        __m512 shifted = _mm512_sub_ps(v, _mm512_set1_ps(maxv));
        __m512 e = _mm512_exp_fast(shifted);
        float sum = _mm512_reduce_add_ps(e);
        __m512 wv = _mm512_loadu_ps(w);
        float val = _mm512_reduce_add_ps(_mm512_mul_ps(e, wv));
        return val / sum;
    }
#endif
    /* Scalar fallback */
    float maxv = logits[0];
    for (int i = 1; i < reg_max; i++)
        if (logits[i] > maxv) maxv = logits[i];

    float sum = 0.0f, val = 0.0f;
    for (int i = 0; i < reg_max; i++) {
        float e = expf(logits[i] - maxv);
        sum += e;
        val += e * w[i];
    }
    return val / sum;
}

/* ============ Decode all scales ============ */

int nn2_decode(const float* bbox_feat,
               const float* cls_feat,
               int num_classes, int input_size, float conf_thresh,
               const float* dfl_w, NN2Det* dets, int max_det)
{
    static const int strides[] = {8, 16, 32};
    int grid_sizes[3];
    for (int i = 0; i < 3; i++)
        grid_sizes[i] = input_size / strides[i];

    int total_anchors = 0;
    for (int i = 0; i < 3; i++)
        total_anchors += grid_sizes[i] * grid_sizes[i];

    /* Pre-compute sigmoid threshold: sigmoid(x) > thresh ⟺ x > ln(thresh/(1-thresh)) */
    float raw_thresh = logf(conf_thresh / (1.0f - conf_thresh));

    int n_det = 0;
    int anchor_offset = 0;

    for (int si = 0; si < 3; si++) {
        int gs = grid_sizes[si];
        int stride = strides[si];
        int n_anchors = gs * gs;

        for (int a = 0; a < n_anchors && n_det < max_det; a++) {
            int idx = anchor_offset + a;

            /* Find best class — threshold on raw logits (no expf needed) */
            const float* cls_ptr = cls_feat + idx;
            float best_raw = -1e9f;
            int best_cls = 0;
#ifdef __AVX512F__
            /* Gather 16 class scores at a time (stride = total_anchors) */
            if (num_classes >= 16) {
                __m512 vmax = _mm512_set1_ps(-1e9f);
                __m512i vidx = _mm512_setzero_epi32();
                int c = 0;
                for (; c + 16 <= num_classes; c += 16) {
                    /* Gather 16 scores with stride = total_anchors * sizeof(float) */
                    __m512i offsets = _mm512_setr_epi32(
                        (c+0)*total_anchors, (c+1)*total_anchors, (c+2)*total_anchors, (c+3)*total_anchors,
                        (c+4)*total_anchors, (c+5)*total_anchors, (c+6)*total_anchors, (c+7)*total_anchors,
                        (c+8)*total_anchors, (c+9)*total_anchors, (c+10)*total_anchors, (c+11)*total_anchors,
                        (c+12)*total_anchors, (c+13)*total_anchors, (c+14)*total_anchors, (c+15)*total_anchors);
                    __m512 scores = _mm512_i32gather_ps(offsets, cls_ptr, 4);
                    __mmask16 gt = _mm512_cmp_ps_mask(scores, vmax, _CMP_GT_OQ);
                    if (gt) {
                        vmax = _mm512_mask_blend_ps(gt, vmax, scores);
                        vidx = _mm512_mask_blend_epi32(gt, vidx, _mm512_add_epi32(_mm512_set1_epi32(c),
                               _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)));
                    }
                }
                /* Horizontal reduction: find max across 16 lanes */
                float vals[16]; int idxs[16];
                _mm512_storeu_ps(vals, vmax);
                _mm512_storeu_epi32(idxs, vidx);
                for (int k = 0; k < 16; k++) {
                    if (vals[k] > best_raw) { best_raw = vals[k]; best_cls = idxs[k]; }
                }
                /* Remaining classes */
                for (; c < num_classes; c++) {
                    float s = cls_ptr[c * total_anchors];
                    if (s > best_raw) { best_raw = s; best_cls = c; }
                }
            } else
#endif
            {
                for (int c = 0; c < num_classes; c++) {
                    float s = cls_ptr[c * total_anchors];
                    if (s > best_raw) { best_raw = s; best_cls = c; }
                }
            }

            /* Early reject on raw logit — saves expf for 95%+ of anchors */
            if (best_raw < raw_thresh) continue;

            float best_score = 1.0f / (1.0f + expf(-best_raw));

            /* DFL decode bbox */
            const float* bbox_ptr = bbox_feat + idx;
            float ltrb[4];
            for (int d = 0; d < 4; d++) {
                float logits[16];
                for (int r = 0; r < NN2_REG_MAX; r++)
                    logits[r] = bbox_ptr[(d * NN2_REG_MAX + r) * total_anchors];
                ltrb[d] = dfl_one(logits, NN2_REG_MAX, dfl_w);
            }

            int ay = a / gs, ax = a % gs;
            float cx = (ax + 0.5f) * stride;
            float cy = (ay + 0.5f) * stride;

            NN2Det* d = &dets[n_det++];
            d->x1 = cx - ltrb[0] * stride;
            d->y1 = cy - ltrb[1] * stride;
            d->x2 = cx + ltrb[2] * stride;
            d->y2 = cy + ltrb[3] * stride;
            d->score = best_score;
            d->cls = best_cls;
        }
        anchor_offset += n_anchors;
    }
    return n_det;
}

/* ============ NMS ============ */

static float iou(const NN2Det* a, const NN2Det* b)
{
    float x1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float y1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float x2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float y2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float inter = (x2-x1>0?x2-x1:0) * (y2-y1>0?y2-y1:0);
    float area_a = (a->x2-a->x1) * (a->y2-a->y1);
    float area_b = (b->x2-b->x1) * (b->y2-b->y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static int cmp_score_desc(const void* a, const void* b)
{
    float sa = ((const NN2Det*)a)->score;
    float sb = ((const NN2Det*)b)->score;
    return (sb > sa) - (sb < sa);
}

int nn2_nms(NN2Det* dets, int n, float iou_thresh)
{
    if (n <= 1) return n;
    qsort(dets, n, sizeof(NN2Det), cmp_score_desc);

    /* Stack-based suppress array — no malloc in NMS hot path */
    uint8_t suppress[1024]; /* max 1024 detections */
    if (n > 1024) n = 1024;
    memset(suppress, 0, n);
    int out_n = 0;
    for (int i = 0; i < n; i++) {
        if (suppress[i]) continue;
        dets[out_n++] = dets[i];
        for (int j = i + 1; j < n; j++) {
            if (!suppress[j] && dets[i].cls == dets[j].cls
                && iou(&dets[i], &dets[j]) > iou_thresh)
                suppress[j] = 1;
        }
    }
    return out_n;
}
