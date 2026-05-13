/*
 * nn2_json.h — Minimal JSON serialization for detection events.
 * No dependencies. Writes to a buffer or FILE*.
 */

#ifndef NN2_JSON_H
#define NN2_JSON_H

#include "nn2.h"
#include "nn2_nvr.h"
#include <stdio.h>

static const char* nn2_coco_names[] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

/* Write detection event as JSON to buffer. Returns bytes written. */
static inline int nn2_event_to_json(const NN2NVREvent* e, char* buf, int bufsize)
{
    int n = snprintf(buf, bufsize,
        "{\"camera\":%d,\"timestamp\":%.1f,\"inference_ms\":%.1f,\"detections\":[",
        e->cam_id, e->timestamp_ms, e->inference_ms);
    for (int i = 0; i < e->det_count && n < bufsize - 100; i++) {
        const char* name = (e->dets[i].cls >= 0 && e->dets[i].cls < 80)
                           ? nn2_coco_names[e->dets[i].cls] : "unknown";
        n += snprintf(buf + n, bufsize - n,
            "%s{\"class\":\"%s\",\"class_id\":%d,\"confidence\":%.3f,"
            "\"bbox\":[%.1f,%.1f,%.1f,%.1f]}",
            i > 0 ? "," : "", name, e->dets[i].cls, e->dets[i].score,
            e->dets[i].x1, e->dets[i].y1, e->dets[i].x2, e->dets[i].y2);
    }
    n += snprintf(buf + n, bufsize - n, "]}");
    return n;
}

/* Write detections as JSON to FILE* */
static inline void nn2_event_to_file(const NN2NVREvent* e, FILE* f)
{
    char buf[4096];
    nn2_event_to_json(e, buf, sizeof(buf));
    fprintf(f, "%s\n", buf);
    fflush(f);
}

#endif /* NN2_JSON_H */
