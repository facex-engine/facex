#include "nn2_internal.h"
#include <string.h>

int main(void) {
    uint8_t dark[32 * 32 * 3] = {0}, frame[sizeof(dark)];
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            memset(frame, 0, sizeof(frame));
            memset(frame + 3 * ((8 + dy) * 32 + 8 + dx), 255, 3);
            int hits = 0;
            for (unsigned phase = 0; phase < 16; phase++) {
                MotionResult r = nn2_motion_detect_phase(dark, frame, 32, 32, 25, .003f, phase);
                hits += r.has_motion;
                if (r.has_motion && r.score != 1.f / 64) return 1;
            }
            if (hits != 1) return 2;
        }
    }
    memset(frame, 255, sizeof(frame));
    for (int h = 1; h <= 32; h++)
        for (int w = 1; w <= 32; w++)
            for (unsigned phase = 0; phase < 16; phase++) {
                MotionResult r = nn2_motion_detect_phase(dark, frame, w, h, 25, .003f, phase);
                int any_sample = (int)(phase % 4) < w && (int)(phase / 4) < h;
                if (r.score != (float)any_sample) return 3;
            }
    return 0;
}
