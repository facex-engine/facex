#include "facex.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

extern int align_face(const uint8_t*, int, int, const float*, float*);

int main(void)
{
    float a[512], b[512];
    const float scales[] = {1.0f, FLT_MIN, FLT_MAX};
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 512; i++) {
            a[i] = i % 2 ? scales[j] : -scales[j];
            b[i] = -a[i];
        }
        if (facex_similarity(a, a) != 1 || facex_similarity(a, b) != -1) return 1;
    }
    static uint8_t img[112*112*3];
    static float crop[112*112*3];
    float kps[] = {38.2946f,51.6963f,73.5318f,51.5014f,56.0252f,
                   71.7366f,41.5493f,92.3655f,70.7299f,92.2041f};
    for (int i = 0; i < 112*112*3; i++) img[i] = (uint8_t)(i % 251);
    if (!align_face(img, 112, 112, kps, crop)) return 2;
    for (int i = 0; i < 112*112*3; i++)
        if (!(fabs(crop[i] - (img[i]/127.5f - 1)) < 1e-6)) { fprintf(stderr, "Identity mismatch %d: %g vs %g\n",i,crop[i],img[i]/127.5f - 1); return 2; }

    for (int i = 0; i < 10; i++) kps[i] = 10;
    if (align_face(img, 112, 112, kps, crop)) return 3;
    for (int i = 0; i < 112*112*3; i++) if (crop[i] != -1) return 3;
    puts("Cosine and alignment checks passed");
    return 0;
}
