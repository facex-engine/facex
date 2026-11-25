/*
 * align.c — Affine face alignment using 5 keypoints.
 *
 * Transforms detected face region to 112×112 aligned crop
 * matching ArcFace/EdgeFace training alignment.
 *
 * Reference landmarks for 112×112 (ArcFace standard):
 *   left_eye:    (38.2946, 51.6963)
 *   right_eye:   (73.5318, 51.5014)
 *   nose:        (56.0252, 71.7366)
 *   left_mouth:  (41.5493, 92.3655)
 *   right_mouth: (70.7299, 92.2041)
 */

#include <math.h>
#include <float.h>
#include <stdint.h>

/* Reference 112×112 landmarks (ArcFace order) */
static const float REF_KPS[10] = {
    38.2946f, 51.6963f,  /* left eye */
    73.5318f, 51.5014f,  /* right eye */
    56.0252f, 71.7366f,  /* nose */
    41.5493f, 92.3655f,  /* left mouth */
    70.7299f, 92.2041f   /* right mouth */
};

/*
 * Compute similarity transform (rotation + uniform scale + translation)
 * from src keypoints to dst keypoints using least-squares.
 *
 * Transform: [a -b tx] [x]   [x']
 *            [b  a ty] [y] = [y']
 *
 * Returns: M[6] = {a, -b, tx, b, a, ty}
 */
static void estimate_similarity_transform(const float src[10], const float dst[10], double M[6]) {
    const int n = 5;
    double sx_m = 0, sy_m = 0, dx_m = 0, dy_m = 0;
    for (int i = 0; i < n; i++) {
        sx_m += src[2*i]; sy_m += src[2*i+1];
        dx_m += dst[2*i]; dy_m += dst[2*i+1];
    }
    sx_m /= n; sy_m /= n; dx_m /= n; dy_m /= n;

    /* Center the points */
    double num_a = 0, num_b = 0, denom = 0;
    for (int i = 0; i < n; i++) {
        double sx = src[i*2] - sx_m, sy = src[i*2+1] - sy_m;
        double dx = dst[i*2] - dx_m, dy = dst[i*2+1] - dy_m;
        num_a += sx*dx + sy*dy;
        num_b += sx*dy - sy*dx;
        denom += sx*sx + sy*sy;
    }

    double a = denom > 0 ? num_a / denom : 0;
    double b = denom > 0 ? num_b / denom : 0;
    double tx = dx_m - a*sx_m + b*sy_m;
    double ty = dy_m - b*sx_m - a*sy_m;

    M[0] = a;  M[1] = -b; M[2] = tx;
    M[3] = b;  M[4] = a;  M[5] = ty;
}

/*
 * Apply affine transform to warp src image to 112×112 aligned face.
 *
 * src_rgb:  source image, uint8 HWC [0,255]
 * src_w, src_h: source dimensions
 * kps:      5 detected keypoints [x0,y0,x1,y1,...,x4,y4]
 * out_f32:  output 112×112×3 float32 HWC [-1, 1]
 */
int align_face(const uint8_t* src_rgb, int src_w, int src_h,
                const float kps[10], float* out_f32) {
    double M[6];
    /* We want: dst_point = M * src_point
     * But for warping we need inverse: for each dst pixel, find src pixel.
     * So compute M: detected_kps → reference_kps, then invert. */
    estimate_similarity_transform(kps, REF_KPS, M);

    double a = M[0], b = M[3], det = a*a + b*b;
    if (!(det > 0) || !isfinite(det)) {
        for (int i = 0; i < 112*112*3; i++) out_f32[i] = -1.0f;
        return 0;
    }
    double u = a / det, v = b / det;
    /* Stabilize integer grid boundaries against roundoff in the inverse. */
    double grid_eps = 64 * DBL_EPSILON * fmax(src_w, src_h);
    for (int dy = 0; dy < 112; dy++) {
        double sx = -u * M[2] + v * (dy - M[5]) + grid_eps;
        double sy_f = v * M[2] + u * (dy - M[5]) + grid_eps;
        for (int dx = 0; dx < 112; dx++, sx += u, sy_f -= v) {
            if (!(sx >= 0 && sx < src_w && sy_f >= 0 && sy_f < src_h)) {
                for (int c = 0; c < 3; c++) out_f32[(dy*112 + dx)*3 + c] = -1.0f;
                continue;
            }
            /* Bilinear interpolation */
            int ix = (int)sx, iy = (int)sy_f;
            float fx = sx - ix, fy = sy_f - iy;

            for (int c = 0; c < 3; c++) {
                float v = 0;
                if (ix < src_w-1 && iy < src_h-1) {
                    float v00 = src_rgb[(iy*src_w + ix)*3 + c];
                    float v10 = src_rgb[(iy*src_w + ix+1)*3 + c];
                    float v01 = src_rgb[((iy+1)*src_w + ix)*3 + c];
                    float v11 = src_rgb[((iy+1)*src_w + ix+1)*3 + c];
                    v = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy;
                } else {
                    v = src_rgb[(iy*src_w + ix)*3 + c];
                }
                out_f32[(dy*112 + dx)*3 + c] = v / 127.5f - 1.0f;  /* normalize to [-1, 1] */
            }
        }
    }
    return 1;
}
