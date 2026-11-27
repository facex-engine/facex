#ifndef FACEX_FRVT_ALIGNMENT_HPP
#define FACEX_FRVT_ALIGNMENT_HPP
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace facex_frvt {
// ===== Similarity transform (rotation + uniform scale + translation) =====
// Umeyama algorithm minimizing sum of squared errors between two point sets.
// Inputs: src[5][2] (detected) and dst[5][2] (template). Output: 2x3 affine M.
static bool umeyama_5pt(const float src[5][2], const float dst[5][2], float M[6])
{
    float src_cx = 0, src_cy = 0, dst_cx = 0, dst_cy = 0;
    for (int i = 0; i < 5; i++) {
        src_cx += src[i][0]; src_cy += src[i][1];
        dst_cx += dst[i][0]; dst_cy += dst[i][1];
    }
    src_cx /= 5; src_cy /= 5; dst_cx /= 5; dst_cy /= 5;
    float a = 0, b = 0, c = 0;
    for (int i = 0; i < 5; i++) {
        float sx = src[i][0] - src_cx, sy = src[i][1] - src_cy;
        float dx = dst[i][0] - dst_cx, dy = dst[i][1] - dst_cy;
        a += sx*dx + sy*dy;
        b += sx*dy - sy*dx;
        c += sx*sx + sy*sy;
    }
    if (!(c > 0) || !std::isfinite(c)) return false;
    float real = a / c, imag = b / c;
    float det = real * real + imag * imag;
    if (!(det >= 1e-9f) || !std::isfinite(det)) return false;
    M[0] = real; M[1] = -imag;
    M[3] = imag; M[4] = real;
    M[2] = dst_cx - M[0] * src_cx - M[1] * src_cy;
    M[5] = dst_cy - M[3] * src_cx - M[4] * src_cy;
    return std::isfinite(M[2]) && std::isfinite(M[5]);
}

// Warp 24-bit RGB image into a dst×dst RGB plane using inverse affine + bilinear.
// We compute the inverse of the 2x3 forward map, then for each dst pixel sample src.
static void warp_affine_bilinear(
    const uint8_t* src, int sw, int sh, int sd,
    uint8_t* dst, int dst_size, const float fwd[6])
{
    float det = fwd[0] * fwd[4] - fwd[1] * fwd[3];
    float inv[6];
    inv[0] =  fwd[4] / det; inv[1] = -fwd[1] / det;
    inv[3] = -fwd[3] / det; inv[4] =  fwd[0] / det;
    inv[2] = -(inv[0] * fwd[2] + inv[1] * fwd[5]);
    inv[5] = -(inv[3] * fwd[2] + inv[4] * fwd[5]);

    int channels = (sd == 24) ? 3 : 1;
    for (int y = 0; y < dst_size; y++) {
        for (int x = 0; x < dst_size; x++) {
            float fx = inv[0] * x + inv[1] * y + inv[2];
            float fy = inv[3] * x + inv[4] * y + inv[5];
            uint8_t* dp = dst + (y * dst_size + x) * 3;
            if (!(fx >= 0 && fx < sw && fy >= 0 && fy < sh)) {
                dp[0] = dp[1] = dp[2] = 0;
                continue;
            }
            int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
            int x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
            float wx = fx - x0, wy = fy - y0;
            const uint8_t* p00 = src + (static_cast<std::size_t>(y0) * sw + x0) * channels;
            const uint8_t* p01 = src + (static_cast<std::size_t>(y0) * sw + x1) * channels;
            const uint8_t* p10 = src + (static_cast<std::size_t>(y1) * sw + x0) * channels;
            const uint8_t* p11 = src + (static_cast<std::size_t>(y1) * sw + x1) * channels;
            for (int c = 0; c < 3; c++) {
                int sc = (channels == 3) ? c : 0;
                float v = (1 - wx) * (1 - wy) * p00[sc]
                        + wx       * (1 - wy) * p01[sc]
                        + (1 - wx) * wy       * p10[sc]
                        + wx       * wy       * p11[sc];
                dp[c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v + 0.5f)));
            }
        }
    }
}

} // namespace facex_frvt
#endif
