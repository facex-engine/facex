#include "../frvt/src/alignment.hpp"
#include <cassert>
#include <cstring>

int main() {
    const float points[5][2] = {{0,0},{2,0},{1,1},{0,2},{2,2}};
    float transformed[5][2], matrix[6];
    for (int i = 0; i < 5; i++) {
        transformed[i][0] = 8 - 2 * points[i][1];
        transformed[i][1] = 3 + 2 * points[i][0];
    }
    assert(facex_frvt::umeyama_5pt(points, transformed, matrix));
    const float expected[6] = {0,-2,8,2,0,3};
    for (int j = 0; j < 6; j++) assert(std::abs(matrix[j] - expected[j]) < 1e-5);
    assert(facex_frvt::umeyama_5pt(points, points, matrix));
    const uint8_t rgb[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t out[12];
    facex_frvt::warp_affine_bilinear(rgb, 2, 2, 24, out, 2, matrix);
    assert(std::memcmp(rgb, out, sizeof(rgb)) == 0);
    const float coincident[5][2] = {};
    assert(!facex_frvt::umeyama_5pt(coincident, points, matrix));
}
