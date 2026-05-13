#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int nn2_load_weights(NN2* ctx, const char* path);
int main(void) {
    NN2 c; memset(&c, 0, sizeof(c));
    nn2_tp_init(1);
    if (nn2_load_weights(&c, "weights/yolov8n.bin")) return 1;
    printf("Loaded OK\n");
    
    /* Test Winograd on c2f0 bottleneck: 16->16, 3x3, s=1 at 160x160 */
    ConvLayer* L = &c.bb_c2f[0].m[0].cv1;
    printf("Layer: cout=%d cin=%d k=%d s=%d p=%d\n", L->cout, L->cin, L->k, L->stride, L->pad);
    
    nn2_prepare_winograd(L);
    printf("Winograd prepared: w_wino=%p\n", (void*)L->w_wino);
    
    int H = 160, W = 160;
    int h_tiles = (H+1)/2, w_tiles = (W+1)/2, n_tiles = h_tiles*w_tiles;
    printf("Tiles: %d x %d = %d\n", h_tiles, w_tiles, n_tiles);
    
    size_t v_size = 16 * L->cin * n_tiles;
    size_t m_size = 16 * L->cout * n_tiles;
    printf("V size: %zu, M size: %zu, total: %zu floats\n", v_size, m_size, v_size + m_size);
    
    float* in = (float*)calloc(L->cin * H * W, sizeof(float));
    float* out = (float*)malloc(L->cout * H * W * sizeof(float));
    float* work = (float*)malloc((v_size + m_size) * sizeof(float));
    
    printf("Buffers allocated, running Winograd...\n");
    fflush(stdout);
    nn2_conv2d(L, in, H, W, out, work);
    printf("Winograd OK! out[0]=%.6f\n", out[0]);
    
    free(in); free(out); free(work);
    nn2_tp_destroy();
    return 0;
}
