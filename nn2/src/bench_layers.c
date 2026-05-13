#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) { LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/f.QuadPart*1000.0; }
#endif
int nn2_load_weights(NN2* ctx, const char* path);

static void bench_conv(const char* name, ConvLayer* L, int H, int W) {
    int Ho=(H+2*L->pad-L->k)/L->stride+1, Wo=(W+2*L->pad-L->k)/L->stride+1;
    float* in=(float*)calloc(L->cin*H*W,4);
    float* out=(float*)malloc(L->cout*Ho*Wo*4);
    float* col=(float*)malloc((size_t)L->cin*L->k*L->k*Ho*Wo*4);
    double t0=now_ms(); int R=10;
    for(int r=0;r<R;r++) nn2_conv2d(L,in,H,W,out,col);
    printf("  %-30s %6.2f ms\n",name,(now_ms()-t0)/R);
    free(in);free(out);free(col);
}
static void bench_gemm(const char* name, int m, int k, int n) {
    float* a=(float*)calloc(m*k,4); float* b=(float*)calloc(k*n,4); float* c=(float*)malloc(m*n*4);
    double t0=now_ms(); int R=10;
    for(int r=0;r<R;r++) nn2_sgemm(m,k,n,a,k,b,n,c,n);
    double ms=(now_ms()-t0)/R;
    printf("  %-30s %6.2f ms  %.1f GFLOPS\n",name,ms,2.0*m*k*n/(ms*1e6));
    free(a);free(b);free(c);
}
int main(void) {
    NN2 c; memset(&c,0,sizeof(c)); nn2_tp_init(4);
    if(nn2_load_weights(&c,"weights/yolov8n.bin")) return 1;
    int S=c.input_size;
    printf("=== nn2 profiling (input=%d, 4 threads) ===\n\nConv layers:\n",S);
    bench_conv("conv0: 3->16 s=2",&c.bb_conv[0],S,S);
    bench_conv("conv1: 16->32 s=2",&c.bb_conv[1],S/2,S/2);
    bench_conv("conv2: 32->64 s=2",&c.bb_conv[2],S/4,S/4);
    bench_conv("conv3: 64->128 s=2",&c.bb_conv[3],S/8,S/8);
    bench_conv("conv4: 128->256 s=2",&c.bb_conv[4],S/16,S/16);
    printf("\n1x1 conv (C2f cv1):\n");
    bench_conv("c2f0.cv1: 32->32 1x1",&c.bb_c2f[0].cv1,S/4,S/4);
    bench_conv("c2f1.cv1: 64->64 1x1",&c.bb_c2f[1].cv1,S/8,S/8);
    printf("\n3x3 conv (C2f bottleneck):\n");
    bench_conv("c2f0.bn: 16->16 3x3",&c.bb_c2f[0].m[0].cv1,S/4,S/4);
    bench_conv("c2f1.bn: 32->32 3x3",&c.bb_c2f[1].m[0].cv1,S/8,S/8);
    printf("\nGEMM:\n");
    bench_gemm("16x27x102400",16,27,102400);
    bench_gemm("32x144x25600",32,144,25600);
    bench_gemm("64x288x6400",64,288,6400);
    bench_gemm("128x576x1600",128,576,1600);
    bench_gemm("32x32x25600",32,32,25600);
    nn2_tp_destroy(); return 0;
}
