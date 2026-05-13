#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return(double)c.QuadPart/f.QuadPart*1000.0;}
#endif
int nn2_load_weights(NN2* ctx, const char* path);
int main(void){
    NN2 c;memset(&c,0,sizeof(c));nn2_tp_init(6);
    if(nn2_load_weights(&c,"weights/yolov8n.bin"))return 1;
    int S=c.input_size;
    float*in=(float*)calloc(256*S*S,4);
    float*out=(float*)malloc(256*S*S*4);
    size_t colsz=(size_t)256*9*S*S/4;
    float*col=(float*)malloc(colsz*4);
    if(!in||!out||!col){printf("alloc fail\n");return 1;}
    int R=20;double t0,ms;
    printf("=== Breakdown (input=%d, 6 threads) ===\n\n",S);
    
    /* im2col benchmarks */
    t0=now();for(int r=0;r<R;r++)nn2_im2col(in,3,S,S,3,3,2,1,col);ms=(now()-t0)/R;
    printf("im2col 3ch 640 s=2:     %5.2f ms\n",ms);
    t0=now();for(int r=0;r<R;r++)nn2_im2col(in,16,S/2,S/2,3,3,2,1,col);ms=(now()-t0)/R;
    printf("im2col 16ch 320 s=2:    %5.2f ms\n",ms);
    t0=now();for(int r=0;r<R;r++)nn2_im2col(in,16,S/4,S/4,3,3,1,1,col);ms=(now()-t0)/R;
    printf("im2col 16ch 160 s=1:    %5.2f ms\n",ms);
    t0=now();for(int r=0;r<R;r++)nn2_im2col(in,32,S/8,S/8,3,3,1,1,col);ms=(now()-t0)/R;
    printf("im2col 32ch 80 s=1:     %5.2f ms\n",ms);
    
    /* GEMM benchmarks */
    t0=now();for(int r=0;r<R;r++)nn2_sgemm(16,27,102400,in,27,col,102400,out,102400);ms=(now()-t0)/R;
    printf("GEMM 16x27x102400:      %5.2f ms  %.0f GFLOPS\n",ms,2.0*16*27*102400/(ms*1e6));
    t0=now();for(int r=0;r<R;r++)nn2_sgemm(32,144,25600,in,144,col,25600,out,25600);ms=(now()-t0)/R;
    printf("GEMM 32x144x25600:      %5.2f ms  %.0f GFLOPS\n",ms,2.0*32*144*25600/(ms*1e6));
    
    /* Full conv */
    t0=now();for(int r=0;r<R;r++)nn2_conv2d(&c.bb_conv[0],in,S,S,out,col);ms=(now()-t0)/R;
    printf("conv0 full (3->16 s=2): %5.2f ms\n",ms);
    t0=now();for(int r=0;r<R;r++)nn2_conv2d(&c.bb_conv[1],in,S/2,S/2,out,col);ms=(now()-t0)/R;
    printf("conv1 full(16->32 s=2): %5.2f ms\n",ms);
    
    free(in);free(out);free(col);nn2_tp_destroy();return 0;
}
