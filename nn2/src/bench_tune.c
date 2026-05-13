#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return(double)c.QuadPart/f.QuadPart*1000.0;}
#endif
int main(void){
    /* Test the bottleneck: conv 16->16, 3x3 s=1 @80x80 (at 320 input) */
    float*in=(float*)calloc(16*80*80,4);
    float*out=(float*)malloc(16*80*80*4);
    float*col=(float*)malloc(144*6400*4);
    float*w=(float*)calloc(16*144,4);
    float bias[16]={0};
    ConvLayer L={w,bias,NULL,NULL,NULL,NULL,0,0, 16,16,3,1,1,1};
    
    for(int t=1;t<=6;t++){
        nn2_tp_init(t);
        double t0=now();int R=100;
        for(int r=0;r<R;r++) nn2_conv2d(&L,in,80,80,out,col);
        double ms=(now()-t0)/R;
        printf("conv 16->16 3x3 @80x80: %dt = %.2f ms\n",t,ms);
        nn2_tp_destroy();
    }
    
    /* Also test full forward at 320 */
    printf("\n");
    free(in);free(out);free(col);
    return 0;
}
