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
    if(nn2_load_weights(&c,"weights/yolov8n_320.bin"))return 1;
    int S=c.input_size;
    float*in=(float*)calloc(256*S*S,4);
    float*out=(float*)malloc(256*S*S*4);
    float*col=(float*)malloc(4*1024*1024*4);
    int R=30;double t0,ms;
    printf("=== 320x320 Layer Profile (6 threads) ===\n\n");
    /* Backbone convs */
    int hs[]={320,160,80,40,20}; int chs[]={16,32,64,128,256};
    for(int i=0;i<5;i++){
        t0=now();for(int r=0;r<R;r++)nn2_conv2d(&c.bb_conv[i],in,hs[i],hs[i],out,col);
        printf("bb_conv%d %3d->%3d s2 @%3d: %5.2f ms\n",i,i==0?3:chs[i-1],chs[i],hs[i],( now()-t0)/R);
    }
    /* C2f cv1 (pointwise) */
    for(int i=0;i<4;i++){
        int h=hs[i+1];
        t0=now();for(int r=0;r<R;r++)nn2_conv2d(&c.bb_c2f[i].cv1,in,h,h,out,col);
        printf("c2f%d.cv1 1x1 @%3d:        %5.2f ms\n",i,h,(now()-t0)/R);
    }
    /* C2f bottleneck 3x3 */
    for(int i=0;i<4;i++){
        if(c.bb_c2f[i].n==0)continue;
        int h=hs[i+1];
        t0=now();for(int r=0;r<R;r++)nn2_conv2d(&c.bb_c2f[i].m[0].cv1,in,h,h,out,col);
        printf("c2f%d.bn 3x3 @%3d:         %5.2f ms\n",i,h,(now()-t0)/R);
    }
    /* SPPF maxpool */
    t0=now();for(int r=0;r<R;r++)nn2_maxpool2d(in,128,10,10,5,1,2,out);
    printf("sppf maxpool @10:          %5.2f ms\n",(now()-t0)/R);
    /* Upsample */
    t0=now();for(int r=0;r<R;r++)nn2_upsample_2x(in,256,10,10,out);
    printf("upsample 256ch 10->20:     %5.2f ms\n",(now()-t0)/R);
    t0=now();for(int r=0;r<R;r++)nn2_upsample_2x(in,128,20,20,out);
    printf("upsample 128ch 20->40:     %5.2f ms\n",(now()-t0)/R);
    /* Decode/NMS overhead */
    printf("\n");
    free(in);free(out);free(col);nn2_tp_destroy();return 0;
}
