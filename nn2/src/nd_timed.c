#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return(double)c.QuadPart/f.QuadPart*1000.0;}
#endif
typedef struct NanoDet NanoDet;
NanoDet* nanodet_init(const char* p, int t);
int nanodet_detect(NanoDet* nd, const uint8_t* rgb, int w, int h, NN2Det* out, int max);
void nanodet_free(NanoDet* nd);

int main(void){
    NanoDet*nd=nanodet_init("weights/nanodet_plus_m_320.bin",6);
    if(!nd)return 1;
    uint8_t*img=(uint8_t*)calloc(640*480*3,1);
    NN2Det dets[300];
    /* Warmup */
    nanodet_detect(nd,img,640,480,dets,300);
    /* Timed */
    int N=20;double t0=now();
    for(int i=0;i<N;i++) nanodet_detect(nd,img,640,480,dets,300);
    double ms=(now()-t0)/N;
    printf("NanoDet 320 6t: %.1f ms (%.0f FPS)\n",ms,1000/ms);
    nanodet_free(nd);free(img);return 0;
}
