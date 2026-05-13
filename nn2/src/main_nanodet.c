#include "nn2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return(double)c.QuadPart/f.QuadPart*1000.0;}
#endif

/* From nanodet.c */
typedef struct NanoDet NanoDet;
NanoDet* nanodet_init(const char* path, int n_threads);
int nanodet_detect(NanoDet* nd, const uint8_t* rgb, int w, int h, NN2Det* out, int max);
void nanodet_free(NanoDet* nd);

int main(int argc, char** argv){
    if(argc<5){fprintf(stderr,"Usage: %s <weights.bin> <image.raw> <w> <h> [threads]\n",argv[0]);return 1;}
    const char*wp=argv[1];const char*ip=argv[2];
    int w=atoi(argv[3]),h=atoi(argv[4]),nt=argc>5?atoi(argv[5]):0;

    FILE*f=fopen(ip,"rb");if(!f)return 1;
    uint8_t*img=(uint8_t*)malloc(w*h*3);fread(img,1,w*h*3,f);fclose(f);

    NanoDet*nd=nanodet_init(wp,nt);
    if(!nd)return 1;

    NN2Det dets[300];
    /* Warmup */
    nanodet_detect(nd,img,w,h,dets,300);

    int N=50;double t0=get_time_ms();int n=0;
    for(int i=0;i<N;i++) n=nanodet_detect(nd,img,w,h,dets,300);
    double ms=(get_time_ms()-t0)/N;
    printf("NanoDet-Plus-m | %dx%d input | %d threads\n",w,h,nt);
    printf("avg %.1f ms (%.1f FPS) over %d runs\n",ms,1000.0/ms,N);
    printf("%d detections:\n",n);
    for(int i=0;i<n&&i<10;i++)
        printf("  cls=%d score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
               dets[i].cls,dets[i].score,dets[i].x1,dets[i].y1,dets[i].x2,dets[i].y2);

    nanodet_free(nd);free(img);return 0;
}
