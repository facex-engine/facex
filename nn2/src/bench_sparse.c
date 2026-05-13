#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
static double now(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return(double)c.QuadPart/f.QuadPart*1000.0;}
#endif

/* Forward declarations from gemm_sparse_int8.c */
typedef struct SparseINT8Weights SparseINT8Weights;
SparseINT8Weights* nn2_sparse_int8_pack(const int8_t* w, int K, int N, int32_t* cs);
void nn2_sparse_int8_free(SparseINT8Weights* sw);
#ifdef __AVX512VNNI__
void nn2_sparse_int8_gemm(const int8_t* A, int M, int K,
    const SparseINT8Weights* sw, const int32_t* cs,
    const float* ws, const float* bias, float as, int act, float* out);
#endif

int main(void){
    srand(42);
    nn2_tp_init(6);
    printf("=== Sparse INT8 VNNI vs Dense FP32 (6 threads) ===\n\n");

    /* Test: 1x1 conv equivalent M=spatial, K=Cin, N=Cout */
    struct{const char*name;int M,K,N;} tests[]={
        {"C2f 1x1 @160x160 (32->32)", 25600, 32, 32},
        {"C2f 1x1 @80x80 (64->64)",   6400, 64, 64},
        {"Head 1x1 @40x40 (128->80)",  1600,128, 80},
    };

    for(float sparsity=0.0f; sparsity<=0.9f; sparsity+=0.25f){
        printf("--- Sparsity: %.0f%% ---\n", sparsity*100);
        for(int t=0;t<3;t++){
            int M=tests[t].M, K=tests[t].K, N=tests[t].N;
            double ops=2.0*M*K*N;

            /* Dense FP32 */
            float*fw=(float*)calloc(N*K,4);float*fi=(float*)calloc(K*M,4);
            float*fo=(float*)malloc(N*M*4);
            double t0=now();int R=30;
            for(int r=0;r<R;r++)nn2_sgemm(N,K,M,fw,K,fi,M,fo,M);
            double fp32=( now()-t0)/R;

            /* Sparse INT8 */
            int8_t*iw=(int8_t*)malloc(N*K);
            for(int i=0;i<N*K;i++)
                iw[i]=(rand()%100<(int)(sparsity*100))?0:(int8_t)(rand()%20-10);
            int32_t*cs=(int32_t*)calloc(N,4);
            SparseINT8Weights*sw=nn2_sparse_int8_pack(iw,K,N,cs);
            float*ws=(float*)malloc(N*4);float*bi=(float*)calloc(N,4);
            for(int i=0;i<N;i++)ws[i]=0.01f;
            int8_t*ia=(int8_t*)calloc(M*K,1);

#ifdef __AVX512VNNI__
            /* Warmup */
            nn2_sparse_int8_gemm(ia,M,K,sw,cs,ws,bi,0.05f,0,fo);
            t0=now();
            for(int r=0;r<R;r++)
                nn2_sparse_int8_gemm(ia,M,K,sw,cs,ws,bi,0.05f,0,fo);
            double sint8=(now()-t0)/R;
            printf("  %-30s FP32:%5.2fms  SparseINT8:%5.2fms  = %.1fx\n",
                   tests[t].name,fp32,sint8,fp32/sint8);
#else
            printf("  %-30s FP32:%5.2fms  (no VNNI)\n",tests[t].name,fp32);
#endif
            free(fw);free(fi);free(fo);free(iw);free(cs);free(ws);free(bi);free(ia);
            nn2_sparse_int8_free(sw);
        }
    }
    nn2_tp_destroy();return 0;
}
