#include "nn2_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return(double)c.QuadPart/f.QuadPart*1000.0;}
#endif
int main(void){
    nn2_tp_init(6);
    printf("=== INT8 VNNI GEMM vs FP32 (6 threads) ===\n\n");
    /* Conv GEMM: C^T[spatial,Cout] = Input^T[spatial,Cin] x Weight_packed[Cin,Cout]
     * INT8 GEMM: C[M,N] = A[M,K] x B_packed[K,N] where M=spatial, K=Cin, N=Cout */
    struct { const char* name; int spatial,cin,cout; } tests[] = {
        {"1x1 32->32  @160x160", 25600, 32, 32},
        {"1x1 64->64  @80x80",   6400,  64, 64},
        {"1x1 128->128 @40x40",  1600, 128,128},
        {"1x1 128->64 @80x80",   6400, 128, 64},
        {"1x1 256->256 @20x20",   400, 256,256},
    };
    int R=20;
    for(int t=0;t<5;t++){
        int M=tests[t].spatial, K=tests[t].cin, N=tests[t].cout;
        double ops=2.0*M*K*N;
        /* FP32: C[Cout,spatial] = W[Cout,Cin] x Input[Cin,spatial] */
        float*fw=(float*)calloc(N*K,4);float*fi=(float*)calloc(K*M,4);float*fo=(float*)malloc(N*M*4);
        double t0=now();
        for(int r=0;r<R;r++) nn2_sgemm(N,K,M,fw,K,fi,M,fo,M);
        double fp32_ms=(now()-t0)/R;
        /* INT8: C^T[spatial,Cout] = Input^T[spatial,Cin] x W_packed */
        int8_t*iw=(int8_t*)calloc(N*K,1); /* weights [Cout,Cin] */
        int psz=nn2_int8_packed_size(K,N);
        void*ipk=malloc(psz);int32_t*cs=(int32_t*)calloc(N,4);
        float*ws=(float*)malloc(N*4);float*bi=(float*)calloc(N,4);
        for(int i=0;i<N;i++)ws[i]=0.01f;
        nn2_int8_pack_weights(iw,K,N,ipk,cs);
        int8_t*ia=(int8_t*)calloc(M*K,1); /* Input^T[spatial,Cin] */
        t0=now();
        for(int r=0;r<R;r++)nn2_int8_gemm_fused(ia,M,K,N,ipk,cs,ws,bi,0.05f,0.05f,0,fo);
        double int8_ms=(now()-t0)/R;
        printf("%-28s FP32:%5.1fms(%3.0fG) INT8:%5.1fms(%3.0fG) %.1fx\n",
               tests[t].name,fp32_ms,ops/(fp32_ms*1e6),int8_ms,ops/(int8_ms*1e6),fp32_ms/int8_ms);
        free(fw);free(fi);free(fo);free(iw);free(ipk);free(cs);free(ws);free(bi);free(ia);
    }
    nn2_tp_destroy();return 0;
}
