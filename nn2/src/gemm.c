/*
 * gemm.c — FP32 GEMM: AVX-512 → AVX2 → scalar fallback.
 *
 * C[M,N] = A[M,K] * B[K,N]  (row-major)
 * AVX-512: MR=6 × NR=32 (ZMM, 16 float/reg), 12 acc + 2 B + 1 A = 15/32 regs.
 * AVX2:    MR=6 × NR=16 (YMM, 8 float/reg),  12 acc + 2 B + 1 A = 15/16 regs.
 * Output-stationary, parallel across N-tiles.
 */

#include "nn2_internal.h"
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

/* ============================================================
 * AVX-512 path: MR=6, NR=32
 * ============================================================ */

#ifdef __AVX512F__

#define NR512 32

static inline void ukernel_6x32_avx512(
    int K,
    const float* restrict A, int lda,
    const float* restrict B, int ldb,
    float* restrict C, int ldc,
    int mr, int nr)
{
    __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
    __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
    __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
    __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
    __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
    __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();
    const float*a0=A,*a1=A,*a2=A,*a3=A,*a4=A,*a5=A;
    if(mr>1)a1=A+lda;if(mr>2)a2=A+2*lda;if(mr>3)a3=A+3*lda;
    if(mr>4)a4=A+4*lda;if(mr>5)a5=A+5*lda;
    for(int k=0;k<K;k++){
        const float*bp=B+k*ldb;
        if(k+2<K)_mm_prefetch((const char*)(B+(k+2)*ldb),_MM_HINT_T0);
        __m512 b0=_mm512_loadu_ps(bp),b1=_mm512_loadu_ps(bp+16);
        __m512 va;
        va=_mm512_set1_ps(a0[k]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01);
        va=_mm512_set1_ps(a1[k]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11);
        va=_mm512_set1_ps(a2[k]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21);
        va=_mm512_set1_ps(a3[k]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31);
        va=_mm512_set1_ps(a4[k]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41);
        va=_mm512_set1_ps(a5[k]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51);
    }
    if(nr>=32){
        #define W512(i,lo,hi) if(i<mr){_mm512_storeu_ps(C+i*ldc,lo);_mm512_storeu_ps(C+i*ldc+16,hi);}
        W512(0,c00,c01);W512(1,c10,c11);W512(2,c20,c21);
        W512(3,c30,c31);W512(4,c40,c41);W512(5,c50,c51);
        #undef W512
    } else {
        __mmask16 ml=(nr>=16)?0xFFFF:((__mmask16)1<<nr)-1;
        __mmask16 mh=(nr<=16)?0:(nr>=32)?0xFFFF:((__mmask16)1<<(nr-16))-1;
        #define WM(i,lo,hi) if(i<mr){_mm512_mask_storeu_ps(C+i*ldc,ml,lo);if(mh)_mm512_mask_storeu_ps(C+i*ldc+16,mh,hi);}
        WM(0,c00,c01);WM(1,c10,c11);WM(2,c20,c21);WM(3,c30,c31);WM(4,c40,c41);WM(5,c50,c51);
        #undef WM
    }
}

#endif /* __AVX512F__ */

typedef struct {
    int M, K, N;
    const float* A; int lda;
    const float* B;
    float* C; int ldc;
} GemmCtx;

#ifdef __AVX512F__
static void gemm_avx512_worker(void* arg, int start, int end)
{
    GemmCtx* g = (GemmCtx*)arg;
    for (int ni = start; ni < end; ni++) {
        int n = ni * NR512;
        int nr = (n + NR512 <= g->N) ? NR512 : g->N - n;
        for (int m = 0; m < g->M; m += NN2_MR) {
            int mr = (m + NN2_MR <= g->M) ? NN2_MR : g->M - m;
            ukernel_6x32_avx512(g->K, g->A + m*g->lda, g->lda,
                                g->B + n, g->N,
                                g->C + m*g->ldc + n, g->ldc, mr, nr);
        }
    }
}

#endif /* __AVX512F__ */

/* ============================================================
 * AVX2 path: MR=6, NR=16 (direct, no B packing)
 * ============================================================ */

#ifdef __AVX2__

static inline void ukernel_6x16_avx2(
    int K,
    const float* restrict A, int lda,
    const float* restrict B, int ldb,
    float* restrict C, int ldc,
    int mr, int nr)
{
    __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps();
    __m256 c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
    __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps();
    __m256 c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
    __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps();
    __m256 c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();

    const float *a0=A, *a1=A, *a2=A, *a3=A, *a4=A, *a5=A;
    if (mr>1) a1=A+lda; if (mr>2) a2=A+2*lda; if (mr>3) a3=A+3*lda;
    if (mr>4) a4=A+4*lda; if (mr>5) a5=A+5*lda;

    int k=0;
    for (; k+3<K; k+=4) {
        #define I256(off) { \
            const float*bp=B+(k+(off))*ldb; \
            if ((off)==0 && k+4<K) _mm_prefetch((const char*)(B+(k+4)*ldb), _MM_HINT_T0); \
            __m256 b0=_mm256_loadu_ps(bp), b1=_mm256_loadu_ps(bp+8); \
            __m256 va; \
            va=_mm256_broadcast_ss(a0+k+(off));c00=_mm256_fmadd_ps(va,b0,c00);c01=_mm256_fmadd_ps(va,b1,c01); \
            va=_mm256_broadcast_ss(a1+k+(off));c10=_mm256_fmadd_ps(va,b0,c10);c11=_mm256_fmadd_ps(va,b1,c11); \
            va=_mm256_broadcast_ss(a2+k+(off));c20=_mm256_fmadd_ps(va,b0,c20);c21=_mm256_fmadd_ps(va,b1,c21); \
            va=_mm256_broadcast_ss(a3+k+(off));c30=_mm256_fmadd_ps(va,b0,c30);c31=_mm256_fmadd_ps(va,b1,c31); \
            va=_mm256_broadcast_ss(a4+k+(off));c40=_mm256_fmadd_ps(va,b0,c40);c41=_mm256_fmadd_ps(va,b1,c41); \
            va=_mm256_broadcast_ss(a5+k+(off));c50=_mm256_fmadd_ps(va,b0,c50);c51=_mm256_fmadd_ps(va,b1,c51); \
        }
        I256(0);I256(1);I256(2);I256(3);
        #undef I256
    }
    for (; k<K; k++) {
        const float*bp=B+k*ldb;
        __m256 b0=_mm256_loadu_ps(bp), b1=_mm256_loadu_ps(bp+8);
        __m256 va;
        va=_mm256_broadcast_ss(a0+k);c00=_mm256_fmadd_ps(va,b0,c00);c01=_mm256_fmadd_ps(va,b1,c01);
        va=_mm256_broadcast_ss(a1+k);c10=_mm256_fmadd_ps(va,b0,c10);c11=_mm256_fmadd_ps(va,b1,c11);
        va=_mm256_broadcast_ss(a2+k);c20=_mm256_fmadd_ps(va,b0,c20);c21=_mm256_fmadd_ps(va,b1,c21);
        va=_mm256_broadcast_ss(a3+k);c30=_mm256_fmadd_ps(va,b0,c30);c31=_mm256_fmadd_ps(va,b1,c31);
        va=_mm256_broadcast_ss(a4+k);c40=_mm256_fmadd_ps(va,b0,c40);c41=_mm256_fmadd_ps(va,b1,c41);
        va=_mm256_broadcast_ss(a5+k);c50=_mm256_fmadd_ps(va,b0,c50);c51=_mm256_fmadd_ps(va,b1,c51);
    }

    if (nr >= 16) {
        #define W256(i,lo,hi) if(i<mr){_mm256_storeu_ps(C+(i)*ldc,lo);_mm256_storeu_ps(C+(i)*ldc+8,hi);}
        W256(0,c00,c01);W256(1,c10,c11);W256(2,c20,c21);
        W256(3,c30,c31);W256(4,c40,c41);W256(5,c50,c51);
        #undef W256
    } else {
        float tmp[6*16] __attribute__((aligned(32)));
        _mm256_store_ps(tmp,c00);_mm256_store_ps(tmp+8,c01);
        _mm256_store_ps(tmp+16,c10);_mm256_store_ps(tmp+24,c11);
        _mm256_store_ps(tmp+32,c20);_mm256_store_ps(tmp+40,c21);
        _mm256_store_ps(tmp+48,c30);_mm256_store_ps(tmp+56,c31);
        _mm256_store_ps(tmp+64,c40);_mm256_store_ps(tmp+72,c41);
        _mm256_store_ps(tmp+80,c50);_mm256_store_ps(tmp+88,c51);
        for (int i=0;i<mr;i++) for (int j=0;j<nr;j++) C[i*ldc+j]=tmp[i*16+j];
    }
}

static void gemm_avx2_worker(void* arg, int start, int end)
{
    GemmCtx* g = (GemmCtx*)arg;
    for (int ni = start; ni < end; ni++) {
        int n = ni * NN2_NR;
        int nr = (n + NN2_NR <= g->N) ? NN2_NR : g->N - n;
        for (int m = 0; m < g->M; m += NN2_MR) {
            int mr = (m + NN2_MR <= g->M) ? NN2_MR : g->M - m;
            ukernel_6x16_avx2(g->K, g->A + m*g->lda, g->lda,
                              g->B + n, g->N,
                              g->C + m*g->ldc + n, g->ldc, mr, nr);
        }
    }
}

#endif /* __AVX2__ */

/* ============ A-packing ============ */

float* nn2_pack_weight_a(const float* A, int M, int K)
{
    int panels = (M + NN2_MR - 1) / NN2_MR;
    float* Ap = (float*)_aligned_malloc((size_t)panels * K * NN2_MR * sizeof(float), 64);
    if (!Ap) return NULL;
    for (int p = 0; p < panels; p++) {
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < NN2_MR; i++) {
                int row = p * NN2_MR + i;
                Ap[(p * K + k) * NN2_MR + i] = (row < M) ? A[row * K + k] : 0.0f;
            }
        }
    }
    return Ap;
}

/* ============ Scalar fallback ============ */

/* ============================================================
 * Fused GEMM: C = A×B + bias[row] + activation
 * Eliminates separate bias+activation pass (saves one full C read+write)
 * ============================================================ */

typedef struct {
    int M, K, N;
    const float* A; int lda;
    const float* B;
    float* C; int ldc;
    const float* bias;
    int act;
} GemmFusedCtx;

#ifdef __AVX512F__

#include "fast_exp.h"
#include "gemm_asm.h"

static inline __m512 _fused_silu512(__m512 x) {
    return _mm512_silu_fast(x);
}

/* KC block size: keeps B panel (KC×NR×4 bytes) in L1 cache */
#define KC_BLOCK 256

static void gemm_fused_avx512_worker(void* arg, int start, int end)
{
    GemmFusedCtx* g = (GemmFusedCtx*)arg;
    for (int ni = start; ni < end; ni++) {
        int n = ni * NR512;
        int nr = (n + NR512 <= g->N) ? NR512 : g->N - n;
        for (int m = 0; m < g->M; m += NN2_MR) {
            int mr = (m + NN2_MR <= g->M) ? NN2_MR : g->M - m;

            __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
            __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
            __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
            __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
            __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
            __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();

            const float*A=g->A+m*g->lda;
            const float*a0=A,*a1=A,*a2=A,*a3=A,*a4=A,*a5=A;
            if(mr>1)a1=A+g->lda;if(mr>2)a2=A+2*g->lda;if(mr>3)a3=A+3*g->lda;
            if(mr>4)a4=A+4*g->lda;if(mr>5)a5=A+5*g->lda;

            /* KC-blocked K-loop with unroll by 4 */
            for (int kc = 0; kc < g->K; kc += KC_BLOCK) {
            int kb = (kc + KC_BLOCK <= g->K) ? KC_BLOCK : g->K - kc;
            int k = kc;
            int kend4 = kc + (kb & ~3);
            #define FK(off) { \
                const float*bp=g->B+(k+(off))*g->N+n; \
                __m512 b0=_mm512_loadu_ps(bp),b1=_mm512_loadu_ps(bp+16); __m512 va; \
                va=_mm512_set1_ps(a0[k+(off)]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01); \
                va=_mm512_set1_ps(a1[k+(off)]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11); \
                va=_mm512_set1_ps(a2[k+(off)]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21); \
                va=_mm512_set1_ps(a3[k+(off)]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31); \
                va=_mm512_set1_ps(a4[k+(off)]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41); \
                va=_mm512_set1_ps(a5[k+(off)]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51); }
            for(;k<kend4;k+=4) { FK(0);FK(1);FK(2);FK(3); }
            for(;k<kc+kb;k++) { FK(0); }
            #undef FK
            }

            float*C=g->C+m*g->ldc+n;
            #define FS(i,lo,hi) if(i<mr){ \
                __m512 vb=_mm512_set1_ps(g->bias[m+(i)]); \
                lo=_mm512_add_ps(lo,vb);hi=_mm512_add_ps(hi,vb); \
                if(g->act==1){lo=_fused_silu512(lo);hi=_fused_silu512(hi);} \
                else if(g->act==3){__m512 vl=_mm512_set1_ps(0.1f),vz=_mm512_setzero_ps(); \
                    lo=_mm512_mask_mul_ps(lo,_mm512_cmplt_ps_mask(lo,vz),lo,vl); \
                    hi=_mm512_mask_mul_ps(hi,_mm512_cmplt_ps_mask(hi,vz),hi,vl);} \
                if(nr>=32){_mm512_storeu_ps(C+i*g->ldc,lo);_mm512_storeu_ps(C+i*g->ldc+16,hi);} \
                else{__mmask16 ml=(nr>=16)?0xFFFF:((__mmask16)1<<nr)-1,mh=(nr<=16)?0:((__mmask16)1<<(nr-16))-1; \
                    _mm512_mask_storeu_ps(C+i*g->ldc,ml,lo);if(mh)_mm512_mask_storeu_ps(C+i*g->ldc+16,mh,hi);} \
            }
            FS(0,c00,c01);FS(1,c10,c11);FS(2,c20,c21);FS(3,c30,c31);FS(4,c40,c41);FS(5,c50,c51);
            #undef FS
        }
    }
}
#endif /* __AVX512F__ */

#ifdef __AVX512F__
typedef struct { int M,K,N; const float*Ap; const float*B; float*C; int ldc; const float*bias; int act; } PackedCtx;

/* KC-blocked packed fused worker.
 * Split K into KC-sized chunks so B panel (KC×NR=KC×32 floats = KC×128 bytes)
 * stays in L1 cache (~48KB). KC=256 → 32KB B panel per tile. */
static void packed_fused_worker(void* arg, int start, int end) {
    PackedCtx* g = (PackedCtx*)arg;
        for (int ni = start; ni < end; ni++) {
            int n = ni * NR512;
            int nr = (n + NR512 <= g->N) ? NR512 : g->N - n;
            int panels = (g->M + NN2_MR - 1) / NN2_MR;
            for (int p = 0; p < panels; p++) {
                int mr = ((p+1)*NN2_MR <= g->M) ? NN2_MR : g->M - p*NN2_MR;
                __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
                __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
                __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
                __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
                __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
                __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();
                const float* ap_base = g->Ap + p * g->K * NN2_MR;

                /* KC-blocked K loop */
                for (int kc = 0; kc < g->K; kc += KC_BLOCK) {
                    int kb = (kc + KC_BLOCK <= g->K) ? KC_BLOCK : g->K - kc;
                    const float* ap = ap_base + kc * NN2_MR;
                    const float* bp_base = g->B + kc * g->N + n;
                    int k = 0;
                    /* Unroll by 4 within KC block, prefetch B+2 rows ahead */
                    for (; k + 3 < kb; k += 4) {
                        if (k + 6 < kb) _mm_prefetch((const char*)(bp_base + (k+6) * g->N), _MM_HINT_T0);
                        #define KCF(off) { \
                            const float* bp = bp_base + (k+(off)) * g->N; \
                            const float* a = ap + (k+(off)) * NN2_MR; \
                            __m512 b0=_mm512_loadu_ps(bp), b1=_mm512_loadu_ps(bp+16); __m512 va; \
                            va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01); \
                            va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11); \
                            va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21); \
                            va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31); \
                            va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41); \
                            va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51); }
                        KCF(0);KCF(1);KCF(2);KCF(3);
                        #undef KCF
                    }
                    for (; k < kb; k++) {
                        const float* bp = bp_base + k * g->N;
                        const float* a = ap + k * NN2_MR;
                        __m512 b0=_mm512_loadu_ps(bp), b1=_mm512_loadu_ps(bp+16);
                        __m512 va;
                        va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01);
                        va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11);
                        va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21);
                        va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31);
                        va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41);
                        va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51);
                    }
                }

                int m = p * NN2_MR;
                float* Cp = g->C + m * g->ldc + n;
                #define FP(i,lo,hi) if(i<mr){ \
                    __m512 vb=_mm512_set1_ps(g->bias[m+(i)]); \
                    lo=_mm512_add_ps(lo,vb);hi=_mm512_add_ps(hi,vb); \
                    if(g->act==1){lo=_fused_silu512(lo);hi=_fused_silu512(hi);} \
                    if(nr>=32){_mm512_storeu_ps(Cp+i*g->ldc,lo);_mm512_storeu_ps(Cp+i*g->ldc+16,hi);} \
                    else{__mmask16 ml=(nr>=16)?0xFFFF:((__mmask16)1<<nr)-1,mh=(nr<=16)?0:((__mmask16)1<<(nr-16))-1; \
                        _mm512_mask_storeu_ps(Cp+i*g->ldc,ml,lo);if(mh)_mm512_mask_storeu_ps(Cp+i*g->ldc+16,mh,hi);}}
                FP(0,c00,c01);FP(1,c10,c11);FP(2,c20,c21);FP(3,c30,c31);FP(4,c40,c41);FP(5,c50,c51);
                #undef FP
            }
        }
}
#endif /* __AVX512F__ */

/* 2D-tiled packed fused worker: tile index → (m_panel, n_tile) */
typedef struct { int M,K,N; const float*Ap; const float*B; float*C; int ldc; const float*bias; int act; int n_tiles; } Packed2DCtx;

static void packed_fused_2d_worker(void* arg, int start, int end) {
    Packed2DCtx* g = (Packed2DCtx*)arg;
    int panels = (g->M + NN2_MR - 1) / NN2_MR;
    for (int ti = start; ti < end; ti++) {
        int ni = ti % g->n_tiles;
        int p = ti / g->n_tiles;
        int n = ni * NR512;
        int nr = (n + NR512 <= g->N) ? NR512 : g->N - n;
        int mr = ((p+1)*NN2_MR <= g->M) ? NN2_MR : g->M - p*NN2_MR;

        __m512 c00=_mm512_setzero_ps(),c01=_mm512_setzero_ps();
        __m512 c10=_mm512_setzero_ps(),c11=_mm512_setzero_ps();
        __m512 c20=_mm512_setzero_ps(),c21=_mm512_setzero_ps();
        __m512 c30=_mm512_setzero_ps(),c31=_mm512_setzero_ps();
        __m512 c40=_mm512_setzero_ps(),c41=_mm512_setzero_ps();
        __m512 c50=_mm512_setzero_ps(),c51=_mm512_setzero_ps();
        const float* ap_base = g->Ap + p * g->K * NN2_MR;
        for (int kc = 0; kc < g->K; kc += KC_BLOCK) {
            int kb = (kc + KC_BLOCK <= g->K) ? KC_BLOCK : g->K - kc;
            const float* ap = ap_base + kc * NN2_MR;
            const float* bp_base = g->B + kc * g->N + n;
            int k = 0;
            for (; k + 3 < kb; k += 4) {
                #define KCF2(off) { \
                    const float* bp = bp_base + (k+(off)) * g->N; \
                    const float* a = ap + (k+(off)) * NN2_MR; \
                    __m512 b0=_mm512_loadu_ps(bp), b1=_mm512_loadu_ps(bp+16); __m512 va; \
                    va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01); \
                    va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11); \
                    va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21); \
                    va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31); \
                    va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41); \
                    va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51); }
                KCF2(0);KCF2(1);KCF2(2);KCF2(3);
                #undef KCF2
            }
            for (; k < kb; k++) {
                const float* bp = bp_base + k * g->N;
                const float* a = ap + k * NN2_MR;
                __m512 b0=_mm512_loadu_ps(bp), b1=_mm512_loadu_ps(bp+16); __m512 va;
                va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01);
                va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11);
                va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21);
                va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31);
                va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41);
                va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51);
            }
        }
        int m = p * NN2_MR;
        float* Cp = g->C + m * g->ldc + n;
        #define FP2(i,lo,hi) if(i<mr){ \
            __m512 vb=_mm512_set1_ps(g->bias[m+(i)]); \
            lo=_mm512_add_ps(lo,vb);hi=_mm512_add_ps(hi,vb); \
            if(g->act==1){lo=_fused_silu512(lo);hi=_fused_silu512(hi);} \
            if(nr>=32){_mm512_storeu_ps(Cp+i*g->ldc,lo);_mm512_storeu_ps(Cp+i*g->ldc+16,hi);} \
            else{__mmask16 ml=(nr>=16)?0xFFFF:((__mmask16)1<<nr)-1,mh=(nr<=16)?0:((__mmask16)1<<(nr-16))-1; \
                _mm512_mask_storeu_ps(Cp+i*g->ldc,ml,lo);if(mh)_mm512_mask_storeu_ps(Cp+i*g->ldc+16,mh,hi);}}
        FP2(0,c00,c01);FP2(1,c10,c11);FP2(2,c20,c21);FP2(3,c30,c31);FP2(4,c40,c41);FP2(5,c50,c51);
        #undef FP2
    }
}

void nn2_sgemm_bias_act_packed(int M, int K, int N,
                               const float* Ap,
                               const float* B, float* C, int ldc,
                               const float* bias, int act)
{
#ifdef __AVX512F__
    int n_tiles = (N + NR512 - 1) / NR512;
    int m_panels = (M + NN2_MR - 1) / NN2_MR;
    int total_2d = m_panels * n_tiles;
    /* Use 2D tiling when N-tiles alone aren't enough to fill threads */
    if (n_tiles <= 1) {
        PackedCtx pg = {M, K, N, Ap, B, C, ldc, bias, act};
        packed_fused_worker(&pg, 0, n_tiles);
    } else if (n_tiles < nn2_tp_num_threads() && total_2d >= 4) {
        Packed2DCtx pg = {M, K, N, Ap, B, C, ldc, bias, act, n_tiles};
        int grain = (total_2d > 64) ? total_2d / (nn2_tp_num_threads() * 4) : 1;
        if (grain < 1) grain = 1;
        nn2_tp_parallel_for(packed_fused_2d_worker, &pg, total_2d, grain);
    } else {
        PackedCtx pg = {M, K, N, Ap, B, C, ldc, bias, act};
        int grain = (n_tiles > 128) ? n_tiles / (nn2_tp_num_threads() * 4) : 1;
        if (grain < 1) grain = 1;
        nn2_tp_parallel_for(packed_fused_worker, &pg, n_tiles, grain);
    }
#else
    (void)Ap;(void)B;(void)C;(void)ldc;(void)bias;(void)act;(void)M;(void)K;(void)N;
#endif
}

void nn2_sgemm_bias_act(int M, int K, int N,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float* C, int ldc,
                        const float* bias, int act)
{
    (void)ldb;
#ifdef __AVX512F__
    GemmFusedCtx g = {M, K, N, A, lda, B, C, ldc, bias, act};
    int n_tiles = (N + NR512 - 1) / NR512;
    int grain = (n_tiles > 128) ? n_tiles / (nn2_tp_num_threads() * 4) : 1;
    if (grain < 1) grain = 1;
    if (n_tiles >= 2)
        nn2_tp_parallel_for(gemm_fused_avx512_worker, &g, n_tiles, grain);
    else
        gemm_fused_avx512_worker(&g, 0, n_tiles);
#else
    /* Fallback: regular GEMM + separate bias+act */
    nn2_sgemm(M, K, N, A, lda, B, ldb, C, ldc);
    for (int m = 0; m < M; m++) {
        float b = bias[m];
        for (int n = 0; n < N; n++) {
            float v = C[m*ldc+n] + b;
            if (act == 1) { float s = 1.0f/(1.0f+expf(-v)); v *= s; }
            else if (act == 3 && v < 0) v *= 0.1f;
            C[m*ldc+n] = v;
        }
    }
#endif
}

/* ============ Scalar fallback ============ */

static void gemm_scalar(int M, int K, int N,
                        const float* A, int lda,
                        const float* B, float* C, int ldc)
{
    for (int m=0;m<M;m++)
        for (int n=0;n<N;n++) {
            float acc=0;
            for (int k=0;k<K;k++) acc+=A[m*lda+k]*B[k*N+n];
            C[m*ldc+n]=acc;
        }
}

/* ============ Public GEMM ============ */

void nn2_sgemm(int M, int K, int N,
               const float* A, int lda,
               const float* B, int ldb,
               float* C, int ldc)
{
    (void)ldb;

#ifdef __AVX512F__
    {
        GemmCtx g = {M, K, N, A, lda, B, C, ldc};
        int n_tiles = (N + NR512 - 1) / NR512;
        int grain = (n_tiles > 128) ? n_tiles / (nn2_tp_num_threads() * 4) : 1;
        if (grain < 1) grain = 1;
        if (n_tiles >= 2 && N >= 32)
            nn2_tp_parallel_for(gemm_avx512_worker, &g, n_tiles, grain);
        else
            gemm_avx512_worker(&g, 0, n_tiles);
        return;
    }
#endif

#ifdef __AVX2__
    {
        GemmCtx g = {M, K, N, A, lda, B, C, ldc};
        int n_tiles = (N + NN2_NR - 1) / NN2_NR;
        int grain = (n_tiles > 256) ? n_tiles / (nn2_tp_num_threads() * 4) : 1;
        if (grain < 1) grain = 1;
        if (n_tiles >= 2 && N >= 32)
            nn2_tp_parallel_for(gemm_avx2_worker, &g, n_tiles, grain);
        else
            gemm_avx2_worker(&g, 0, n_tiles);
        return;
    }
#endif

    gemm_scalar(M, K, N, A, lda, B, C, ldc);
}
