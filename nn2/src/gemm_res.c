/*
 * gemm_res.c — Fused GEMM + bias + activation + residual add.
 *
 * SEPARATE compilation unit to avoid icache pollution of the main GEMM.
 * result[m][n] = act(A_packed×B[m][n] + bias[m]) + residual[m][n]
 */

#include "nn2_internal.h"
#include <immintrin.h>

#ifdef __AVX512F__

#include "fast_exp.h"

#define NR512 32
#define KC_BLOCK 256

static inline __m512 _res_silu512(__m512 x) { return _mm512_silu_fast(x); }

typedef struct {
    int M, K, N;
    const float* Ap;
    const float* B;
    float* C; int ldc;
    const float* bias;
    int act;
    const float* res; int ldr;
    int n_tiles;
} ResCtx;

static void res_worker(void* arg, int start, int end) {
    ResCtx* g = (ResCtx*)arg;
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
                #define RK(off) { \
                    const float* bp = bp_base + (k+(off)) * g->N; \
                    const float* a = ap + (k+(off)) * NN2_MR; \
                    __m512 b0=_mm512_loadu_ps(bp), b1=_mm512_loadu_ps(bp+16); __m512 va; \
                    va=_mm512_set1_ps(a[0]);c00=_mm512_fmadd_ps(va,b0,c00);c01=_mm512_fmadd_ps(va,b1,c01); \
                    va=_mm512_set1_ps(a[1]);c10=_mm512_fmadd_ps(va,b0,c10);c11=_mm512_fmadd_ps(va,b1,c11); \
                    va=_mm512_set1_ps(a[2]);c20=_mm512_fmadd_ps(va,b0,c20);c21=_mm512_fmadd_ps(va,b1,c21); \
                    va=_mm512_set1_ps(a[3]);c30=_mm512_fmadd_ps(va,b0,c30);c31=_mm512_fmadd_ps(va,b1,c31); \
                    va=_mm512_set1_ps(a[4]);c40=_mm512_fmadd_ps(va,b0,c40);c41=_mm512_fmadd_ps(va,b1,c41); \
                    va=_mm512_set1_ps(a[5]);c50=_mm512_fmadd_ps(va,b0,c50);c51=_mm512_fmadd_ps(va,b1,c51); }
                RK(0); RK(1); RK(2); RK(3);
                #undef RK
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
        const float* Rp = g->res + m * g->ldr + n;
        #define FR(i,lo,hi) if(i<mr){ \
            __m512 vb=_mm512_set1_ps(g->bias[m+(i)]); \
            lo=_mm512_add_ps(lo,vb); hi=_mm512_add_ps(hi,vb); \
            if(g->act==1){lo=_res_silu512(lo);hi=_res_silu512(hi);} \
            lo=_mm512_add_ps(lo,_mm512_loadu_ps(Rp+i*g->ldr)); \
            hi=_mm512_add_ps(hi,_mm512_loadu_ps(Rp+i*g->ldr+16)); \
            if(nr>=32){_mm512_storeu_ps(Cp+i*g->ldc,lo);_mm512_storeu_ps(Cp+i*g->ldc+16,hi);} \
            else{__mmask16 ml=(nr>=16)?0xFFFF:((__mmask16)1<<nr)-1,mh=(nr<=16)?0:((__mmask16)1<<(nr-16))-1; \
                _mm512_mask_storeu_ps(Cp+i*g->ldc,ml,lo);if(mh)_mm512_mask_storeu_ps(Cp+i*g->ldc+16,mh,hi);}}
        FR(0,c00,c01);FR(1,c10,c11);FR(2,c20,c21);FR(3,c30,c31);FR(4,c40,c41);FR(5,c50,c51);
        #undef FR
    }
}

void nn2_sgemm_bias_act_res_packed(int M, int K, int N,
                                    const float* Ap, const float* B,
                                    float* C, int ldc,
                                    const float* bias, int act,
                                    const float* residual, int ldr)
{
    int n_tiles = (N + NR512 - 1) / NR512;
    int m_panels = (M + NN2_MR - 1) / NN2_MR;
    int total = m_panels * n_tiles;
    ResCtx ctx = {M, K, N, Ap, B, C, ldc, bias, act, residual, ldr, n_tiles};
    int grain = (total > 64) ? total / (nn2_tp_num_threads() * 4) : 1;
    if (grain < 1) grain = 1;
    nn2_tp_parallel_for(res_worker, &ctx, total, grain);
}

#else /* no AVX-512 */

void nn2_sgemm_bias_act_res_packed(int M, int K, int N,
                                    const float* Ap, const float* B,
                                    float* C, int ldc,
                                    const float* bias, int act,
                                    const float* residual, int ldr)
{
    nn2_sgemm_bias_act_packed(M, K, N, Ap, B, C, ldc, bias, act);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[m*ldc+n] += residual[m*ldr+n];
}

#endif
