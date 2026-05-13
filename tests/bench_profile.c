/*
 * bench_profile.c — Profile FaceX embedding: GEMM vs other ops.
 * Measures total time and GEMM-only time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "facex.h"
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart * 1000.0;
}
#endif

/* Hook into matmul to count calls and measure time */
static volatile int64_t g_gemm_calls = 0;
static volatile double g_gemm_total_ms = 0;

/* We can't easily hook, so let's measure differently:
 * Run with INT8 ON, then OFF, difference = GEMM overhead */

int main(int argc, char** argv) {
    const char* weights = argc > 1 ? argv[1] : "data/edgeface_xs_fp32.bin";
    int loops = 100;

    printf("FaceX Profile: GEMM vs Total\n\n");

    float input[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        input[i] = (float)(i % 256) / 128.0f - 1.0f;
    float emb[512];

    /* Run and measure */
    FaceX* fx = facex_init(weights, NULL, NULL);
    if (!fx) { fprintf(stderr, "Cannot load\n"); return 1; }

    /* Warmup */
    for (int i = 0; i < 10; i++) facex_embed(fx, input, emb);

    /* Detailed timing: run many iterations, measure min/avg */
    double best = 1e9, total = 0;
    for (int i = 0; i < loops; i++) {
        double t0 = now_ms();
        facex_embed(fx, input, emb);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        total += dt;
    }

    printf("With INT8 (current):\n");
    printf("  best:  %.3f ms\n", best);
    printf("  avg:   %.3f ms\n", total / loops);

    /* Measure individual operations by running benchmark code */
    /* Test: how fast is just the quantize + GEMM portion? */
    extern void matmul_dynamic_int8(const float*, int, int, int,
                                     const void*, const int32_t*, const float*, float*);

    /* Use the PackedMM info for a representative layer (stage 2: K=100, N=400) */
    printf("\n  Micro-benchmark: matmul sizes typical in EdgeFace:\n");

    /* Just measure raw FP32 vs INT8 GEMM for known sizes */
    extern void matmul_fp32(const float*, const float*, float*, int, int, int);

    /* M=49, K=100, N=400 (stage 2 main layer) */
    int M=49, K=100, N=400;
    float *A = (float*)malloc(M*K*4), *B = (float*)malloc(K*N*4), *C = (float*)malloc(M*N*4);
    for (int i=0; i<M*K; i++) A[i] = 0.01f * (i%100);
    for (int i=0; i<K*N; i++) B[i] = 0.01f * (i%100);

    /* FP32 */
    double best_fp=1e9;
    for (int i=0; i<1000; i++) {
        double t0 = now_ms();
        matmul_fp32(A, B, C, M, K, N);
        double dt = now_ms() - t0;
        if (dt < best_fp) best_fp = dt;
    }
    printf("    FP32 matmul %dx%dx%d: %.3f ms\n", M, K, N, best_fp);

    /* Large layer: M=49, K=192, N=768 */
    M=9; K=192; N=768;
    free(A); free(B); free(C);
    A = (float*)malloc(M*K*4); B = (float*)malloc(K*N*4); C = (float*)malloc(M*N*4);
    for (int i=0; i<M*K; i++) A[i] = 0.01f * (i%100);
    for (int i=0; i<K*N; i++) B[i] = 0.01f * (i%100);
    best_fp = 1e9;
    for (int i=0; i<1000; i++) {
        double t0 = now_ms();
        matmul_fp32(A, B, C, M, K, N);
        double dt = now_ms() - t0;
        if (dt < best_fp) best_fp = dt;
    }
    printf("    FP32 matmul %dx%dx%d: %.3f ms\n", M, K, N, best_fp);

    /* INT8 GEMM micro-benchmark (including quantize overhead) */
    extern void pack_weights_4x8c8(const int8_t*, const float*, int, int, void*, int32_t*);
    extern int packed_weights_size_4x8c8(int, int);
    extern void int8_gemm_4x8c8(const int8_t*, int, int, int,
                                  const void*, int32_t*, const int32_t*);

    M=49; K=100; N=400;
    A = (float*)malloc(M*K*4); B = (float*)malloc(K*N*4); C = (float*)malloc(M*N*4);
    for (int i=0; i<M*K; i++) A[i] = 0.01f * (i%100 - 50);
    for (int i=0; i<K*N; i++) B[i] = 0.01f * (i%100 - 50);

    /* Quantize B (one-time) */
    int8_t* B_int8 = (int8_t*)malloc(N*K);
    float* w_sc = (float*)malloc(N*4);
    for (int n=0; n<N; n++) {
        float mx=0;
        for (int k=0; k<K; k++) { float v=B[k*N+n]; if(v<0)v=-v; if(v>mx)mx=v; }
        w_sc[n] = mx/127.0f;
        for (int k=0; k<K; k++) {
            int q = (int)(B[k*N+n]/(w_sc[n]+1e-9f));
            if(q>127)q=127; if(q<-128)q=-128;
            B_int8[n*K+k]=(int8_t)q;
        }
    }
    int pw_sz = packed_weights_size_4x8c8(K, N);
    void* pw = malloc(pw_sz);
    int32_t* cs = (int32_t*)calloc((N+7)&~7, 4);
    pack_weights_4x8c8(B_int8, NULL, K, N, pw, cs);

    /* Quantize A (one-time) */
    int K_pad = (K+7)&~7;
    int8_t* A_i8 = (int8_t*)calloc(M*K_pad,1);
    float a_max=0;
    for(int i=0;i<M*K;i++){float v=A[i];if(v<0)v=-v;if(v>a_max)a_max=v;}
    float a_sc=a_max/127.0f, a_inv=1.0f/a_sc;
    for(int m=0;m<M;m++) for(int k=0;k<K;k++) {
        int q=(int)(A[m*K+k]*a_inv); if(q>127)q=127; if(q<-128)q=-128;
        A_i8[m*K_pad+k]=(int8_t)q;
    }
    int32_t* C_i32 = (int32_t*)calloc(M*N,4);

    /* Pure INT8 GEMM (no quantize overhead) */
    double best_i8=1e9;
    for (int i=0; i<1000; i++) {
        double t0 = now_ms();
        int8_gemm_4x8c8(A_i8, M, K_pad, N, pw, C_i32, cs);
        double dt = now_ms() - t0;
        if (dt < best_i8) best_i8 = dt;
    }
    printf("    INT8 GEMM  %dx%dx%d: %.3f ms (pure, no quant overhead)\n", M, K, N, best_i8);

    /* Dynamic INT8 (with quantize) */
    double best_dyn=1e9;
    for (int i=0; i<1000; i++) {
        double t0 = now_ms();
        matmul_dynamic_int8(A, M, K, N, pw, cs, w_sc, C);
        double dt = now_ms() - t0;
        if (dt < best_dyn) best_dyn = dt;
    }
    printf("    INT8 dyn   %dx%dx%d: %.3f ms (with quant overhead)\n", M, K, N, best_dyn);
    printf("    Quant overhead: %.3f ms (%.0f%%)\n", best_dyn - best_i8,
           (best_dyn - best_i8) / best_dyn * 100);
    printf("    INT8/FP32 pure GEMM ratio: %.2fx\n", best_fp / best_i8);

    /* FP32 packed (the one actually used in engine) */
    extern void pack_b_fp32(const float*, int, int, float*);
    extern void matmul_fp32_packed(const float*, const float*, float*, int, int, int);
    /* Packed size: ceil(N/NR) panels × K × NR floats. NR=16 for AVX-512 */
    int NR = 16; /* AVX-512 */
    int pb_sz = ((N + NR - 1) / NR) * K * NR * (int)sizeof(float);
    float* pb = (float*)malloc(pb_sz);
    pack_b_fp32(B, K, N, pb);
    double best_fp_pk = 1e9;
    for (int i=0; i<1000; i++) {
        double t0 = now_ms();
        matmul_fp32_packed(A, pb, C, M, K, N);
        double dt = now_ms() - t0;
        if (dt < best_fp_pk) best_fp_pk = dt;
    }
    printf("    FP32 packed %dx%dx%d: %.3f ms\n", M, K, N, best_fp_pk);
    printf("    INT8/FP32-packed ratio: %.2fx\n", best_fp_pk / best_i8);
    free(pb);

    free(A); free(B); free(C); free(B_int8); free(w_sc); free(pw); free(cs);
    free(A_i8); free(C_i32);
    facex_free(fx);
    return 0;
}
