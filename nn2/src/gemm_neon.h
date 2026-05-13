/*
 * gemm_neon.h — ARM NEON FP32 GEMM + SiLU for AArch64.
 *
 * Microkernel: MR=8, NR=4 (8 rows × 4 cols = 8 accumulators).
 * NEON: 128-bit, 4 floats per register, 32 registers available.
 * 8 acc + 1 B load + 8 A broadcasts = 17 registers. Fits.
 *
 * For Raspberry Pi 4 (Cortex-A72): ~12 GFLOPS FP32 peak.
 * For Rockchip RK3588 (Cortex-A76): ~50 GFLOPS FP32 peak.
 */

#ifndef GEMM_NEON_H
#define GEMM_NEON_H

#ifdef __aarch64__
#include <arm_neon.h>

#define NEON_MR 8
#define NEON_NR 4

/* ============ Cephes exp for NEON ============ */

static inline float32x4_t neon_exp_fast(float32x4_t x)
{
    const float32x4_t log2e  = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693145751953125f);
    const float32x4_t ln2_lo = vdupq_n_f32(1.42860682030941723212e-6f);

    /* Clamp */
    x = vmaxq_f32(x, vdupq_n_f32(-87.3f));
    x = vminq_f32(x, vdupq_n_f32(88.7f));

    /* n = round(x / ln2) */
    float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));

    /* f = x - n * ln2 */
    float32x4_t f = vfmsq_f32(x, n, ln2_hi);
    f = vfmsq_f32(f, n, ln2_lo);

    /* Polynomial: 1 + f + f²/2 + f³/6 + ... */
    float32x4_t p = vdupq_n_f32(0.00138888889039885e0f);
    p = vfmaq_f32(vdupq_n_f32(0.00833333333371142972e0f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.0416666666665065678e0f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.166666666666666019037f), p, f);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);

    /* 2^n via bit manipulation */
    int32x4_t ni = vcvtq_s32_f32(n);
    ni = vaddq_s32(ni, vdupq_n_s32(127));
    ni = vshlq_n_s32(ni, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(ni);

    return vmulq_f32(p, pow2n);
}

static inline float32x4_t neon_sigmoid(float32x4_t x)
{
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t neg_x = vnegq_f32(x);
    return vdivq_f32(one, vaddq_f32(one, neon_exp_fast(neg_x)));
}

static inline float32x4_t neon_silu(float32x4_t x)
{
    return vmulq_f32(x, neon_sigmoid(x));
}

/* ============ GEMM 8×4 microkernel ============ */

static inline void neon_ukernel_8x4(
    int K,
    const float* restrict A, int lda,
    const float* restrict B, int ldb,
    float* restrict C, int ldc,
    int mr)
{
    float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0);
    float32x4_t c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);
    float32x4_t c4 = vdupq_n_f32(0), c5 = vdupq_n_f32(0);
    float32x4_t c6 = vdupq_n_f32(0), c7 = vdupq_n_f32(0);

    const float *a0=A, *a1=A, *a2=A, *a3=A, *a4=A, *a5=A, *a6=A, *a7=A;
    if(mr>1) a1=A+lda; if(mr>2) a2=A+2*lda; if(mr>3) a3=A+3*lda;
    if(mr>4) a4=A+4*lda; if(mr>5) a5=A+5*lda;
    if(mr>6) a6=A+6*lda; if(mr>7) a7=A+7*lda;

    for (int k = 0; k < K; k++) {
        float32x4_t b = vld1q_f32(B + k * ldb);
        c0 = vfmaq_n_f32(c0, b, a0[k]);
        c1 = vfmaq_n_f32(c1, b, a1[k]);
        c2 = vfmaq_n_f32(c2, b, a2[k]);
        c3 = vfmaq_n_f32(c3, b, a3[k]);
        c4 = vfmaq_n_f32(c4, b, a4[k]);
        c5 = vfmaq_n_f32(c5, b, a5[k]);
        c6 = vfmaq_n_f32(c6, b, a6[k]);
        c7 = vfmaq_n_f32(c7, b, a7[k]);
    }

    if(mr>0) vst1q_f32(C + 0*ldc, c0);
    if(mr>1) vst1q_f32(C + 1*ldc, c1);
    if(mr>2) vst1q_f32(C + 2*ldc, c2);
    if(mr>3) vst1q_f32(C + 3*ldc, c3);
    if(mr>4) vst1q_f32(C + 4*ldc, c4);
    if(mr>5) vst1q_f32(C + 5*ldc, c5);
    if(mr>6) vst1q_f32(C + 6*ldc, c6);
    if(mr>7) vst1q_f32(C + 7*ldc, c7);
}

/* ============ Fused GEMM + bias + SiLU for NEON ============ */

static inline void neon_ukernel_8x4_fused(
    int K,
    const float* restrict A, int lda,
    const float* restrict B, int ldb,
    float* restrict C, int ldc,
    const float* bias, int m_start, int mr, int act)
{
    float32x4_t c0={0},c1={0},c2={0},c3={0},c4={0},c5={0},c6={0},c7={0};
    const float *a0=A, *a1=A, *a2=A, *a3=A, *a4=A, *a5=A, *a6=A, *a7=A;
    if(mr>1) a1=A+lda; if(mr>2) a2=A+2*lda; if(mr>3) a3=A+3*lda;
    if(mr>4) a4=A+4*lda; if(mr>5) a5=A+5*lda;
    if(mr>6) a6=A+6*lda; if(mr>7) a7=A+7*lda;

    for (int k = 0; k < K; k++) {
        float32x4_t b = vld1q_f32(B + k * ldb);
        c0=vfmaq_n_f32(c0,b,a0[k]); c1=vfmaq_n_f32(c1,b,a1[k]);
        c2=vfmaq_n_f32(c2,b,a2[k]); c3=vfmaq_n_f32(c3,b,a3[k]);
        c4=vfmaq_n_f32(c4,b,a4[k]); c5=vfmaq_n_f32(c5,b,a5[k]);
        c6=vfmaq_n_f32(c6,b,a6[k]); c7=vfmaq_n_f32(c7,b,a7[k]);
    }

    /* Fused: bias + activation + store */
    #define NEON_STORE(i, acc) if(i<mr) { \
        acc = vaddq_f32(acc, vdupq_n_f32(bias[m_start+(i)])); \
        if (act == 1) acc = neon_silu(acc); \
        else if (act == 3) { \
            uint32x4_t neg = vcltq_f32(acc, vdupq_n_f32(0)); \
            acc = vbslq_f32(neg, vmulq_n_f32(acc, 0.1f), acc); } \
        vst1q_f32(C + (i)*ldc, acc); }
    NEON_STORE(0,c0); NEON_STORE(1,c1); NEON_STORE(2,c2); NEON_STORE(3,c3);
    NEON_STORE(4,c4); NEON_STORE(5,c5); NEON_STORE(6,c6); NEON_STORE(7,c7);
    #undef NEON_STORE
}

/* ============ Depthwise conv 3×3 NEON ============ */

static inline void neon_dwconv3x3(
    const float* src, const float* wt, float bias,
    int H, int W, int act, float* dst)
{
    float32x4_t vw[9];
    for (int i = 0; i < 9; i++) vw[i] = vdupq_n_f32(wt[i]);
    float32x4_t vb = vdupq_n_f32(bias);

    for (int oh = 0; oh < H; oh++) {
        const float* r0 = (oh > 0) ? src + (oh-1)*W : NULL;
        const float* r1 = src + oh*W;
        const float* r2 = (oh < H-1) ? src + (oh+1)*W : NULL;

        int ow = 1; /* skip first column (padding) */
        for (; ow + 4 <= W - 1; ow += 4) {
            float32x4_t sum = vb;
            if (r0) {
                sum = vfmaq_f32(sum, vld1q_f32(r0+ow-1), vw[0]);
                sum = vfmaq_f32(sum, vld1q_f32(r0+ow),   vw[1]);
                sum = vfmaq_f32(sum, vld1q_f32(r0+ow+1), vw[2]);
            }
            sum = vfmaq_f32(sum, vld1q_f32(r1+ow-1), vw[3]);
            sum = vfmaq_f32(sum, vld1q_f32(r1+ow),   vw[4]);
            sum = vfmaq_f32(sum, vld1q_f32(r1+ow+1), vw[5]);
            if (r2) {
                sum = vfmaq_f32(sum, vld1q_f32(r2+ow-1), vw[6]);
                sum = vfmaq_f32(sum, vld1q_f32(r2+ow),   vw[7]);
                sum = vfmaq_f32(sum, vld1q_f32(r2+ow+1), vw[8]);
            }
            if (act == 1) sum = neon_silu(sum);
            else if (act == 3) {
                uint32x4_t neg = vcltq_f32(sum, vdupq_n_f32(0));
                sum = vbslq_f32(neg, vmulq_n_f32(sum, 0.1f), sum);
            }
            vst1q_f32(dst + oh*W + ow, sum);
        }
        /* Scalar edges */
        for (int x = 0; x < W; x++) {
            if (x >= 1 && x < W-1 && (x-1) % 4 == 0 && x+3 < W-1) continue; /* done by NEON */
            float s = bias;
            for (int kh = 0; kh < 3; kh++) {
                int ih = oh-1+kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = x-1+kw;
                    if (iw >= 0 && iw < W) s += src[ih*W+iw] * wt[kh*3+kw];
                }
            }
            if (act == 1) { float sig = 1.0f/(1.0f+expf(-s)); s *= sig; }
            else if (act == 3 && s < 0) s *= 0.1f;
            dst[oh*W+x] = s;
        }
    }
}

#endif /* __aarch64__ */
#endif /* GEMM_NEON_H */
