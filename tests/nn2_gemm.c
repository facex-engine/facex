/* Exercise tile tails and padded row strides against a scalar oracle. */
#include "nn2_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float activation(float x, int act) {
    if (act == 1) return x / (1 + expf(-x));
    if (act == 2) return 1 / (1 + expf(-x));
    return act == 3 && x < 0 ? 0.1f * x : x;
}

static void check_shape(int n) {
    enum { m = 13, k = 17 };
    int lda = k + 2, ldb = n + 3, ldc = n + 4;
    float a[m*lda], dense_a[m*k], b[k*ldb], dense_b[k*n], bias[m];
    float c[m*ldc], res[m*ldc];
    for (int i = 0; i < m; i++) {
        bias[i] = (i - 4) * 0.12f;
        for (int j = 0; j < k; j++)
            a[i*lda+j] = dense_a[i*k+j] = ((i + j*7) % 13 - 6) * 0.1f;
    }
    for (int j = 0; j < k; j++)
        for (int i = 0; i < ldb; i++) {
            b[j*ldb+i] = ((j*3 + i) % 19 - 9) * 0.12f;
            if (i < n) dense_b[j*n+i] = b[j*ldb+i];
        }
    for (int i = 0; i < m*ldc; i++) res[i] = (i % 11 - 5) * 0.25f;
    float *packed = nn2_pack_weight_a(dense_a, m, k);
    assert(packed);
    for (int mode = 0; mode < 4; mode++) {
        for (int act = 0; act < 4; act++) {
            for (int i = 0; i < m*ldc; i++) c[i] = 123.0f;
            if (mode == 0) nn2_sgemm(m,k,n,a,lda,b,ldb,c,ldc);
            if (mode == 1) nn2_sgemm_bias_act(m,k,n,a,lda,b,ldb,c,ldc,bias,act);
            if (mode == 2) nn2_sgemm_bias_act_packed(m,k,n,packed,dense_b,c,ldc,bias,act);
            if (mode == 3) nn2_sgemm_bias_act_res_packed(m,k,n,packed,dense_b,c,ldc,bias,act,res,ldc);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < ldc; j++) {
                    if (j >= n) { assert(c[i*ldc+j] == 123.0f); continue; }
                    float x = 0;
                    for (int v = 0; v < k; v++) x += a[i*lda+v] * b[v*ldb+j];
                    if (mode) x = activation(x + bias[i], act);
                    if (mode == 3) x += res[i*ldc+j];
                    assert(fabsf(x - c[i*ldc+j]) < 3e-5f);
                }
            }
        }
    }
    free(packed);
}

int main(void) {
    int widths[] = {1, 8, 15, 16, 17, 31, 32, 33, 64, 65};
    nn2_tp_init(3);
    for (unsigned i = 0; i < sizeof(widths)/sizeof(*widths); i++) check_shape(widths[i]);
    nn2_tp_destroy();
    puts("Plain, packed and residual GEMM agree with the scalar reference");
}
