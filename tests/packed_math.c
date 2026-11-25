/* Check the public packer and its consumers against direct products.
 * Non-multiple panel widths and short row tiles exercise both tails. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void pack_b_fp32(const float*, int, int, float*);
int packed_b_fp32_size(int, int);
void matmul_fp32_packed(const float*, const float*, float*, int, int, int);
void matmul_fp32_packed_bias(const float*, const float*, const float*, float*, int, int, int);
void matmul_bias_gelu_packed(const float*, const float*, const float*, float*, int, int, int);

int main(void) {
    const int widths[] = {1, 5, 8, 9, 15, 16, 17, 32, 33};
    const int inner[] = {1, 3, 8, 19};
    unsigned checked = 0, failed = 0;
    for (size_t i = 0; i < sizeof(widths)/sizeof(widths[0]); i++)
        for (size_t j = 0; j < sizeof(inner)/sizeof(inner[0]); j++)
            for (int m = 1; m <= 5; m++) {
                int k = inner[j], n = widths[i];
                float a[5*19], b[19*33], bias[33], c[5*33];
                size_t bytes = (size_t)packed_b_fp32_size(k,n) * sizeof(float);
                void* storage = malloc(bytes + 63);
                if (!storage) return 2;
                float* p = (float*)(((uintptr_t)storage + 63) & ~(uintptr_t)63);
                for (int t=0; t<m*k; t++) a[t]=(t%11-5)*0.125f;
                for (int t=0; t<k*n; t++) b[t]=(t%13-6)*0.25f;
                for (int t=0; t<n; t++) bias[t]=(t%5-2)*0.25f;
                pack_b_fp32(b,k,n,p);
                for (int op=0; op<3; op++) {
                    if (op==0) matmul_fp32_packed(a,p,c,m,k,n);
                    if (op==1) matmul_fp32_packed_bias(a,p,bias,c,m,k,n);
                    if (op==2) matmul_bias_gelu_packed(a,p,bias,c,m,k,n);
                    for (int r=0; r<m; r++) for (int col=0; col<n; col++) {
                        float ref=op ? bias[col] : 0;
                        for (int h=0; h<k; h++) ref+=a[r*k+h]*b[h*n+col];
                        if (op==2) ref*=0.5f*(1.0f+erff(ref*0.70710678118f));
                        float tol=op==2 ? 0.001f : 0.00001f;
                        if (!(fabsf(c[r*n+col]-ref)<=tol)) {
                            if (failed++==0) fprintf(stderr,"M=%d K=%d N=%d op=%d got=%g expected=%g\n",m,k,n,op,c[r*n+col],ref);
                        }
                        checked++;
                    }
                }
                free(storage);
            }
    printf("%u packed outputs, %u mismatches\n",checked,failed);
    return failed != 0;
}
