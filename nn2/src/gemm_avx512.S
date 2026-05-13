/*
 * gemm_avx512.S — Hand-written AVX-512 6×32 microkernel.
 *
 * C ABI wrapper passes all args via a single struct pointer.
 * void gemm_6x32_kern(const GemmKernArgs* args);
 *
 * struct GemmKernArgs {
 *     int K;                  // offset 0
 *     const float* a[6];     // offset 8 (8 bytes each = 48 bytes)
 *     const float* B;        // offset 56
 *     int ldb_bytes;         // offset 64
 *     float* C;              // offset 72
 *     int ldc_bytes;         // offset 80
 * };
 */

    .text
    .globl gemm_6x32_kern

gemm_6x32_kern:
    /* rcx = args pointer (Win64) */
    push    %rbx
    push    %rsi
    push    %rdi
    push    %r12
    push    %r13
    push    %r14
    push    %r15

    mov     %rcx, %rbx              /* args base */

    /* Load args from struct */
    mov     (%rbx), %eax             /* K (int, 4 bytes, but at offset 0) */
    /* Actually struct has int K at offset 0, then padding to 8 for pointers */
    mov     8(%rbx), %r10            /* a[0] */
    mov     16(%rbx), %r11           /* a[1] */
    mov     24(%rbx), %r12           /* a[2] */
    mov     32(%rbx), %r13           /* a[3] */
    mov     40(%rbx), %r14           /* a[4] */
    mov     48(%rbx), %r15           /* a[5] */
    mov     56(%rbx), %rsi           /* B */
    mov     64(%rbx), %edi           /* ldb_bytes */
    mov     72(%rbx), %r8            /* C */
    mov     80(%rbx), %r9d           /* ldc_bytes */

    /* Zero accumulators */
    vpxord  %zmm0, %zmm0, %zmm0
    vpxord  %zmm1, %zmm1, %zmm1
    vpxord  %zmm2, %zmm2, %zmm2
    vpxord  %zmm3, %zmm3, %zmm3
    vpxord  %zmm4, %zmm4, %zmm4
    vpxord  %zmm5, %zmm5, %zmm5
    vpxord  %zmm6, %zmm6, %zmm6
    vpxord  %zmm7, %zmm7, %zmm7
    vpxord  %zmm8, %zmm8, %zmm8
    vpxord  %zmm9, %zmm9, %zmm9
    vpxord  %zmm10, %zmm10, %zmm10
    vpxord  %zmm11, %zmm11, %zmm11

    /* K loop — ecx = byte offset into A rows */
    xor     %ecx, %ecx
    shl     $2, %eax                 /* K *= 4 (byte count) */

.Lloop:
    cmp     %eax, %ecx
    jge     .Ldone

    /* Load B[k][0:32] */
    vmovups (%rsi), %zmm28
    vmovups 64(%rsi), %zmm29

    /* 6 rows: broadcast A[row][k] × B[k] */
    vbroadcastss (%r10, %rcx, 1), %zmm30
    vfmadd231ps %zmm30, %zmm28, %zmm0
    vfmadd231ps %zmm30, %zmm29, %zmm1

    vbroadcastss (%r11, %rcx, 1), %zmm30
    vfmadd231ps %zmm30, %zmm28, %zmm2
    vfmadd231ps %zmm30, %zmm29, %zmm3

    vbroadcastss (%r12, %rcx, 1), %zmm30
    vfmadd231ps %zmm30, %zmm28, %zmm4
    vfmadd231ps %zmm30, %zmm29, %zmm5

    vbroadcastss (%r13, %rcx, 1), %zmm30
    vfmadd231ps %zmm30, %zmm28, %zmm6
    vfmadd231ps %zmm30, %zmm29, %zmm7

    vbroadcastss (%r14, %rcx, 1), %zmm30
    vfmadd231ps %zmm30, %zmm28, %zmm8
    vfmadd231ps %zmm30, %zmm29, %zmm9

    vbroadcastss (%r15, %rcx, 1), %zmm30
    vfmadd231ps %zmm30, %zmm28, %zmm10
    vfmadd231ps %zmm30, %zmm29, %zmm11

    /* Advance */
    movsxd  %edi, %rdx
    add     %rdx, %rsi               /* B += ldb_bytes */
    add     $4, %ecx                 /* k_offset += 4 */
    jmp     .Lloop

.Ldone:
    /* Store C */
    vmovups %zmm0, (%r8)
    vmovups %zmm1, 64(%r8)
    movsxd  %r9d, %rdx
    add     %rdx, %r8

    vmovups %zmm2, (%r8)
    vmovups %zmm3, 64(%r8)
    add     %rdx, %r8

    vmovups %zmm4, (%r8)
    vmovups %zmm5, 64(%r8)
    add     %rdx, %r8

    vmovups %zmm6, (%r8)
    vmovups %zmm7, 64(%r8)
    add     %rdx, %r8

    vmovups %zmm8, (%r8)
    vmovups %zmm9, 64(%r8)
    add     %rdx, %r8

    vmovups %zmm10, (%r8)
    vmovups %zmm11, 64(%r8)

    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rdi
    pop     %rsi
    pop     %rbx
    ret
