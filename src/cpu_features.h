/*
 * cpu_features.h — runtime CPU feature detection for FaceX kernels.
 *
 * Today this exists primarily to gate the SME / SME2 dispatch on Apple
 * Silicon (M4 and newer). It's structured generically so we can add other
 * runtime probes (FP16, BF16, dot-product, future extensions) without
 * touching the matmul call sites.
 *
 * Detection is cached on first call — these helpers are safe to invoke
 * from the inner loop without measurable overhead.
 *
 * Platform support:
 *   - macOS (arm64): real detection via sysctlbyname.
 *   - Linux (aarch64): TODO — getauxval(AT_HWCAP2) probe for HWCAP2_SME.
 *   - Everything else / non-arm64: all probes return 0.
 */

#ifndef FACEX_CPU_FEATURES_H
#define FACEX_CPU_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns non-zero if the host CPU supports the Arm Scalable Matrix
 * Extension (FEAT_SME). On macOS this maps to hw.optional.arm.FEAT_SME. */
int facex_has_sme(void);

/* Returns non-zero if the host CPU supports SME2 (FEAT_SME2). On Apple
 * Silicon today, M4 reports SME but not SME2; this matters for the BF16
 * outer-product extensions and a handful of higher-throughput op forms. */
int facex_has_sme2(void);

/* Permanently disables SME dispatch for this process. Used by the
 * runtime self-check when SME output diverges from the NEON reference —
 * better to keep running on NEON than to ship wrong embeddings. */
void facex_disable_sme(void);

/* Returns the streaming-vector-length hint reported by the OS, or 0
 * if not available. On Apple M4 this is currently 512 bits (16 FP32
 * lanes). Caller may use this to size scratch buffers. */
int facex_sme_vl_bits(void);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_CPU_FEATURES_H */
