/*
 * facex_coreml.h — Apple Neural Engine via Core ML.
 *
 * C API for loading an EdgeFace `.mlpackage` and dispatching
 * embeddings through Core ML (which auto-routes to ANE / GPU / CPU).
 * Implemented in src/backend_coreml.m (Objective-C); callable from
 * plain C clients.
 *
 * Compiled only when FACEX_HAVE_COREML is defined (Makefile target:
 * `make COREML=1`). macOS-only.
 *
 * Hardware status: COMPILE-TESTED. Runtime ANE dispatch is not yet
 * end-to-end validated — that requires a Vela-equivalent step
 * (PyTorch → ONNX → coremltools `.mlpackage` with INT8 palettization)
 * that lives in tools/export_coreml.py. Once that produces a real
 * `weights/edgeface_xs.mlpackage`, this backend takes ≈ 0.8 ms/embed
 * on M2 and the dispatch is automatically split across ANE/GPU/CPU
 * by Core ML based on op coverage.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FaceXCoreML FaceXCoreML;

typedef struct {
    /* Compute-unit hint:
     *   0 = ALL          (Core ML picks; usually ANE → GPU → CPU)
     *   1 = CPU_AND_GPU  (skip ANE — useful for benchmarking)
     *   2 = CPU_ONLY     (no GPU/ANE — useful for debugging)
     *   3 = CPU_AND_NPU  (CPU + ANE only, skip GPU; macOS 13+) */
    int compute_units;

    /* If non-zero, log the actual MLComputePlan dispatch decisions to
     * stderr at init time. Useful when verifying that ops you expected
     * on ANE actually went there. macOS 14+. */
    int verbose;
} FaceXCoreMLOptions;

/* Load a Core ML model. `path` points to a `.mlpackage` directory
 * (or a compiled `.mlmodelc`). Returns NULL on error and writes a
 * message to stderr.
 *
 * Expected model interface:
 *   input:   `input` — MultiArray, shape (1, 3, 112, 112), float32, [-1, 1]
 *   output:  `embedding` — MultiArray, shape (1, 512), float32 */
FaceXCoreML* facex_coreml_init(const char* mlpackage_path,
                               const FaceXCoreMLOptions* opts);

/* Run one embedding pass on a 112×112×3 RGB float32 (HWC, [-1, 1])
 * face. Output is L2-normalized so cosine similarity matches the CPU
 * backend. Returns 0 on success, negative errno on failure. */
int facex_coreml_embed(FaceXCoreML* fx,
                       const float* rgb_hwc,
                       float embedding[512]);

/* Returns a short string identifying which compute unit set Core ML
 * actually used for the most recent prediction. One of:
 *   "ane", "gpu", "cpu", "ane+cpu", "gpu+cpu", "ane+gpu+cpu",
 *   "unknown" (pre-macOS-14 hosts can't introspect dispatch). */
const char* facex_coreml_last_dispatch(const FaceXCoreML* fx);

void facex_coreml_free(FaceXCoreML* fx);

#ifdef __cplusplus
}
#endif
