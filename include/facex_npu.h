/*
 * facex_npu.h — i.MX NPU public API.
 *
 * Same shape as facex.h (single FaceX* handle, FaceXResult struct), but
 * loads pre-compiled .tflite models and dispatches inference via a
 * runtime-selected TFLite delegate. Targets:
 *
 *   i.MX 8M Plus → NXP VxDelegate (VIP9000 NPU, 2.3 TOPS)
 *   i.MX 93      → Arm Ethos-U external delegate (Ethos-U65, ~0.5 TOPS)
 *   i.MX 95      → NXP eIQ Neutron delegate (Neutron N3 NPU)
 *   any AArch64  → XNNPACK CPU fallback (no NPU, slower, useful for dev)
 *
 * Models are produced offline:
 *   tools/onnx_to_tflite.py   PyTorch → ONNX → quantized .tflite
 *   tools/compile_vela.sh     .tflite → Ethos-U65 command stream  (i.MX 93)
 *   neutron-converter         .tflite → Neutron command stream    (i.MX 95;
 *                                       ships with NXP eIQ Toolkit)
 *
 * The application code is identical regardless of target — drop a different
 * .tflite file in place and the same binary runs.
 *
 * https://github.com/facex-engine/facex
 * License: Apache 2.0
 */

#ifndef FACEX_NPU_H
#define FACEX_NPU_H

#include "facex.h"  /* FaceXResult */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FaceXNpu FaceXNpu;

typedef struct {
    /* Hint for which TFLite delegate to attempt first. NULL = auto.
     * Set to "neutron" / "vx" / "ethos-u" / "xnnpack" / "armnn" to force one. */
    const char* preferred_delegate;

    /* Optional absolute path to an external delegate .so to dlopen directly,
     * bypassing the built-in registry. The shared object must expose the
     * standard TFLite external-delegate ABI (`tflite_plugin_create_delegate`
     * + `tflite_plugin_destroy_delegate`). Useful for benchmarking unusual
     * delegates without rebuilding libfacex_npu.so, and for matching
     * NXP's `benchmark_model --external_delegate_path=…` invocation.
     * If both this and `preferred_delegate` are set, the path wins. */
    const char* external_delegate_path;

    /* Number of CPU threads for the XNNPACK fallback and for any layers the
     * NPU rejects (kept on CPU). 0 = autodetect. */
    int num_threads;

    /* If non-zero, the backend prints its delegate-selection decisions to
     * stderr at init time. Useful when debugging NPU dispatch. */
    int verbose;
} FaceXNpuOptions;

/*
 * Initialise NPU engine.
 *   embed_tflite:  Vela-compiled .tflite for the embedder (required).
 *   detect_tflite: Vela-compiled .tflite for the detector (optional, NULL
 *                  for embed-only mode).
 *   opts:          may be NULL (uses defaults).
 *   Returns: handle, or NULL on error (writes a message to stderr).
 *
 * Errors include: TFLite C library not found at runtime, both delegates
 * missing AND XNNPACK fallback explicitly disabled, malformed .tflite,
 * model expects an input shape that doesn't match the engine's contract.
 */
FaceXNpu* facex_npu_init(const char* embed_tflite,
                         const char* detect_tflite,
                         const FaceXNpuOptions* opts);

/*
 * Detect + align + embed (full pipeline). Same contract as
 * facex.h:facex_detect — fills FaceXResult.bbox/score/kps/embedding.
 */
int facex_npu_detect(FaceXNpu* fx,
                     const uint8_t* rgb_hwc, int width, int height,
                     FaceXResult* out, int max_faces);

/*
 * Embed-only on a pre-aligned 112×112 RGB face (float32 HWC, [-1,1]).
 */
int facex_npu_embed(FaceXNpu* fx, const float* rgb_hwc, float embedding[512]);

/* Cosine similarity helper — identical to facex_similarity. Provided here
 * so callers using the NPU API don't need to also include facex.h. */
float facex_npu_similarity(const float emb1[512], const float emb2[512]);

void facex_npu_set_score_threshold(FaceXNpu* fx, float t);
void facex_npu_set_nms_threshold(FaceXNpu* fx, float t);

/*
 * Returns the actual delegate that was selected at runtime — useful for
 * logging / metrics. One of: "vx", "ethos-u", "xnnpack", "armnn", "cpu".
 * Owned by the engine; do not free.
 */
const char* facex_npu_active_delegate(const FaceXNpu* fx);

void facex_npu_free(FaceXNpu* fx);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_NPU_H */
