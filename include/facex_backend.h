/*
 * facex_backend.h — Pluggable backend interface.
 *
 * See docs/implementation.md §3 for the i.MX NPU consumer of this
 * vtable, and §2 for the Mac perf paths that share the same shape.
 *
 * A backend is anything that can answer the question "given an RGB image,
 * give me face boxes + 512-dim embeddings". Today we ship two:
 *
 *   - facex/cpu      — the existing C engine in src/edgeface_engine.c.
 *                      Always available, runs on every supported arch.
 *   - facex/tflite   — a TFLite C-API wrapper that loads a precompiled
 *                      .tflite model and dispatches it to a runtime-selected
 *                      delegate. Selection order:
 *                        1. NXP VxDelegate          (libvx_delegate.so)        → i.MX 8M Plus VIP9000
 *                        2. Arm Ethos-U external    (libethosu_delegate.so)    → i.MX 93 / 95 Ethos-U65
 *                        3. XNNPACK CPU fallback    (built into TFLite)
 *
 * Future backends (Apple Core ML / ANE, ESP-NN, NXP eIQ Vela inference
 * runtime) plug in by implementing the same vtable.
 *
 * Invariant: every backend MUST produce a FaceXResult that is byte-compatible
 * with facex.h (same struct layout) — embeddings are L2-comparable across
 * backends so a face enrolled on a Mac with the CPU backend matches the same
 * face detected on an i.MX 95 with the Ethos-U65 backend (within INT8
 * quantization noise, ≤ 0.5% LFW per S9 acceptance test).
 *
 * https://github.com/facex-engine/facex
 * License: Apache 2.0
 */

#ifndef FACEX_BACKEND_H
#define FACEX_BACKEND_H

#include "facex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FacexBackend FacexBackend;

/* Backend identifier — informational, returned by facex_backend_name(). */
typedef enum {
    FACEX_BACKEND_CPU       = 0,  /* src/edgeface_engine.c */
    FACEX_BACKEND_TFLITE    = 1,  /* src/backend_tflite.c, runtime-selected delegate */
    FACEX_BACKEND_COREML    = 2,  /* future: src/backend_coreml.m (Apple ANE) */
    FACEX_BACKEND_ESPNN     = 3,  /* future: src/backend_espnn.c (ESP32-P4) */
} FacexBackendKind;

/* Backend vtable. All function pointers are required unless marked optional. */
struct FacexBackend {
    FacexBackendKind kind;
    const char* name;          /* "cpu", "tflite-vx", "tflite-ethosu", "tflite-xnnpack", … */

    /* Opaque per-instance state. Initialised by `init`, freed by `free`. */
    void* state;

    /* Open weights/models, allocate buffers. Returns 0 on success.
     *   embed_path:  required (engine artifact for this backend).
     *   detect_path: optional — NULL for embed-only backends.
     *   options:     opaque, backend-specific (e.g. delegate name, num threads). */
    int (*init)(FacexBackend* self,
                const char* embed_path,
                const char* detect_path,
                const void* options);

    /* Detect + align + embed. Mirrors facex_detect() in facex.h. */
    int (*detect)(FacexBackend* self,
                  const uint8_t* rgb_hwc,
                  int width, int height,
                  FaceXResult* out, int max_faces);

    /* Embed-only on a pre-aligned 112×112 face. Mirrors facex_embed(). */
    int (*embed)(FacexBackend* self,
                 const float* rgb_hwc, /* [112*112*3], values in [-1,1] */
                 float embedding[512]);

    /* Optional: thresholds, etc. Pass NULL if backend has no tunables. */
    void (*set_score_threshold)(FacexBackend* self, float t);
    void (*set_nms_threshold)(FacexBackend* self, float t);

    /* Free state. After this call the FacexBackend is dead. */
    void (*free)(FacexBackend* self);
};

/* ---- Built-in backend factories ----------------------------------------- */

/* Returns a heap-allocated CPU backend. Always succeeds.
 * Free with self->free(self). */
FacexBackend* facex_backend_cpu(void);

#ifdef FACEX_BACKEND_TFLITE
/* Returns a heap-allocated TFLite backend, or NULL if libtensorflowlite_c.so
 * cannot be located at runtime. The actual NPU delegate is selected lazily on
 * first init() call (see src/backend_tflite.c). */
FacexBackend* facex_backend_tflite(void);

/* Hint to the TFLite backend: which delegate to attempt first.
 * NULL = auto (NXP VX → Arm Ethos-U → XNNPACK). Examples:
 *   "vx"       — only the VIP9000 delegate; fail if missing.
 *   "ethos-u"  — only the Arm Ethos-U external delegate; fail if missing.
 *   "xnnpack"  — CPU-only path inside TFLite (useful for dev / fallback test). */
void facex_backend_tflite_set_preferred_delegate(FacexBackend* self,
                                                 const char* name);
#endif

/* ---- Helpers ------------------------------------------------------------ */

const char* facex_backend_name(const FacexBackend* self);

#ifdef __cplusplus
}
#endif

#endif /* FACEX_BACKEND_H */
