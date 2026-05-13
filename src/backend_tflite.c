/*
 * backend_tflite.c — TFLite C-API wrapper that dispatches to a
 * runtime-selected delegate. Powers the i.MX NPU path (VxDelegate on 8M
 * Plus, Arm Ethos-U external delegate on 93, eIQ Neutron delegate on 95)
 * plus a CPU XNNPACK fallback.
 *
 * Build:
 *   - Compile only when FACEX_BACKEND_TFLITE is defined.
 *   - Link against libtensorflowlite_c.so + libdl.
 *   - Delegates are dlopen'd at runtime so the same libfacex_npu.so works
 *     on a board with the NPU and on a dev box without it.
 *
 * Status:
 *   - Embedder path (facex_npu_embed, and embed-stage of facex_npu_detect):
 *     fully wired. Quantizes the float input to INT8 per the model's input
 *     scale/zero-point, invokes the interpreter, dequantizes the 512-d
 *     output, and L2-normalizes.
 *   - Detector path (facex_npu_detect when detect_tflite != NULL):
 *     STUB — hardware-untested. Anchor decode + NMS for arbitrary YuNet /
 *     SCRFD topology is fragile; the recommended deployment per
 *     docs/imx_npu.md §4 is the hybrid pipeline (CPU detect via libfacex
 *     + NPU embed via this backend). When detect_tflite is NULL the
 *     engine returns -ENOTSUP from facex_npu_detect.
 *
 * Hardware testing: NEEDED. This compiles cleanly and follows the
 * documented TFLite C API + delegate ABI; getting a real .tflite to run on a
 * real board is its own milestone. See docs/imx_npu.md for the bring-up
 * checklist.
 */

#ifdef FACEX_BACKEND_TFLITE

#include "../include/facex_npu.h"
#include "../include/facex.h"

#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>

#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOGE(fx, fmt, ...)  fprintf(stderr, "facex/npu: " fmt "\n", ##__VA_ARGS__)
#define LOGV(fx, fmt, ...)  do { if ((fx) && (fx)->verbose) fprintf(stderr, "facex/npu: " fmt "\n", ##__VA_ARGS__); } while (0)

/* ----- Delegate loader (dlopen + dlsym) ---------------------------------- */
/*
 * Each NPU vendor ships a TFLite "external delegate" plugin that exposes a
 * standard C entry point (`tflite_plugin_create_delegate`) plus an opaque
 * options struct. We load the .so dynamically so libfacex_npu.so itself
 * has no link dependency on it — boards that don't have the NPU just fall
 * through to the next candidate.
 */

/* The shape of the standardized external-delegate factory.
 * This function pointer signature is stable across TFLite >= 2.5. */
typedef TfLiteDelegate* (*tflite_plugin_create_delegate_fn)(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*));
typedef void (*tflite_plugin_destroy_delegate_fn)(TfLiteDelegate*);

typedef struct {
    const char* name;            /* short id used by user / logs */
    const char* libname;         /* dlopen target */
    const char* create_sym;      /* dlsym factory entry */
    const char* destroy_sym;     /* dlsym destructor entry */
} DelegateSpec;

/* Search order — first match wins unless the user pins a preference. */
static const DelegateSpec kKnownDelegates[] = {
    /* NXP eIQ Neutron N3 (i.MX 95). Driver is /dev/neutron0, delegate
     * shipped by NXP in BSP /usr/lib/. Listed first so on a 95 EVK we
     * pick Neutron over anything else also present. */
    { "neutron",  "libneutron_delegate.so",  "tflite_plugin_create_delegate",
                                              "tflite_plugin_destroy_delegate" },
    /* NXP VIP9000 (i.MX 8M Plus). NXP ships this in BSP /usr/lib/. */
    { "vx",       "libvx_delegate.so",       "tflite_plugin_create_delegate",
                                              "tflite_plugin_destroy_delegate" },
    /* Arm Ethos-U external delegate (i.MX 93). Comes from
     * ml-extensions/ethos-u-delegate, NXP ships in BSP. */
    { "ethos-u",  "libethosu_delegate.so",   "tflite_plugin_create_delegate",
                                              "tflite_plugin_destroy_delegate" },
    /* Arm NN delegate — broader op coverage, GPU on Mali, useful on i.MX 8M
     * for layers the VIP9000 rejects. */
    { "armnn",    "libarmnnDelegate.so",     "tflite_plugin_create_delegate",
                                              "tflite_plugin_destroy_delegate" },
    { NULL, NULL, NULL, NULL }
};

typedef struct {
    void*                            lib;
    TfLiteDelegate*                  delegate;
    tflite_plugin_destroy_delegate_fn destroy;
    char                             name[32];
} LoadedDelegate;

static int try_load_one(const DelegateSpec* spec, LoadedDelegate* out, int verbose) {
    void* lib = dlopen(spec->libname, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        if (verbose) fprintf(stderr, "facex/npu: %s not available (%s)\n",
                             spec->name, dlerror());
        return -1;
    }
    tflite_plugin_create_delegate_fn create =
        (tflite_plugin_create_delegate_fn)dlsym(lib, spec->create_sym);
    tflite_plugin_destroy_delegate_fn destroy =
        (tflite_plugin_destroy_delegate_fn)dlsym(lib, spec->destroy_sym);
    if (!create || !destroy) {
        fprintf(stderr, "facex/npu: %s loaded but missing symbols\n", spec->name);
        dlclose(lib);
        return -1;
    }
    TfLiteDelegate* d = create(NULL, NULL, 0, NULL);
    if (!d) {
        fprintf(stderr, "facex/npu: %s create returned NULL\n", spec->name);
        dlclose(lib);
        return -1;
    }
    out->lib = lib;
    out->delegate = d;
    out->destroy = destroy;
    snprintf(out->name, sizeof(out->name), "%s", spec->name);
    if (verbose) fprintf(stderr, "facex/npu: selected delegate '%s'\n", spec->name);
    return 0;
}

/* Derive a short logging name from a delegate library path.
 *   /usr/lib/libneutron_delegate.so → "neutron"
 *   ./build/libfoo.so                → "foo"
 *   anything unparseable             → "external" */
static void derive_path_name(const char* path, char* buf, size_t buflen) {
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;
    /* Strip "lib" prefix and any of the common delegate-suffix shapes. */
    if (strncmp(base, "lib", 3) == 0) base += 3;
    snprintf(buf, buflen, "%s", base);
    char* dot = strchr(buf, '.');                  /* drop ".so" / ".dylib" */
    if (dot) *dot = 0;
    char* sfx = strstr(buf, "_delegate");          /* drop "_delegate" */
    if (sfx) *sfx = 0;
    sfx = strstr(buf, "Delegate");
    if (sfx) *sfx = 0;
    if (!buf[0]) snprintf(buf, buflen, "external");
}

/* Selects a delegate. Returns 0 + populates `out` on success.
 * If `path` is non-NULL it is dlopen'd directly, bypassing the registry.
 * Otherwise if `preferred` is non-NULL, ONLY that delegate is attempted (no
 * fallback); if both are NULL, the kKnownDelegates list is walked in order.
 * Returns -1 if no delegate could be loaded — caller must decide whether
 * to fall back to XNNPACK (CPU). */
static int select_delegate(const char* preferred, const char* path,
                           LoadedDelegate* out, int verbose) {
    if (path && path[0]) {
        char nm[32];
        derive_path_name(path, nm, sizeof(nm));
        DelegateSpec spec = {
            .name        = nm,
            .libname     = path,
            .create_sym  = "tflite_plugin_create_delegate",
            .destroy_sym = "tflite_plugin_destroy_delegate",
        };
        return try_load_one(&spec, out, verbose);
    }
    if (preferred && strcmp(preferred, "xnnpack") == 0) return -1;  /* explicit CPU */
    for (const DelegateSpec* s = kKnownDelegates; s->name; s++) {
        if (preferred && strcmp(preferred, s->name) != 0) continue;
        if (try_load_one(s, out, verbose) == 0) return 0;
        if (preferred) return -1;  /* user pinned, don't try others */
    }
    return -1;
}

/* Some delegates only claim ops from a model that was processed by their
 * specific offline compiler. When that's missing, the delegate loads fine
 * but TFLite logs "0 nodes delegated" and execution silently falls back to
 * the CPU kernels — same latency as XNNPACK, no NPU offload. The C API
 * doesn't expose a clean post-modify-graph node count, so we print a
 * heads-up describing the failure mode and how to fix it; the user pairs
 * this with the TFLite log to diagnose. */
static void print_offline_compiler_hint(const char* delegate, int verbose) {
    if (!verbose) return;
    if (strcmp(delegate, "neutron") == 0) {
        fprintf(stderr,
            "facex/npu: hint — Neutron only accelerates ops from a model\n"
            "           pre-compiled by neutron-converter (NXP eIQ Toolkit).\n"
            "           If TFLite logs '0 nodes delegated', re-run your\n"
            "           .tflite through tools/compile_neutron.sh first.\n");
    } else if (strcmp(delegate, "ethos-u") == 0) {
        fprintf(stderr,
            "facex/npu: hint — Ethos-U only accelerates ops compiled by Vela.\n"
            "           If TFLite logs '0 nodes delegated', run\n"
            "           tools/compile_vela.sh on the .tflite first.\n");
    }
    /* vx / armnn ingest plain INT8 .tflite — no offline step needed. */
}

/* ----- Engine state ------------------------------------------------------ */

struct FaceXNpu {
    TfLiteModel*       emb_model;
    TfLiteInterpreter* emb_interp;

    TfLiteModel*       det_model;        /* may be NULL — embed-only mode */
    TfLiteInterpreter* det_interp;

    LoadedDelegate     delegate;          /* zeroed if XNNPACK fallback */
    int                using_xnnpack;     /* 1 if delegate field unused */
    char               active[32];

    float              score_thresh;
    float              nms_thresh;
    int                verbose;
};

/* ----- Helpers ----------------------------------------------------------- */

static TfLiteModel* load_model_file(const char* path) {
    TfLiteModel* m = TfLiteModelCreateFromFile(path);
    if (!m) fprintf(stderr, "facex/npu: TfLiteModelCreateFromFile failed for %s\n", path);
    return m;
}

static TfLiteInterpreter* build_interpreter(TfLiteModel* m,
                                            TfLiteDelegate* delegate,
                                            int num_threads,
                                            int use_xnnpack) {
    TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
    if (num_threads > 0) TfLiteInterpreterOptionsSetNumThreads(opts, num_threads);
    if (delegate) TfLiteInterpreterOptionsAddDelegate(opts, delegate);
    /* When use_xnnpack is set we let TFLite pick its built-in XNNPACK delegate
     * via the default delegates code path — this is the C-API equivalent of
     * Interpreter::ApplyOptionsAfterInit on Python. No extra call needed:
     * TFLite enables XNNPACK by default for float models since 2.10. */
    (void)use_xnnpack;
    TfLiteInterpreter* it = TfLiteInterpreterCreate(m, opts);
    TfLiteInterpreterOptionsDelete(opts);
    if (!it) {
        fprintf(stderr, "facex/npu: TfLiteInterpreterCreate failed\n");
        return NULL;
    }
    if (TfLiteInterpreterAllocateTensors(it) != kTfLiteOk) {
        fprintf(stderr, "facex/npu: AllocateTensors failed\n");
        TfLiteInterpreterDelete(it);
        return NULL;
    }
    return it;
}

static void l2_normalize_512(float* v) {
    double s = 0;
    for (int i = 0; i < 512; i++) s += (double)v[i] * v[i];
    if (s < 1e-12) return;
    float inv = (float)(1.0 / sqrt(s));
    for (int i = 0; i < 512; i++) v[i] *= inv;
}

/* Quantize a float array to INT8 using the tensor's affine quantization. */
static void quantize_to_int8(const float* src, int8_t* dst, int n,
                             float scale, int32_t zero_point) {
    if (scale <= 0) scale = 1.0f;
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(src[i] / scale) + zero_point;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        dst[i] = (int8_t)q;
    }
}

static void dequantize_int8(const int8_t* src, float* dst, int n,
                            float scale, int32_t zero_point) {
    for (int i = 0; i < n; i++) dst[i] = ((int32_t)src[i] - zero_point) * scale;
}

/* ----- Public API -------------------------------------------------------- */

FaceXNpu* facex_npu_init(const char* embed_tflite,
                         const char* detect_tflite,
                         const FaceXNpuOptions* opts) {
    if (!embed_tflite) {
        fprintf(stderr, "facex/npu: embed_tflite is required\n");
        return NULL;
    }

    FaceXNpu* fx = (FaceXNpu*)calloc(1, sizeof(*fx));
    if (!fx) return NULL;
    fx->score_thresh = 0.5f;
    fx->nms_thresh   = 0.4f;
    if (opts) fx->verbose = opts->verbose;

    /* 1. Pick a delegate (NPU first, XNNPACK fallback). */
    const char* pref = (opts && opts->preferred_delegate) ? opts->preferred_delegate : NULL;
    const char* path = (opts && opts->external_delegate_path) ? opts->external_delegate_path : NULL;
    if (select_delegate(pref, path, &fx->delegate, fx->verbose) != 0) {
        if (path) {
            fprintf(stderr, "facex/npu: external delegate at '%s' failed to load\n", path);
            free(fx);
            return NULL;
        }
        if (pref) {
            fprintf(stderr, "facex/npu: requested delegate '%s' unavailable\n", pref);
            free(fx);
            return NULL;
        }
        fx->using_xnnpack = 1;
        snprintf(fx->active, sizeof(fx->active), "xnnpack");
        if (fx->verbose) fprintf(stderr, "facex/npu: no NPU delegate found — using XNNPACK\n");
    } else {
        snprintf(fx->active, sizeof(fx->active), "%s", fx->delegate.name);
        print_offline_compiler_hint(fx->delegate.name, fx->verbose);
    }

    int n_threads = (opts && opts->num_threads > 0) ? opts->num_threads : 0;
    TfLiteDelegate* d = fx->using_xnnpack ? NULL : fx->delegate.delegate;

    /* 2. Embedder. */
    fx->emb_model = load_model_file(embed_tflite);
    if (!fx->emb_model) { facex_npu_free(fx); return NULL; }
    fx->emb_interp = build_interpreter(fx->emb_model, d, n_threads, fx->using_xnnpack);
    if (!fx->emb_interp) { facex_npu_free(fx); return NULL; }

    /* 3. Detector (optional). */
    if (detect_tflite) {
        fx->det_model = load_model_file(detect_tflite);
        if (!fx->det_model) { facex_npu_free(fx); return NULL; }
        fx->det_interp = build_interpreter(fx->det_model, d, n_threads, fx->using_xnnpack);
        if (!fx->det_interp) { facex_npu_free(fx); return NULL; }
    }

    return fx;
}

int facex_npu_embed(FaceXNpu* fx, const float* rgb_hwc, float embedding[512]) {
    if (!fx || !fx->emb_interp || !rgb_hwc || !embedding) return -EINVAL;

    TfLiteTensor* in = TfLiteInterpreterGetInputTensor(fx->emb_interp, 0);
    const TfLiteTensor* out = TfLiteInterpreterGetOutputTensor(fx->emb_interp, 0);
    if (!in || !out) return -EIO;

    /* Input is 1×112×112×3. Embedder is INT8-quantized for NPU; XNNPACK can
     * also accept float input (model dependent). Branch on tensor dtype. */
    TfLiteType in_type = TfLiteTensorType(in);
    if (in_type == kTfLiteInt8) {
        TfLiteQuantizationParams qp = TfLiteTensorQuantizationParams(in);
        size_t n = (size_t)112 * 112 * 3;
        int8_t* buf = (int8_t*)TfLiteTensorData(in);
        quantize_to_int8(rgb_hwc, buf, (int)n, qp.scale, qp.zero_point);
    } else if (in_type == kTfLiteFloat32) {
        memcpy(TfLiteTensorData(in), rgb_hwc, (size_t)112 * 112 * 3 * sizeof(float));
    } else {
        LOGE(fx, "embedder input dtype not supported (%d)", in_type);
        return -ENOTSUP;
    }

    if (TfLiteInterpreterInvoke(fx->emb_interp) != kTfLiteOk) {
        LOGE(fx, "Invoke failed");
        return -EIO;
    }

    TfLiteType out_type = TfLiteTensorType(out);
    if (out_type == kTfLiteInt8) {
        TfLiteQuantizationParams qp = TfLiteTensorQuantizationParams(out);
        const int8_t* src = (const int8_t*)TfLiteTensorData(out);
        dequantize_int8(src, embedding, 512, qp.scale, qp.zero_point);
    } else if (out_type == kTfLiteFloat32) {
        memcpy(embedding, TfLiteTensorData(out), 512 * sizeof(float));
    } else {
        LOGE(fx, "embedder output dtype not supported (%d)", out_type);
        return -ENOTSUP;
    }

    /* Vela / NXP quantizers don't always emit a final L2 — normalize here
     * so cosine similarity behaves identically to the CPU backend. */
    l2_normalize_512(embedding);
    return 0;
}

int facex_npu_detect(FaceXNpu* fx,
                     const uint8_t* rgb_hwc, int width, int height,
                     FaceXResult* out, int max_faces) {
    (void)rgb_hwc; (void)width; (void)height; (void)out; (void)max_faces;
    if (!fx) return -EINVAL;
    if (!fx->det_interp) {
        LOGE(fx, "facex_npu_detect requires detect_tflite at init time, "
                 "or use the CPU detector via facex.h and call facex_npu_embed per face");
        return -ENOTSUP;
    }
    /* HARDWARE-UNTESTED. Anchor decode + NMS depends on the exact detector
     * topology produced by tools/onnx_to_tflite.py. The recommended
     * deployment is the hybrid pipeline: detect on CPU (libfacex), embed on
     * NPU (this backend). See docs/imx_npu.md. */
    LOGE(fx, "detect path on NPU is not implemented yet — use hybrid pipeline");
    return -ENOSYS;
}

float facex_npu_similarity(const float emb1[512], const float emb2[512]) {
    double dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < 512; i++) {
        dot += (double)emb1[i] * emb2[i];
        n1  += (double)emb1[i] * emb1[i];
        n2  += (double)emb2[i] * emb2[i];
    }
    double denom = sqrt(n1) * sqrt(n2);
    return (denom > 1e-8) ? (float)(dot / denom) : 0.0f;
}

void facex_npu_set_score_threshold(FaceXNpu* fx, float t) { if (fx) fx->score_thresh = t; }
void facex_npu_set_nms_threshold(FaceXNpu* fx, float t)   { if (fx) fx->nms_thresh   = t; }

const char* facex_npu_active_delegate(const FaceXNpu* fx) {
    return fx ? fx->active : "";
}

void facex_npu_free(FaceXNpu* fx) {
    if (!fx) return;
    if (fx->emb_interp) TfLiteInterpreterDelete(fx->emb_interp);
    if (fx->emb_model)  TfLiteModelDelete(fx->emb_model);
    if (fx->det_interp) TfLiteInterpreterDelete(fx->det_interp);
    if (fx->det_model)  TfLiteModelDelete(fx->det_model);
    if (!fx->using_xnnpack && fx->delegate.delegate) {
        fx->delegate.destroy(fx->delegate.delegate);
        if (fx->delegate.lib) dlclose(fx->delegate.lib);
    }
    free(fx);
}

#endif /* FACEX_BACKEND_TFLITE */
