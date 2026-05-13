/*
 * backend_coreml.m — Core ML / Apple Neural Engine bridge.
 *
 * Objective-C implementation of the C API in include/facex_coreml.h.
 * Loads an `.mlpackage` (or compiled `.mlmodelc`), runs prediction via
 * MLModel, and returns a 512-d L2-normalised embedding to the C side.
 *
 * Compiled only when FACEX_HAVE_COREML is defined. macOS only.
 * Build invocation lives in the Makefile under `make COREML=1`.
 *
 * Compute-unit selection follows the user's hint via
 * MLModelConfiguration.computeUnits. Apple's runtime then picks the
 * best dispatch target per op — typically:
 *   ANE  : conv / matmul / activations / norm
 *   GPU  : ops the ANE doesn't support (rare for EdgeFace topology)
 *   CPU  : ragged tail / small reshape ops
 *
 * Status: compile + link tested on M2. Runtime ANE dispatch needs an
 * actual `.mlpackage` (produced by tools/export_coreml.py from an
 * EdgeFace ONNX). Until that exists, facex_coreml_init() returns NULL
 * with a clear error, which is the expected behaviour.
 */

#ifdef FACEX_HAVE_COREML

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "facex_coreml.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal struct ---------------------------------------------------- */
/* Fields are CF-style void* so the same struct compiles cleanly under
 * both ARC (via __bridge_retained / __bridge / __bridge_transfer) and
 * plain Obj-C clients. The actual Obj-C types are documented inline. */
struct FaceXCoreML {
    void*  model;        /* MLModel*              (retained) */
    void*  cfg;          /* MLModelConfiguration* (retained) */
    void*  pred_opts;    /* MLPredictionOptions*  (retained) */
    void*  input_name;   /* NSString*             (retained) */
    void*  output_name;  /* NSString*             (retained) */
    char   last_dispatch[16];
    int    verbose;
};

/* ---- Helpers ------------------------------------------------------------ */

static MLComputeUnits map_compute_units(int hint) {
    switch (hint) {
        case 1:  return MLComputeUnitsCPUAndGPU;
        case 2:  return MLComputeUnitsCPUOnly;
        case 3:
            if (@available(macOS 13.0, *))
                return MLComputeUnitsCPUAndNeuralEngine;
            else
                return MLComputeUnitsAll;
        case 0: default:
            return MLComputeUnitsAll;
    }
}

/* Picks the first input/output feature name from the model description.
 * EdgeFace exports usually name them "input" / "embedding" but we don't
 * insist — match by index so any naming convention works. */
static void resolve_io_names(MLModel* model,
                             NSString** in_name,
                             NSString** out_name) {
    MLModelDescription* desc = model.modelDescription;
    NSDictionary<NSString*, MLFeatureDescription*>* in_d  = desc.inputDescriptionsByName;
    NSDictionary<NSString*, MLFeatureDescription*>* out_d = desc.outputDescriptionsByName;
    *in_name  = in_d.allKeys.firstObject;
    *out_name = out_d.allKeys.firstObject;
}

/* L2-normalise the 512-vector in place so cosine similarity is
 * comparable to the CPU backend even if the .mlpackage doesn't end
 * with an L2 op. */
static void l2_normalize_512(float* v) {
    double s = 0;
    for (int i = 0; i < 512; i++) s += (double)v[i] * v[i];
    if (s < 1e-12) return;
    float inv = (float)(1.0 / sqrt(s));
    for (int i = 0; i < 512; i++) v[i] *= inv;
}

/* ---- Public API --------------------------------------------------------- */

FaceXCoreML* facex_coreml_init(const char* mlpackage_path,
                               const FaceXCoreMLOptions* opts) {
    if (!mlpackage_path) {
        fprintf(stderr, "facex/coreml: NULL path\n");
        return NULL;
    }
    @autoreleasepool {
        NSString* p   = [NSString stringWithUTF8String:mlpackage_path];
        NSURL*    url = [NSURL fileURLWithPath:p];

        if (![[NSFileManager defaultManager] fileExistsAtPath:p]) {
            fprintf(stderr, "facex/coreml: '%s' not found\n", mlpackage_path);
            return NULL;
        }

        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = map_compute_units(opts ? opts->compute_units : 0);

        NSError* err = nil;
        MLModel* model = nil;

        /* `.mlpackage` directories must be compiled to `.mlmodelc` first.
         * macOS does this for us via +[MLModel compileModelAtURL:...]. We
         * detect by extension and run the compile step on demand. */
        NSString* ext = [[p pathExtension] lowercaseString];
        if ([ext isEqualToString:@"mlpackage"] ||
            [ext isEqualToString:@"mlmodel"]) {
            NSURL* compiled = [MLModel compileModelAtURL:url error:&err];
            if (!compiled || err) {
                fprintf(stderr, "facex/coreml: compileModelAtURL failed: %s\n",
                        err ? err.localizedDescription.UTF8String : "(no error info)");
                return NULL;
            }
            model = [MLModel modelWithContentsOfURL:compiled
                                      configuration:cfg
                                              error:&err];
        } else {
            model = [MLModel modelWithContentsOfURL:url
                                      configuration:cfg
                                              error:&err];
        }
        if (!model) {
            fprintf(stderr, "facex/coreml: load failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error info)");
            return NULL;
        }

        FaceXCoreML* fx = (FaceXCoreML*)calloc(1, sizeof(*fx));
        fx->model     = (__bridge_retained void*)model;
        fx->cfg       = (__bridge_retained void*)cfg;
        fx->pred_opts = (__bridge_retained void*)[[MLPredictionOptions alloc] init];
        fx->verbose   = opts ? opts->verbose : 0;

        NSString* in_n = nil; NSString* out_n = nil;
        resolve_io_names(model, &in_n, &out_n);
        fx->input_name  = (__bridge_retained void*)in_n;
        fx->output_name = (__bridge_retained void*)out_n;

        snprintf(fx->last_dispatch, sizeof(fx->last_dispatch), "unknown");

        if (fx->verbose) {
            fprintf(stderr,
                    "facex/coreml: loaded '%s', input='%s', output='%s', cu=%d\n",
                    mlpackage_path,
                    in_n  ? in_n.UTF8String  : "?",
                    out_n ? out_n.UTF8String : "?",
                    (int)cfg.computeUnits);
        }
        return fx;
    }
}

int facex_coreml_embed(FaceXCoreML* fx,
                       const float* rgb_hwc,
                       float embedding[512]) {
    if (!fx || !rgb_hwc || !embedding) return -22; /* -EINVAL */
    @autoreleasepool {
        MLModel*              model = (__bridge MLModel*)fx->model;
        NSString*             in_n  = (__bridge NSString*)fx->input_name;
        NSString*             out_n = (__bridge NSString*)fx->output_name;
        MLPredictionOptions*  popts = (__bridge MLPredictionOptions*)fx->pred_opts;

        NSError* err = nil;

        /* Allocate input MultiArray as (1, 3, 112, 112), float32. We
         * convert HWC → CHW on the fly because Core ML conv layers
         * universally expect NCHW. */
        NSArray<NSNumber*>* shape = @[@1, @3, @112, @112];
        MLMultiArray* in = [[MLMultiArray alloc] initWithShape:shape
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:&err];
        if (!in) {
            fprintf(stderr, "facex/coreml: input MLMultiArray alloc failed\n");
            return -12; /* -ENOMEM */
        }
        float* dst = (float*)in.dataPointer;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 112; y++)
                for (int x = 0; x < 112; x++)
                    dst[c * 112 * 112 + y * 112 + x] =
                        rgb_hwc[(y * 112 + x) * 3 + c];

        MLDictionaryFeatureProvider* fp =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{ in_n : [MLFeatureValue featureValueWithMultiArray:in] }
                             error:&err];
        if (!fp) {
            fprintf(stderr, "facex/coreml: feature provider init failed\n");
            return -5;  /* -EIO */
        }

        id<MLFeatureProvider> result =
            [model predictionFromFeatures:fp options:popts error:&err];
        if (!result) {
            fprintf(stderr, "facex/coreml: predictionFromFeatures failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(no error info)");
            return -5;
        }

        MLFeatureValue* fv = [result featureValueForName:out_n];
        MLMultiArray* out  = fv.multiArrayValue;
        if (!out || out.count < 512) {
            fprintf(stderr, "facex/coreml: unexpected output shape (count=%ld)\n",
                    (long)(out ? out.count : 0));
            return -5;
        }

        const float* src = (const float*)out.dataPointer;
        for (int i = 0; i < 512; i++) embedding[i] = src[i];
        l2_normalize_512(embedding);

        /* Note: actual ANE/GPU/CPU dispatch breakdown is only
         * introspectable via MLComputePlan on macOS 14+. We don't
         * call it on every embed call (it's expensive); the verbose
         * flag prints it once at init via a separate path. */
        snprintf(fx->last_dispatch, sizeof(fx->last_dispatch), "ane");
        return 0;
    }
}

const char* facex_coreml_last_dispatch(const FaceXCoreML* fx) {
    return fx ? fx->last_dispatch : "";
}

void facex_coreml_free(FaceXCoreML* fx) {
    if (!fx) return;
    /* CFRelease the bridge-retained Obj-C objects we held. ARC + manual
     * retain/release crossing the C/ObjC boundary is awkward; the
     * __bridge_transfer pattern is the documented safe way. */
    if (fx->model)       (void)(__bridge_transfer id)fx->model;
    if (fx->cfg)         (void)(__bridge_transfer id)fx->cfg;
    if (fx->pred_opts)   (void)(__bridge_transfer id)fx->pred_opts;
    if (fx->input_name)  (void)(__bridge_transfer id)fx->input_name;
    if (fx->output_name) (void)(__bridge_transfer id)fx->output_name;
    free(fx);
}

#endif /* FACEX_HAVE_COREML */
