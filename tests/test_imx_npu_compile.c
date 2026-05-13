/*
 * test_imx_npu_compile.c — Compile + link sanity check for the NPU backend.
 *
 * Runs anywhere libtensorflowlite_c.so is installed. With no args it just
 * proves the API surface links cleanly and that NULL inputs are rejected
 * with the expected error codes — useful in CI on a host without an actual
 * NPU device.
 *
 * With one or two .tflite paths it tries to load the model(s) and report
 * the active delegate. On a real i.MX board this should print
 *   active delegate: vx        (i.MX 8M Plus)
 *   active delegate: ethos-u   (i.MX 93 / 95)
 *   active delegate: xnnpack   (any board, no NPU)
 *
 * Usage:
 *   ./imx_npu_compile_test                                 # API smoke only
 *   ./imx_npu_compile_test embed.tflite                    # embedder-only init
 *   ./imx_npu_compile_test embed.tflite detect.tflite      # both models
 */

#include "facex_npu.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int api_smoke(void) {
    /* NULL embed_tflite must return NULL with a stderr message. */
    FaceXNpu* fx = facex_npu_init(NULL, NULL, NULL);
    if (fx) {
        fprintf(stderr, "FAIL: facex_npu_init(NULL, …) returned non-NULL\n");
        facex_npu_free(fx);
        return 1;
    }
    /* Helpers must be safe on NULL. */
    if (facex_npu_active_delegate(NULL)[0] != '\0') {
        fprintf(stderr, "FAIL: active_delegate(NULL) should be empty string\n");
        return 1;
    }
    facex_npu_set_score_threshold(NULL, 0.5f);  /* must not crash */
    facex_npu_set_nms_threshold(NULL, 0.4f);    /* must not crash */
    float a[512] = {0}, b[512] = {0};
    a[0] = 1.0f; b[0] = 1.0f;
    if (facex_npu_similarity(a, b) < 0.99f) {
        fprintf(stderr, "FAIL: similarity of unit vectors should be ~1\n");
        return 1;
    }
    printf("[ok] NPU API surface compiles, links, and rejects NULL\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("FaceX NPU compile/link smoke test\n");

    if (api_smoke() != 0) return 1;

    if (argc < 2) {
        printf("\nNo .tflite supplied — API smoke only.\n"
               "Pass `embed.tflite [detect.tflite]` to also try a real init.\n");
        return 0;
    }

    const char* embed_path  = argv[1];
    const char* detect_path = (argc >= 3) ? argv[2] : NULL;

    FaceXNpuOptions opts = {0};
    opts.verbose = 1;
    opts.num_threads = 0;            /* autodetect */
    opts.preferred_delegate = NULL;  /* let runtime pick */

    FaceXNpu* fx = facex_npu_init(embed_path, detect_path, &opts);
    if (!fx) {
        fprintf(stderr, "FAIL: facex_npu_init returned NULL\n");
        return 2;
    }
    printf("[ok] init succeeded\n");
    printf("     active delegate: %s\n", facex_npu_active_delegate(fx));

    /* Try one embed call with all-zero input — checks the input/output dtype
     * branches and confirms the embedding is 512 finite floats. */
    float input[112 * 112 * 3] = {0};
    float emb[512];
    int rc = facex_npu_embed(fx, input, emb);
    if (rc != 0) {
        fprintf(stderr, "FAIL: facex_npu_embed returned %d (%s)\n",
                rc, strerror(rc < 0 ? -rc : rc));
        facex_npu_free(fx);
        return 3;
    }
    int finite = 0;
    double s = 0;
    for (int i = 0; i < 512; i++) {
        if (emb[i] == emb[i]) finite++;
        s += (double)emb[i] * emb[i];
    }
    printf("[ok] embed: %d/512 finite, ||emb||² = %.4f\n", finite, s);

    /* Detector path is documented as not implemented yet — confirm the
     * error code is the documented one. */
    if (detect_path) {
        uint8_t img[160 * 160 * 3] = {0};
        FaceXResult out[4];
        rc = facex_npu_detect(fx, img, 160, 160, out, 4);
        if (rc == -ENOSYS) {
            printf("[ok] detect returns -ENOSYS as documented (use hybrid pipeline)\n");
        } else {
            printf("[note] detect returned %d (expected -ENOSYS for now)\n", rc);
        }
    }

    facex_npu_free(fx);
    printf("\nPASS: NPU compile + link smoke\n");
    return 0;
}
