/*
 * tools/bench_npu.c — TFLite-side companion to tools/bench.c.
 *
 * Same synthetic-input recipe and same CSV/JSON schema as facex-bench, but
 * dispatches inference through libfacex_npu.so → TFLite C API → external
 * delegate. Lets us compare CPU NEON, XNNPACK, eIQ Neutron, Ethos-U, and
 * VxDelegate side-by-side in one harness with a single output format.
 *
 * Why a separate binary: facex-bench links libfacex.a and runs everywhere;
 * facex-bench-npu links libfacex_npu.so and pulls in libtensorflowlite_c.
 * Keeping them separate preserves the "facex-bench runs on any host"
 * promise — and matches how the libraries themselves are split.
 *
 * Args:
 *   --iters N             measurement iterations (default 100)
 *   --warmup K            warmup iterations (default 10)
 *   --format md|csv|json  (default md)
 *   --label STR           tag copied verbatim into output (build config etc.)
 *   --embed PATH          .tflite embedder model (required)
 *   --delegate NAME       force a registered delegate by name
 *                         (neutron / vx / ethos-u / xnnpack / armnn)
 *   --external-delegate PATH
 *                         dlopen this .so directly, bypassing the registry.
 *                         Standard TFLite external-delegate ABI is required.
 *   --threads N           CPU threads for fallback layers (default: autodetect)
 *
 * Build: `make facex-bench-npu` (depends on libfacex_npu.so being built).
 * E2E stage is intentionally absent — facex_npu_detect is -ENOSYS today
 * and routing detect through the CPU path here would conflate backends.
 */

#include "facex_npu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

typedef enum { FMT_MD, FMT_CSV, FMT_JSON } Fmt;

typedef struct {
    int    iters;
    int    warmup;
    int    threads;
    Fmt    fmt;
    const char* label;
    const char* embed_path;
    const char* delegate_name;
    const char* delegate_path;
} Args;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int cmp_d(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

typedef struct { double min, median, p95, p99, mean; int n; } Stats;

static Stats compute(double* xs, int n) {
    Stats s = {0};
    s.n = n;
    if (n <= 0) return s;
    qsort(xs, n, sizeof(double), cmp_d);
    s.min    = xs[0];
    s.median = xs[n / 2];
    s.p95    = xs[(int)(n * 0.95)];
    s.p99    = xs[(int)(n * 0.99)];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += xs[i];
    s.mean = sum / n;
    return s;
}

static void usage(void) {
    fputs(
        "facex-bench-npu — TFLite delegate latency benchmark\n"
        "Usage: facex-bench-npu --embed PATH.tflite [options]\n"
        "  --embed PATH                .tflite embedder model (required)\n"
        "  --iters N                   measurement iterations (default 100)\n"
        "  --warmup K                  warmup iterations (default 10)\n"
        "  --format md|csv|json        (default md)\n"
        "  --label STR                 tag copied to output\n"
        "  --delegate NAME             registered delegate (neutron/vx/ethos-u/xnnpack/armnn)\n"
        "  --external-delegate PATH    dlopen this .so directly (overrides --delegate)\n"
        "  --threads N                 CPU threads for fallback layers\n"
        "  -h, --help                  this help\n", stderr);
}

static int parse_args(int argc, char** argv, Args* a) {
    a->iters = 100;
    a->warmup = 10;
    a->threads = 0;
    a->fmt = FMT_MD;
    a->label = "";
    a->embed_path = NULL;
    a->delegate_name = NULL;
    a->delegate_path = NULL;
    for (int i = 1; i < argc; i++) {
        const char* k = argv[i];
        const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(k, "--iters")   && v) { a->iters   = atoi(v); i++; }
        else if (!strcmp(k, "--warmup")  && v) { a->warmup  = atoi(v); i++; }
        else if (!strcmp(k, "--threads") && v) { a->threads = atoi(v); i++; }
        else if (!strcmp(k, "--format")  && v) {
            if      (!strcmp(v, "md"))   a->fmt = FMT_MD;
            else if (!strcmp(v, "csv"))  a->fmt = FMT_CSV;
            else if (!strcmp(v, "json")) a->fmt = FMT_JSON;
            else { fprintf(stderr, "unknown format: %s\n", v); return -1; }
            i++;
        }
        else if (!strcmp(k, "--label")             && v) { a->label = v; i++; }
        else if (!strcmp(k, "--embed")             && v) { a->embed_path = v; i++; }
        else if (!strcmp(k, "--delegate")          && v) { a->delegate_name = v; i++; }
        else if (!strcmp(k, "--external-delegate") && v) { a->delegate_path = v; i++; }
        else if (!strcmp(k, "-h") || !strcmp(k, "--help")) { usage(); exit(0); }
        else { fprintf(stderr, "unknown arg: %s\n", k); usage(); return -1; }
    }
    if (!a->embed_path) { fprintf(stderr, "--embed is required\n"); return -1; }
    if (a->iters  < 1)  { fprintf(stderr, "--iters must be >= 1\n");  return -1; }
    if (a->warmup < 0)  { fprintf(stderr, "--warmup must be >= 0\n"); return -1; }
    return 0;
}

/* ---- output formatters ------------------------------------------------- */
/* Schema is identical to tools/bench.c so rows can be concatenated. The
 * "compiled" column is fixed to "TFLite" since the actual op kernels live
 * inside the delegate / TFLite runtime, not in libfacex_npu.so itself. */

static void emit_md(const Args* a, const char* active, const Stats* s) {
    printf("# FaceX NPU bench\n\n");
    if (a->label[0]) printf("**label:** %s  \n", a->label);
    printf("**backends compiled:** TFLite  \n");
    printf("**backends active:** %s  \n", active);
    printf("**model:** %s  \n\n", a->embed_path);
    printf("| stage | iters | min ms | median ms | mean ms | p95 ms | p99 ms |\n");
    printf("|---|--:|--:|--:|--:|--:|--:|\n");
    printf("| embed | %d | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
           s->n, s->min, s->median, s->mean, s->p95, s->p99);
    printf("\n");
}

static void emit_csv(const Args* a, const char* active, const Stats* s) {
    printf("label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face\n");
    printf("\"%s\",\"TFLite\",\"%s\",embed,%d,%.3f,%.3f,%.3f,%.3f,%.3f,\n",
           a->label, active,
           s->n, s->min, s->median, s->mean, s->p95, s->p99);
}

static void emit_json(const Args* a, const char* active, const Stats* s) {
    printf("{\n");
    printf("  \"label\": \"%s\",\n", a->label);
    printf("  \"backends_compiled\": \"TFLite\",\n");
    printf("  \"backends_active\":   \"%s\",\n", active);
    printf("  \"model\": \"%s\",\n", a->embed_path);
    printf("  \"stages\": [\n");
    printf("    { \"name\": \"embed\", \"iters\": %d, \"min_ms\": %.3f, \"median_ms\": %.3f, \"mean_ms\": %.3f, \"p95_ms\": %.3f, \"p99_ms\": %.3f }\n",
           s->n, s->min, s->median, s->mean, s->p95, s->p99);
    printf("  ]\n}\n");
}

/* ---- main -------------------------------------------------------------- */

int main(int argc, char** argv) {
    Args a;
    if (parse_args(argc, argv, &a) != 0) return 2;

    FaceXNpuOptions opts = {
        .preferred_delegate     = a.delegate_name,
        .external_delegate_path = a.delegate_path,
        .num_threads            = a.threads,
        .verbose                = 1,
    };

    FaceXNpu* fx = facex_npu_init(a.embed_path, NULL, &opts);
    if (!fx) {
        fprintf(stderr, "facex_npu_init failed for %s\n", a.embed_path);
        return 3;
    }

    /* Same input pattern as tools/bench.c so the magnitudes line up. */
    float in[112 * 112 * 3];
    for (int i = 0; i < 112 * 112 * 3; i++)
        in[i] = (float)(i % 256) / 128.0f - 1.0f;
    float emb[512];

    for (int i = 0; i < a.warmup; i++) facex_npu_embed(fx, in, emb);

    double* samples = (double*)malloc(a.iters * sizeof(double));
    if (!samples) { facex_npu_free(fx); return 4; }

    for (int i = 0; i < a.iters; i++) {
        double t0 = now_ms();
        facex_npu_embed(fx, in, emb);
        samples[i] = now_ms() - t0;
    }
    Stats s = compute(samples, a.iters);
    free(samples);

    const char* active = facex_npu_active_delegate(fx);
    if (!active) active = "unknown";

    switch (a.fmt) {
        case FMT_MD:   emit_md(&a, active, &s);   break;
        case FMT_CSV:  emit_csv(&a, active, &s);  break;
        case FMT_JSON: emit_json(&a, active, &s); break;
    }

    facex_npu_free(fx);
    return 0;
}
