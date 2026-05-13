/*
 * tools/bench.c — Unified latency bench for FaceX.
 *
 * One binary, one input format (synthetic deterministic), one output
 * schema. Replaces the scattered ad-hoc benches with something the
 * matrix in docs/coverage_matrix.md and CI can consume.
 *
 * Why synthetic input: makes every backend / build / host directly
 * comparable. The camera bench (tools/bench_camera_mac.swift) remains
 * the right tool for live-camera throughput — it measures a
 * different thing (capture pipeline + dispatch + display).
 *
 * Args:
 *   --iters N      — measurement iterations (default 100)
 *   --warmup K     — warmup iterations (default 10)
 *   --stage embed|e2e|both   (default both; e2e requires the detector)
 *   --format md|csv|json     (default md)
 *   --label STR    — string copied verbatim into output (lets a sweep
 *                    script tag rows with the build config)
 *   --embed PATH   — embedder weights (default data/edgeface_xs_fp32.bin)
 *   --detect PATH  — detector weights (default weights/yunet_fp32.bin)
 *
 * Build: `make bench` produces ./facex-bench.
 */

#include "facex.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef FACEX_HAVE_SME
extern int facex_has_sme(void);
extern int facex_has_sme2(void);
#endif
#ifdef FACEX_HAVE_ACCELERATE
extern int facex_accelerate_enabled(void);
#endif

#define MAX_FACES 8

typedef enum { FMT_MD, FMT_CSV, FMT_JSON } Fmt;

typedef struct {
    int    iters;
    int    warmup;
    int    do_embed;
    int    do_e2e;
    Fmt    fmt;
    const char* label;
    const char* embed_path;
    const char* detect_path;
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
        "facex-bench — unified latency benchmark\n"
        "Usage: facex-bench [options]\n"
        "  --iters N        measurement iterations (default 100)\n"
        "  --warmup K       warmup iterations (default 10)\n"
        "  --stage S        embed | e2e | both (default both)\n"
        "  --format F       md | csv | json (default md)\n"
        "  --label STR      label copied to output (e.g. build config)\n"
        "  --embed  PATH    embedder weights (default data/edgeface_xs_fp32.bin)\n"
        "  --detect PATH    detector weights (default weights/yunet_fp32.bin; pass '' to disable)\n"
        "  -h, --help       this help\n", stderr);
}

static int parse_args(int argc, char** argv, Args* a) {
    a->iters = 100;
    a->warmup = 10;
    a->do_embed = 1;
    a->do_e2e   = 1;
    a->fmt = FMT_MD;
    a->label = "";
    a->embed_path  = "data/edgeface_xs_fp32.bin";
    a->detect_path = "weights/yunet_fp32.bin";
    for (int i = 1; i < argc; i++) {
        const char* k = argv[i];
        const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(k, "--iters")  && v) { a->iters  = atoi(v); i++; }
        else if (!strcmp(k, "--warmup") && v) { a->warmup = atoi(v); i++; }
        else if (!strcmp(k, "--stage")  && v) {
            if      (!strcmp(v, "embed"))  { a->do_embed = 1; a->do_e2e = 0; }
            else if (!strcmp(v, "e2e"))    { a->do_embed = 0; a->do_e2e = 1; }
            else if (!strcmp(v, "both"))   { a->do_embed = 1; a->do_e2e = 1; }
            else { fprintf(stderr, "unknown stage: %s\n", v); return -1; }
            i++;
        }
        else if (!strcmp(k, "--format") && v) {
            if      (!strcmp(v, "md"))   a->fmt = FMT_MD;
            else if (!strcmp(v, "csv"))  a->fmt = FMT_CSV;
            else if (!strcmp(v, "json")) a->fmt = FMT_JSON;
            else { fprintf(stderr, "unknown format: %s\n", v); return -1; }
            i++;
        }
        else if (!strcmp(k, "--label")  && v) { a->label = v; i++; }
        else if (!strcmp(k, "--embed")  && v) { a->embed_path = v; i++; }
        else if (!strcmp(k, "--detect") && v) { a->detect_path = (*v ? v : NULL); i++; }
        else if (!strcmp(k, "-h") || !strcmp(k, "--help")) { usage(); exit(0); }
        else { fprintf(stderr, "unknown arg: %s\n", k); usage(); return -1; }
    }
    if (a->iters  < 1)  { fprintf(stderr, "--iters must be >= 1\n");  return -1; }
    if (a->warmup < 0)  { fprintf(stderr, "--warmup must be >= 0\n"); return -1; }
    return 0;
}

/* ---- backend reporting (compile-time + runtime) ------------------------ */

static void print_backends_compiled(char* buf, size_t n) {
    int written = 0;
    buf[0] = 0;
#define APPEND(s) do { int r = snprintf(buf + written, n - written, "%s%s", written ? "+" : "", s); if (r > 0) written += r; } while (0)
#ifdef FACEX_HAVE_ACCELERATE
    APPEND("Accelerate");
#endif
#ifdef FACEX_HAVE_SME
    APPEND("SME");
#endif
#ifdef FACEX_HAVE_COREML
    APPEND("CoreML");
#endif
    APPEND("NEON");
#undef APPEND
}

static void print_backends_active(char* buf, size_t n) {
    int written = 0;
    buf[0] = 0;
#define APPEND(s) do { int r = snprintf(buf + written, n - written, "%s%s", written ? "+" : "", s); if (r > 0) written += r; } while (0)
#ifdef FACEX_HAVE_ACCELERATE
    if (facex_accelerate_enabled()) APPEND("Accelerate(AMX)");
#endif
#ifdef FACEX_HAVE_SME
    if (facex_has_sme())  APPEND("SME");
    if (facex_has_sme2()) APPEND("SME2");
#endif
    APPEND("NEON");
#undef APPEND
}

/* ---- output formatters ------------------------------------------------- */

static void emit_md(const Args* a,
                    const char* compiled, const char* active,
                    const Stats* s_embed, const Stats* s_e2e,
                    int e2e_have_face) {
    printf("# FaceX bench\n\n");
    if (a->label[0]) printf("**label:** %s  \n", a->label);
    printf("**backends compiled:** %s  \n",  compiled);
    printf("**backends active:** %s  \n\n", active);
    printf("| stage | iters | min ms | median ms | mean ms | p95 ms | p99 ms |\n");
    printf("|---|--:|--:|--:|--:|--:|--:|\n");
    if (a->do_embed && s_embed) {
        printf("| embed | %d | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
               s_embed->n, s_embed->min, s_embed->median, s_embed->mean, s_embed->p95, s_embed->p99);
    }
    if (a->do_e2e && s_e2e) {
        printf("| e2e (detect+align+embed%s) | %d | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
               e2e_have_face ? "" : ", no face",
               s_e2e->n, s_e2e->min, s_e2e->median, s_e2e->mean, s_e2e->p95, s_e2e->p99);
    }
    printf("\n");
}

static void emit_csv(const Args* a,
                     const char* compiled, const char* active,
                     const Stats* s_embed, const Stats* s_e2e,
                     int e2e_have_face) {
    /* Header */
    printf("label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face\n");
    if (a->do_embed && s_embed) {
        printf("\"%s\",\"%s\",\"%s\",embed,%d,%.3f,%.3f,%.3f,%.3f,%.3f,\n",
               a->label, compiled, active,
               s_embed->n, s_embed->min, s_embed->median, s_embed->mean, s_embed->p95, s_embed->p99);
    }
    if (a->do_e2e && s_e2e) {
        printf("\"%s\",\"%s\",\"%s\",e2e,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
               a->label, compiled, active,
               s_e2e->n, s_e2e->min, s_e2e->median, s_e2e->mean, s_e2e->p95, s_e2e->p99,
               e2e_have_face);
    }
}

static void emit_json(const Args* a,
                      const char* compiled, const char* active,
                      const Stats* s_embed, const Stats* s_e2e,
                      int e2e_have_face) {
    printf("{\n");
    printf("  \"label\": \"%s\",\n", a->label);
    printf("  \"backends_compiled\": \"%s\",\n", compiled);
    printf("  \"backends_active\":   \"%s\",\n", active);
    printf("  \"stages\": [\n");
    int first = 1;
#define ROW(name, st, has_face) do { \
        if (!first) printf(",\n"); first = 0; \
        printf("    { \"name\": \"%s\", \"iters\": %d, \"min_ms\": %.3f, \"median_ms\": %.3f, \"mean_ms\": %.3f, \"p95_ms\": %.3f, \"p99_ms\": %.3f%s }", \
               name, (st)->n, (st)->min, (st)->median, (st)->mean, (st)->p95, (st)->p99, \
               (has_face) >= 0 ? (has_face ? ", \"e2e_face\": true" : ", \"e2e_face\": false") : ""); \
    } while (0)
    if (a->do_embed && s_embed) ROW("embed", s_embed, -1);
    if (a->do_e2e   && s_e2e)   ROW("e2e",   s_e2e,   e2e_have_face);
#undef ROW
    printf("\n  ]\n}\n");
}

/* ---- main -------------------------------------------------------------- */

int main(int argc, char** argv) {
    Args a;
    if (parse_args(argc, argv, &a) != 0) return 2;

    /* Detector is optional. If the user explicitly disables it via
     * `--detect ''` OR the file is missing, we silently drop e2e stage. */
    if (a.detect_path) {
        FILE* f = fopen(a.detect_path, "rb");
        if (f) fclose(f);
        else   a.detect_path = NULL;
    }
    int have_detector = (a.detect_path != NULL);
    if (a.do_e2e && !have_detector) a.do_e2e = 0;

    FaceX* fx = facex_init(a.embed_path, a.detect_path, NULL);
    if (!fx) {
        fprintf(stderr, "facex_init failed for embed=%s detect=%s\n",
                a.embed_path, a.detect_path ? a.detect_path : "(none)");
        return 3;
    }

    char compiled[128], active[128];
    print_backends_compiled(compiled, sizeof(compiled));
    print_backends_active(active,     sizeof(active));

    Stats s_embed = {0}, s_e2e = {0};
    int e2e_have_face = 0;

    if (a.do_embed) {
        /* Deterministic input: same pattern as test_mac.c so numbers
         * line up with the existing smoke test. */
        float in[112 * 112 * 3];
        for (int i = 0; i < 112 * 112 * 3; i++)
            in[i] = (float)(i % 256) / 128.0f - 1.0f;
        float emb[512];

        for (int i = 0; i < a.warmup; i++) facex_embed(fx, in, emb);
        double* samples = (double*)malloc(a.iters * sizeof(double));
        for (int i = 0; i < a.iters; i++) {
            double t0 = now_ms();
            facex_embed(fx, in, emb);
            samples[i] = now_ms() - t0;
        }
        s_embed = compute(samples, a.iters);
        free(samples);
    }

    if (a.do_e2e) {
        /* Use the bundled 160×160 face if it's there. Otherwise generate
         * a deterministic non-face frame; the bench still measures the
         * detector cost (NMS, anchor decode) but with 0 faces, which
         * exercises the cheaper code path. */
        uint8_t img[160 * 160 * 3];
        FILE* f = fopen("tests/test_face_160.raw", "rb");
        if (f) {
            size_t n = fread(img, 1, sizeof(img), f);
            fclose(f);
            if (n != sizeof(img)) {
                fprintf(stderr, "warn: short read on tests/test_face_160.raw — using synthetic frame\n");
                for (size_t i = 0; i < sizeof(img); i++) img[i] = (uint8_t)(i & 0xFF);
            }
        } else {
            for (size_t i = 0; i < sizeof(img); i++) img[i] = (uint8_t)(i & 0xFF);
        }

        FaceXResult res[MAX_FACES];
        facex_set_score_threshold(fx, 0.5f);
        for (int i = 0; i < a.warmup; i++) {
            (void)facex_detect(fx, img, 160, 160, res, MAX_FACES);
        }
        double* samples = (double*)malloc(a.iters * sizeof(double));
        for (int i = 0; i < a.iters; i++) {
            double t0 = now_ms();
            int n = facex_detect(fx, img, 160, 160, res, MAX_FACES);
            samples[i] = now_ms() - t0;
            if (i == 0 && n > 0) e2e_have_face = 1;
        }
        s_e2e = compute(samples, a.iters);
        free(samples);
    }

    switch (a.fmt) {
        case FMT_MD:   emit_md(&a, compiled, active, &s_embed, &s_e2e, e2e_have_face);   break;
        case FMT_CSV:  emit_csv(&a, compiled, active, &s_embed, &s_e2e, e2e_have_face);  break;
        case FMT_JSON: emit_json(&a, compiled, active, &s_embed, &s_e2e, e2e_have_face); break;
    }

    facex_free(fx);
    return 0;
}
