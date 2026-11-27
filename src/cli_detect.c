/* Binary detector transport used by the Python subprocess backend. */
#include "facex.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static int read_limit(const char *s, int *value) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || end == s || *end || n < 0 || n > INT_MAX) return 0;
    *value = (int)n;
    return 1;
}

static int read_threshold(const char *s, float *value) {
    char *end;
    float n = strtof(s, &end);
    if (end == s || *end || !isfinite(n) || n < 0 || n > 1) return 0;
    *value = n;
    return 1;
}

int facex_cli_detect(int argc, char **argv) {
    const char *detector = NULL, *embedder = NULL;
    int limit = 20, min_side = 16, embed = 1;
    float score = 0.5f, nms = 0.4f;
    if (argc < 3 || strcmp(argv[2], "--pipe")) return 2;
    for (int i = 3; i < argc; i++) {
        const char *flag = argv[i];
        if (!strcmp(flag, "--no-embeddings")) { embed = 0; continue; }
        if (++i == argc) return 2;
        const char *value = argv[i];
        if (!strcmp(flag, "--detect-weights")) detector = value;
        else if (!strcmp(flag, "--embed-weights")) embedder = value;
        else if (!strcmp(flag, "--max-faces") && read_limit(value, &limit)) continue;
        else if (!strcmp(flag, "--min-box-side") && read_limit(value, &min_side) &&
                 (min_side == 8 || min_side == 12 || min_side == 16)) continue;
        else if (!strcmp(flag, "--score-threshold") && read_threshold(value, &score)) continue;
        else if (!strcmp(flag, "--nms-threshold") && read_threshold(value, &nms)) continue;
        else return 2;
    }
    if (!detector) { fputs("Missing detector weights\n", stderr); return 2; }
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    int32_t size[2];
    if (fread(size, sizeof(size), 1, stdin) != 1 || size[0] <= 0 || size[1] <= 0 ||
        (size_t)size[0] > SIZE_MAX / 3 / (size_t)size[1]) return 1;
    size_t bytes = (size_t)size[0] * size[1] * 3;
    uint8_t *rgb = malloc(bytes);
    if (!rgb) return 1;
    int ok = 0;
    if (fread(rgb, 1, bytes, stdin) == bytes) {
        FaceX *fx = facex_init(embed ? embedder : NULL, detector, getenv("LICENSE_KEY"));
        if (fx) {
            FaceXResult faces[32];
            facex_set_nms_threshold(fx, nms);
            facex_set_min_box_side(fx, min_side);
            int32_t n = facex_detect_prepared_ex(fx, rgb, size[0], size[1], rgb,
                size[0], size[1], size[0], size[1], faces, limit, embed, score);
            ok = n >= 0 && fwrite(&n, sizeof(n), 1, stdout) == 1 &&
                 fwrite(faces, sizeof(*faces), n, stdout) == (size_t)n && fflush(stdout) == 0;
            facex_free(fx);
        }
    }
    free(rgb);
    return ok ? 0 : 1;
}
