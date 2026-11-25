/* Link the native objects with GNU ld --wrap for each allocation below. */
#include "facex.h"
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

static int live, calls, fail_at;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_aligned_alloc(size_t, size_t);
void __real_free(void *);

static void *record(void *p) {
    if (p) live++;
    return p;
}
void *__wrap_malloc(size_t n) {
    return ++calls == fail_at ? NULL : record(__real_malloc(n));
}
void *__wrap_calloc(size_t n, size_t size) {
    return ++calls == fail_at ? NULL : record(__real_calloc(n, size));
}
void *__wrap_aligned_alloc(size_t alignment, size_t size) {
    return ++calls == fail_at ? NULL : record(__real_aligned_alloc(alignment, size));
}
void __wrap_free(void *p) {
    if (p) live--;
    __real_free(p);
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int allocations = 0;
    for (int i = 0; i < 100; i++) {
        calls = 0;
        FaceX *fx = facex_init(argv[1], NULL, NULL);
        assert(fx);
        facex_free(fx);
        assert(live == 0);
        allocations = calls;
    }
    for (fail_at = 1; fail_at <= allocations; fail_at++) {
        calls = 0;
        FaceX *fx = facex_init(argv[1], NULL, NULL);
        assert(!fx && live == 0);
    }
    printf("100 cycles and %d allocation failure paths balanced\n", allocations);
}
