/*
 * threadpool_pthread.c — Persistent pthread thread pool.
 *
 * Drop-in replacement for threadpool.c on platforms without futex /
 * WaitOnAddress (macOS, FreeBSD, generic POSIX). Uses standard
 * pthread mutex + condition variable for sleep / wake. Higher
 * dispatch latency than the futex impl (~5–10 µs vs <500 ns) but
 * the GEMM work items are large enough that the overhead is amortized.
 *
 * API matches threadpool.h.
 */

#include "threadpool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#define TP_MAX_THREADS 32

typedef struct {
    pthread_t handle;
    int id;
} Worker;

static Worker         g_workers[TP_MAX_THREADS];
static int            g_n_threads = 0;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv_task   = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cv_done   = PTHREAD_COND_INITIALIZER;

static tp_task_fn      g_task_fn = NULL;
static void*           g_task_ctx = NULL;
static int             g_task_total = 0;
static int             g_task_grain = 1;
static atomic_int      g_task_next;          /* next chunk start */
static atomic_int      g_task_active;        /* workers still running this batch */
static atomic_int      g_phase;              /* monotonic phase counter */
static atomic_int      g_shutdown;

static void* worker_fn(void* arg) {
    (void)arg;
    int last_phase = 0;

    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (atomic_load(&g_phase) == last_phase &&
               !atomic_load(&g_shutdown)) {
            pthread_cond_wait(&g_cv_task, &g_mu);
        }
        if (atomic_load(&g_shutdown)) {
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        last_phase = atomic_load(&g_phase);
        pthread_mutex_unlock(&g_mu);

        /* Grab chunks until exhausted */
        for (;;) {
            int start = atomic_fetch_add(&g_task_next, g_task_grain);
            if (start >= g_task_total) break;
            int end = start + g_task_grain;
            if (end > g_task_total) end = g_task_total;
            g_task_fn(g_task_ctx, start, end);
        }

        /* Mark this worker done; signal master if last */
        if (atomic_fetch_sub(&g_task_active, 1) == 1) {
            pthread_mutex_lock(&g_mu);
            pthread_cond_signal(&g_cv_done);
            pthread_mutex_unlock(&g_mu);
        }
    }
}

void tp_init(int n_threads) {
    if (g_n_threads > 0) return; /* idempotent */

    if (n_threads <= 0) {
#if defined(__APPLE__)
        int n = 0; size_t sz = sizeof(n);
        if (sysctlbyname("hw.activecpu", &n, &sz, NULL, 0) != 0 || n <= 0) n = 4;
        n_threads = n;
#else
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        n_threads = (n > 0) ? (int)n : 4;
#endif
    }
    if (n_threads > TP_MAX_THREADS) n_threads = TP_MAX_THREADS;

    atomic_store(&g_phase, 0);
    atomic_store(&g_shutdown, 0);
    atomic_store(&g_task_active, 0);
    g_n_threads = n_threads;

    for (int i = 0; i < n_threads; i++) {
        g_workers[i].id = i;
        pthread_create(&g_workers[i].handle, NULL, worker_fn, &g_workers[i]);
    }
}

void tp_parallel_for(tp_task_fn fn, void* ctx, int total, int grain) {
    if (total <= 0) return;
    if (grain <= 0) grain = 1;

    /* Single-threaded fast path */
    if (g_n_threads <= 1) {
        for (int i = 0; i < total; i += grain) {
            int end = i + grain;
            if (end > total) end = total;
            fn(ctx, i, end);
        }
        return;
    }

    pthread_mutex_lock(&g_mu);
    g_task_fn    = fn;
    g_task_ctx   = ctx;
    g_task_total = total;
    g_task_grain = grain;
    atomic_store(&g_task_next, 0);
    atomic_store(&g_task_active, g_n_threads);
    atomic_fetch_add(&g_phase, 1);
    pthread_cond_broadcast(&g_cv_task);

    while (atomic_load(&g_task_active) > 0) {
        pthread_cond_wait(&g_cv_done, &g_mu);
    }
    pthread_mutex_unlock(&g_mu);
}

void tp_destroy(void) {
    if (g_n_threads <= 0) return;
    pthread_mutex_lock(&g_mu);
    atomic_store(&g_shutdown, 1);
    pthread_cond_broadcast(&g_cv_task);
    pthread_mutex_unlock(&g_mu);
    for (int i = 0; i < g_n_threads; i++) {
        pthread_join(g_workers[i].handle, NULL);
    }
    g_n_threads = 0;
}

int tp_num_threads(void) { return g_n_threads; }
