/*
 * pool.c — Lock-free thread pool (from facex pattern).
 *
 * Atomic counter + WaitOnAddress/futex. Workers spin ~1µs before sleeping.
 * Main thread participates in work.
 */

#include "nn2_internal.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ATOMIC_LOAD(p)       __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define ATOMIC_STORE(p, v)   __atomic_store_n(p, v, __ATOMIC_RELEASE)
#define ATOMIC_ADD(p, v)     __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL)
#elif defined(_MSC_VER)
#include <intrin.h>
#define ATOMIC_LOAD(p)       (*(volatile int*)(p))
#define ATOMIC_STORE(p, v)   (*(volatile int*)(p) = (v))
#define ATOMIC_ADD(p, v)     _InterlockedExchangeAdd((volatile long*)(p), (v))
#else
#include <stdatomic.h>
#define ATOMIC_LOAD(p)       atomic_load(p)
#define ATOMIC_STORE(p, v)   atomic_store(p, v)
#define ATOMIC_ADD(p, v)     atomic_fetch_add(p, v)
#endif

#define MAX_THREADS 16

typedef struct {
    nn2_task_fn fn;
    void*       ctx;
    volatile int next_idx;
    int          total;
    int          grain;
    volatile int done_count;
    int          n_workers;
    volatile int phase;
} TaskState;

static TaskState g_task;

typedef struct {
    int id;
#ifdef _WIN32
    HANDLE handle;
#else
    pthread_t handle;
#endif
} Worker;

static Worker g_workers[MAX_THREADS];
static int g_n_threads = 0;
static volatile int g_shutdown = 0;

static
#ifdef _WIN32
unsigned __stdcall
#else
void*
#endif
worker_fn(void* arg)
{
    int last_phase = 0;

    while (!ATOMIC_LOAD(&g_shutdown)) {
        int phase = ATOMIC_LOAD(&g_task.phase);
        if (phase == last_phase) {
            int spins = 100;
            while (spins-- > 0) {
                phase = ATOMIC_LOAD(&g_task.phase);
                if (phase != last_phase) break;
#ifdef _WIN32
                YieldProcessor();
#else
                __asm__ __volatile__("pause");
#endif
            }
            if (phase == last_phase) {
#ifdef _WIN32
                WaitOnAddress((volatile void*)&g_task.phase, &last_phase,
                              sizeof(int), INFINITE);
#else
                syscall(SYS_futex, &g_task.phase, FUTEX_WAIT, last_phase,
                        NULL, NULL, 0);
#endif
                continue;
            }
        }
        last_phase = phase;

        while (1) {
            int idx = ATOMIC_ADD(&g_task.next_idx, g_task.grain);
            if (idx >= g_task.total) break;
            int end = idx + g_task.grain;
            if (end > g_task.total) end = g_task.total;
            g_task.fn(g_task.ctx, idx, end);
        }

        ATOMIC_ADD(&g_task.done_count, 1);
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

void nn2_tp_init(int n_threads)
{
    if (n_threads <= 0) {
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        n_threads = si.dwNumberOfProcessors;
#else
        n_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
        if (n_threads < 1) n_threads = 1;
    }

    g_n_threads = n_threads;
    int n_workers = n_threads - 1;
    memset(&g_task, 0, sizeof(g_task));
    ATOMIC_STORE(&g_shutdown, 0);

    for (int i = 0; i < n_workers; i++) {
        g_workers[i].id = i;
#ifdef _WIN32
        g_workers[i].handle = (HANDLE)_beginthreadex(
            NULL, 0, worker_fn, &g_workers[i], 0, NULL);
#else
        pthread_create(&g_workers[i].handle, NULL, worker_fn, &g_workers[i]);
#endif
    }
}

static volatile int g_dispatch_count = 0;
int nn2_tp_dispatch_count(void) { return g_dispatch_count; }
void nn2_tp_reset_dispatch_count(void) { g_dispatch_count = 0; }

void nn2_tp_parallel_for(nn2_task_fn fn, void* ctx, int total, int grain)
{
    if (total <= 0) return;
    if (grain <= 0) grain = 1;

    int n_workers = g_n_threads - 1;
    if (n_workers <= 0 || total <= grain) {
        fn(ctx, 0, total);
        return;
    }
    g_dispatch_count++;

    g_task.fn = fn;
    g_task.ctx = ctx;
    g_task.total = total;
    g_task.grain = grain;
    ATOMIC_STORE(&g_task.next_idx, 0);
    ATOMIC_STORE(&g_task.done_count, 0);
    g_task.n_workers = n_workers;

    ATOMIC_ADD(&g_task.phase, 1);
#ifdef _WIN32
    WakeByAddressAll((void*)&g_task.phase);
#else
    syscall(SYS_futex, &g_task.phase, FUTEX_WAKE, n_workers, NULL, NULL, 0);
#endif

    /* Main thread participates */
    while (1) {
        int idx = ATOMIC_ADD(&g_task.next_idx, grain);
        if (idx >= total) break;
        int end = idx + grain;
        if (end > total) end = total;
        fn(ctx, idx, end);
    }

    /* Spin-wait for workers — pure spin, no sleep (fastest for short tasks) */
    while (ATOMIC_LOAD(&g_task.done_count) < n_workers) {
#ifdef _WIN32
        YieldProcessor();
#else
        __asm__ __volatile__("pause");
#endif
    }
}

void nn2_tp_destroy(void)
{
    int n_workers = g_n_threads - 1;
    ATOMIC_STORE(&g_shutdown, 1);
    ATOMIC_ADD(&g_task.phase, 1);
#ifdef _WIN32
    WakeByAddressAll((void*)&g_task.phase);
    for (int i = 0; i < n_workers; i++)
        WaitForSingleObject(g_workers[i].handle, INFINITE);
#else
    syscall(SYS_futex, &g_task.phase, FUTEX_WAKE, MAX_THREADS, NULL, NULL, 0);
    for (int i = 0; i < n_workers; i++)
        pthread_join(g_workers[i].handle, NULL);
#endif
    g_n_threads = 0;
}

int nn2_tp_num_threads(void) { return g_n_threads; }
