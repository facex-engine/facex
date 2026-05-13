/*
 * cpu_features.c — runtime CPU feature detection.
 *
 * macOS arm64 path: sysctlbyname against hw.optional.arm.FEAT_*.
 * Other platforms: stubs that always say "no".
 */

#include "cpu_features.h"

#include <stdatomic.h>
#include <stddef.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#elif defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP2_SME
#define HWCAP2_SME (1UL << 23)
#endif
#ifndef HWCAP2_SME2
#define HWCAP2_SME2 (1UL << 37)
#endif
#endif

/* Detection results are computed once and cached. We use atomics so the
 * detection function is safe to call from any thread without synchronization
 * cost on the fast path. */
typedef struct {
    atomic_int probed;   /* 0 until first probe completes */
    atomic_int has_sme;
    atomic_int has_sme2;
    atomic_int vl_bits;
    atomic_int disabled; /* set by facex_disable_sme() */
} CpuState;

static CpuState g_cpu;

#if defined(__APPLE__) && defined(__aarch64__)
static int sysctl_probe(const char* name) {
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) return 0;
    return v ? 1 : 0;
}

static int sysctl_int(const char* name, int fallback) {
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) return fallback;
    return v;
}
#endif

static void probe_once(void) {
    if (atomic_load_explicit(&g_cpu.probed, memory_order_acquire)) return;

    int sme = 0, sme2 = 0, vl = 0;

#if defined(__APPLE__) && defined(__aarch64__)
    sme  = sysctl_probe("hw.optional.arm.FEAT_SME");
    sme2 = sysctl_probe("hw.optional.arm.FEAT_SME2");
    /* Apple doesn't currently surface SVL via sysctl. M4 is documented at
     * 512 bits. We hardcode that hint when SME is on; callers who care
     * compute the runtime SVL via svcntw() from inside a streaming function. */
    if (sme) vl = 512;
#elif defined(__linux__) && defined(__aarch64__)
    unsigned long h2 = getauxv(AT_HWCAP2);
    sme  = (h2 & HWCAP2_SME)  ? 1 : 0;
    sme2 = (h2 & HWCAP2_SME2) ? 1 : 0;
#endif

    atomic_store_explicit(&g_cpu.has_sme,  sme,  memory_order_relaxed);
    atomic_store_explicit(&g_cpu.has_sme2, sme2, memory_order_relaxed);
    atomic_store_explicit(&g_cpu.vl_bits,  vl,   memory_order_relaxed);
    atomic_store_explicit(&g_cpu.probed,   1,    memory_order_release);
}

int facex_has_sme(void) {
    probe_once();
    if (atomic_load_explicit(&g_cpu.disabled, memory_order_acquire)) return 0;
    return atomic_load_explicit(&g_cpu.has_sme, memory_order_relaxed);
}

int facex_has_sme2(void) {
    probe_once();
    if (atomic_load_explicit(&g_cpu.disabled, memory_order_acquire)) return 0;
    return atomic_load_explicit(&g_cpu.has_sme2, memory_order_relaxed);
}

void facex_disable_sme(void) {
    atomic_store_explicit(&g_cpu.disabled, 1, memory_order_release);
}

int facex_sme_vl_bits(void) {
    probe_once();
    return atomic_load_explicit(&g_cpu.vl_bits, memory_order_relaxed);
}
