#include "uenr/simd_dispatch.h"

#include <stdio.h>
#include <string.h>

static int g_level = -1; /* -1 = not yet detected */

/* Implemented in src/ops/dispatch_ops.c; refreshes its function table. */
void uenr_ops_refresh(void);

int uenr_cpu_has_avx2_fma(void) {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    static int cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") &&
                 __builtin_cpu_supports("fma");
    }
    return cached;
#else
    return 0;
#endif
}

UenrResult uenr_simd_configure(const char *mode) {
    if (!mode || !mode[0] || !strcmp(mode, "auto"))
        g_level = uenr_cpu_has_avx2_fma() ? UENR_SIMD_AVX2 : UENR_SIMD_AVX1;
    else if (!strcmp(mode, "avx1"))
        g_level = UENR_SIMD_AVX1;
    else if (!strcmp(mode, "avx2")) {
        if (!uenr_cpu_has_avx2_fma()) {
            fprintf(stderr, "simd: avx2 requested but the CPU lacks AVX2+FMA\n");
            return UENR_ERR_INVALID;
        }
        g_level = UENR_SIMD_AVX2;
    } else {
        fprintf(stderr, "simd: unknown mode '%s' (auto|avx1|avx2)\n", mode);
        return UENR_ERR_INVALID;
    }
    uenr_ops_refresh();
    return UENR_OK;
}

int uenr_simd_level(void) {
    if (g_level < 0)
        g_level = uenr_cpu_has_avx2_fma() ? UENR_SIMD_AVX2 : UENR_SIMD_AVX1;
    return g_level;
}

const char *uenr_simd_name(void) {
    return uenr_simd_level() == UENR_SIMD_AVX2 ? "avx2+fma" : "avx1";
}
