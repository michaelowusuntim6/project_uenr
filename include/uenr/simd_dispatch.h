#ifndef UENR_SIMD_DISPATCH_H
#define UENR_SIMD_DISPATCH_H

#include "uenr/result.h"

/* Runtime SIMD selection.  Both paths are always linked: the Ivy Bridge
 * (AVX1+F16C) kernels are the laptop path, the AVX2+FMA copies are selected
 * only when the CPU reports both features.  The selection is process-wide and
 * is read by the hot-kernel dispatchers in src/ops. */
enum {
    UENR_SIMD_AVX1 = 0,
    UENR_SIMD_AVX2 = 1,
    UENR_SIMD_AVX512 = 2
};

/* "auto" (default), "avx1", "avx2" or "avx512".  Returns UENR_ERR_INVALID for an
 * unknown name and for an explicit avx2 request on a CPU without AVX2+FMA;
 * the previous selection is left untouched on error. */
UenrResult uenr_simd_configure(const char *mode);

/* Current level.  Lazily auto-detects on first use so unit tests and small
 * tools that never call uenr_simd_configure still get the right path. */
int        uenr_simd_level(void);
const char *uenr_simd_name(void);
int        uenr_cpu_has_avx2_fma(void);
int        uenr_cpu_has_avx512(void);

#endif /* UENR_SIMD_DISPATCH_H */
