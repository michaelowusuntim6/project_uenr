/* AVX2+FMA build of swiglu.c.  Runtime-selected; never executed on AVX1 CPUs. */
#define UENR_SIMD_SUFFIX avx2
#include "swiglu.c"
