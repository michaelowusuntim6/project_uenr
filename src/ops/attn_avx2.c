/* AVX2+FMA build of attn.c.  Runtime-selected; never executed on AVX1 CPUs. */
#define UENR_SIMD_SUFFIX avx2
#include "attn.c"
