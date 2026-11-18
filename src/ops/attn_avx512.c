/* AVX-512 attention build (16-wide dot and AXPY). Runtime-selected only. */
#define UENR_SIMD_SUFFIX avx512
#include "attn.c"
