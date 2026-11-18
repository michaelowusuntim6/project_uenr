/* Include before "uenr/ops.h" in an op implementation.
 *
 * The ordinary build compiles each *.c with UENR_SIMD_SUFFIX=avx1; the
 * *_avx2.c translation units define UENR_SIMD_SUFFIX=avx2 and then include
 * the same source, so there is exactly one implementation of each kernel and
 * no drift between the two ISA paths.  Public entry points live in
 * dispatch_ops.c and select between these suffixed symbols. */
#ifndef UENR_SIMD_SUFFIX
#define UENR_SIMD_SUFFIX avx1
#endif

#define UENR_CAT2_(a, b) a##b
#define UENR_CAT2(a, b) UENR_CAT2_(a, b)
#define UENR_IMPL2_(s, n) uenr_##s##_##n
#define UENR_IMPL2(s, n) UENR_IMPL2_(s, n)
#define UENR_OP(n) UENR_IMPL2(UENR_SIMD_SUFFIX, n)
