#ifndef UENR_SIMD_H
#define UENR_SIMD_H

#include <immintrin.h>
#include <stdint.h>

/*
 * Zero-cost SIMD wrappers. Target: Ivy Bridge (i5-3210M).
 * AVX 256-bit FP is available; AVX2 is NOT; FMA is NOT.
 * Every 256-bit integer operation is done as two 128-bit halves (AVX1-safe).
 */

#if defined(__AVX__)
typedef __m256 uenr_v8f;
#  define UENR_LANES 8
#else
typedef __m128 uenr_v8f;
#  define UENR_LANES 4
#endif

/* ---- aligned / unaligned load & store ---------------------------------- */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_load(const float *p) {
#if defined(__AVX__)
    return _mm256_load_ps(p);
#else
    return _mm_load_ps(p);
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_loadu(const float *p) {
#if defined(__AVX__)
    return _mm256_loadu_ps(p);
#else
    return _mm_loadu_ps(p);
#endif
}

static inline __attribute__((always_inline)) void uenr_v_store(float *p, uenr_v8f v) {
#if defined(__AVX__)
    _mm256_store_ps(p, v);
#else
    _mm_store_ps(p, v);
#endif
}

static inline __attribute__((always_inline)) void uenr_v_storeu(float *p, uenr_v8f v) {
#if defined(__AVX__)
    _mm256_storeu_ps(p, v);
#else
    _mm_storeu_ps(p, v);
#endif
}

/* ---- basic arithmetic --------------------------------------------------- */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_set1(float x) {
#if defined(__AVX__)
    return _mm256_set1_ps(x);
#else
    return _mm_set1_ps(x);
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_zero(void) {
    return uenr_v_set1(0.0f);
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_add(uenr_v8f a, uenr_v8f b) {
#if defined(__AVX__)
    return _mm256_add_ps(a, b);
#else
    return _mm_add_ps(a, b);
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_sub(uenr_v8f a, uenr_v8f b) {
#if defined(__AVX__)
    return _mm256_sub_ps(a, b);
#else
    return _mm_sub_ps(a, b);
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_mul(uenr_v8f a, uenr_v8f b) {
#if defined(__AVX__)
    return _mm256_mul_ps(a, b);
#else
    return _mm_mul_ps(a, b);
#endif
}

/* Polynomial helper: deliberately mul+add even in AVX2 translation units.
 * The exp/log/sin/cos polynomials are documented in FORMULAS.md and use this
 * helper so both SIMD paths evaluate them in the same order (bit-identical). */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_muladd(uenr_v8f a, uenr_v8f b,
                                                                     uenr_v8f c) {
    return uenr_v_add(uenr_v_mul(a, b), c);
}

/* Linear-algebra FMA helpers.  In the Ivy Bridge build these expand to the
 * exact mul+add pair the AVX1 kernels always used; only AVX2 translation
 * units compiled with -mfma emit hardware FMA. */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_fmadd(uenr_v8f a, uenr_v8f b,
                                                                   uenr_v8f c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
static inline __attribute__((always_inline)) uenr_v8f uenr_v_fmsub(uenr_v8f a, uenr_v8f b,
                                                                   uenr_v8f c) {
#if defined(__FMA__)
    return _mm256_fmsub_ps(a, b, c);
#else
    return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
#endif
}
static inline __attribute__((always_inline)) uenr_v8f uenr_v_fnmadd(uenr_v8f a, uenr_v8f b,
                                                                    uenr_v8f c) {
#if defined(__FMA__)
    return _mm256_fnmadd_ps(a, b, c);
#else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_max(uenr_v8f a, uenr_v8f b) {
#if defined(__AVX__)
    return _mm256_max_ps(a, b);
#else
    return _mm_max_ps(a, b);
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_min(uenr_v8f a, uenr_v8f b) {
#if defined(__AVX__)
    return _mm256_min_ps(a, b);
#else
    return _mm_min_ps(a, b);
#endif
}

/* ---- horizontal reductions --------------------------------------------- */
static inline __attribute__((always_inline)) float uenr_v_hsum(uenr_v8f v) {
#if defined(__AVX__)
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
#else
    __m128 s = v;
#endif
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline __attribute__((always_inline)) float uenr_v_hmax(uenr_v8f v) {
#if defined(__AVX__)
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
#else
    __m128 m = v;
#endif
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    return _mm_cvtss_f32(m);
}

/* ---- reciprocal / rsqrt with one Newton-Raphson step -------------------- */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_rcp(uenr_v8f x) {
#if defined(__AVX__)
    __m256 y = _mm256_rcp_ps(x);
    /* y = y * (2 - x*y) */
    __m256 t = _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(x, y));
    return _mm256_mul_ps(y, t);
#else
    __m128 y = _mm_rcp_ps(x);
    __m128 t = _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(x, y));
    return _mm_mul_ps(y, t);
#endif
}

static inline __attribute__((always_inline)) uenr_v8f uenr_v_rsqrt(uenr_v8f x) {
#if defined(__AVX__)
    __m256 y = _mm256_rsqrt_ps(x);
    __m256 half_x = _mm256_mul_ps(_mm256_set1_ps(0.5f), x);
    __m256 t = _mm256_sub_ps(_mm256_set1_ps(1.5f),
                             _mm256_mul_ps(_mm256_mul_ps(half_x, y), y));
    return _mm256_mul_ps(y, t);
#else
    __m128 y = _mm_rsqrt_ps(x);
    __m128 half_x = _mm_mul_ps(_mm_set1_ps(0.5f), x);
    __m128 t = _mm_sub_ps(_mm_set1_ps(1.5f),
                          _mm_mul_ps(_mm_mul_ps(half_x, y), y));
    return _mm_mul_ps(y, t);
#endif
}

/* ---- AVX1-safe 2^k for float k (integer work split into 128-bit halves) - */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_pow2i(uenr_v8f kf) {
#if defined(__AVX2__)
    __m256i ki = _mm256_cvtps_epi32(kf);
    ki = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_castsi256_ps(ki);
#elif defined(__AVX__)
    __m256i ki = _mm256_cvtps_epi32(kf);          /* AVX */
    __m256  kif = _mm256_castsi256_ps(ki);
    __m128  lo = _mm256_castps256_ps128(kif);
    __m128  hi = _mm256_extractf128_ps(kif, 1);
    __m128i lo_i = _mm_castps_si128(lo);
    __m128i hi_i = _mm_castps_si128(hi);
    lo_i = _mm_slli_epi32(_mm_add_epi32(lo_i, _mm_set1_epi32(127)), 23);
    hi_i = _mm_slli_epi32(_mm_add_epi32(hi_i, _mm_set1_epi32(127)), 23);
    __m256 out = _mm256_castps128_ps256(_mm_castsi128_ps(lo_i));
    return _mm256_insertf128_ps(out, _mm_castsi128_ps(hi_i), 1);
#else
    __m128i ki = _mm_cvtps_epi32(kf);
    ki = _mm_slli_epi32(_mm_add_epi32(ki, _mm_set1_epi32(127)), 23);
    return _mm_castsi128_ps(ki);
#endif
}

/* Scalar FMA for the linear-algebra paths; exact mul+add on Ivy Bridge. */
static inline __attribute__((always_inline)) float uenr_fmaf(float a, float b, float c) {
#if defined(__FMA__)
    return __builtin_fmaf(a, b, c);
#else
    return a * b + c;
#endif
}

/* ---- FP16 storage conversion (F16C) ------------------------------------ */
static inline __attribute__((always_inline)) uenr_v8f uenr_v_load_f16(const uint16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

static inline __attribute__((always_inline)) void uenr_v_store_f16(uint16_t *p, uenr_v8f v) {
    _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

#endif /* UENR_SIMD_H */
