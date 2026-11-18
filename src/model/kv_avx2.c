/* AVX2 unpack/FMA helpers for the quantized KV decode path.
 *
 * INT8 K/V are stored as signed bytes but the query is FP32.  An integer dot
 * product would require quantizing the query (changing attention scores), so
 * the AVX2 path converts 8/16 int8 values per instruction and keeps the FP32
 * dot, using FMA.  Q4 is a true integer path: 32 nibbles are unpacked with
 * 256-bit AND/shift, sign-extended, and accumulated with maddubs/madd.
 * Not called unless uenr_simd_level() == UENR_SIMD_AVX2. */
#include <immintrin.h>
#include <stdint.h>

static inline __m128i q4_nibbles16(__m128i b) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(b, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    __m128i v = _mm_unpacklo_epi8(lo, hi);
    return _mm_sub_epi8(_mm_xor_si128(v, _mm_set1_epi8(8)), _mm_set1_epi8(8));
}

static inline int hsum_epi32_256(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

int uenr_kv_dot_q4_i8_avx2(const int8_t *q, const uint8_t *p, int n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m128i a = q4_nibbles16(_mm_loadl_epi64((const __m128i *)(p + i / 2)));
        __m128i b = q4_nibbles16(_mm_loadl_epi64((const __m128i *)(p + i / 2 + 8)));
        __m256i k8 = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
        __m256i qv = _mm256_loadu_si256((const __m256i *)(q + i));
        __m256i ks = _mm256_sign_epi8(k8, qv);
        __m256i aq = _mm256_abs_epi8(qv);
        __m256i p16 = _mm256_maddubs_epi16(aq, ks);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
    }
    int sum = hsum_epi32_256(acc);
    for (; i < n; i++) {
        uint8_t b = p[i / 2];
        int qv = (i & 1) ? (b >> 4) : (b & 0x0F);
        if (qv >= 8) qv -= 16;
        sum += q[i] * qv;
    }
    return sum;
}

void uenr_kv_add_q4_f32_avx2(float *o, const uint8_t *p, int n,
                             float scale, float w) {
    __m256 wv = _mm256_set1_ps(w * scale);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(p + i / 2));
        __m128i v = q4_nibbles16(b);
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
        __m256 f1 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
        _mm256_storeu_ps(o + i,
            _mm256_fmadd_ps(f0, wv, _mm256_loadu_ps(o + i)));
        _mm256_storeu_ps(o + i + 8,
            _mm256_fmadd_ps(f1, wv, _mm256_loadu_ps(o + i + 8)));
    }
    for (; i < n; i++) {
        uint8_t b = p[i / 2];
        int qv = (i & 1) ? (b >> 4) : (b & 0x0F);
        if (qv >= 8) qv -= 16;
        o[i] += w * (float)qv * scale;
    }
}

float uenr_kv_dot_int8_f32_avx2(const float *q, const int8_t *k, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b = _mm_loadu_si128((const __m128i *)(k + i));
        __m256 k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        __m256 k1 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(b, 8)));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), k0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), k1, a1);
    }
    float s = 0.0f;
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(_mm256_add_ps(a0, a1)),
                           _mm256_extractf128_ps(_mm256_add_ps(a0, a1), 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += q[i] * (float)k[i];
    return s;
}

void uenr_kv_add_int8_f32_avx2(float *o, const int8_t *k, int n, float w) {
    __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b = _mm_loadu_si128((const __m128i *)(k + i));
        __m256 k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        __m256 k1 = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(b, 8)));
        _mm256_storeu_ps(o + i,
            _mm256_fmadd_ps(k0, wv, _mm256_loadu_ps(o + i)));
        _mm256_storeu_ps(o + i + 8,
            _mm256_fmadd_ps(k1, wv, _mm256_loadu_ps(o + i + 8)));
    }
    for (; i < n; i++) o[i] += w * (float)k[i];
}
