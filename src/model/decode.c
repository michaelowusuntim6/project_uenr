#include "uenr/model.h"
#include "uenr/math.h"
#include "uenr/ops.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static float *ac(size_t n) {
    size_t bytes = ((n * sizeof(float)) + 63) & ~(size_t)63;
    return (float *)aligned_alloc(64, bytes ? bytes : 64);
}
static int *ai(size_t n) {
    size_t bytes = ((n * sizeof(int)) + 63) & ~(size_t)63;
    return (int *)aligned_alloc(64, bytes ? bytes : 64);
}

UenrResult uenr_kv_cache_init(UenrKVCache *c, const UenrConfig *cfg, int cap) {
    if (!c || !cfg || cap <= 0) return UENR_ERR_INVALID;
    memset(c, 0, sizeof(*c));
    int L = cfg->model.layers;
    int G = cfg->model.kv_heads;
    int hd = cfg->model.head_dim;
    int KV = G * hd;
    int C = cfg->model.hidden;
    int F = cfg->model.ffn;
    int EF = cfg->moe.expert_ffn;
    int E = cfg->moe.num_experts;
    int K = cfg->moe.experts_per_tok;
    int maxF = (F > EF) ? F : EF;
    c->L = L; c->KV = KV; c->cap = cap; c->len = 0;
    c->int8 = cfg->model.kv_adapter ? 0 : cfg->infer.kv_int8;
    c->q4 = cfg->model.kv_adapter ? 0 : (!c->int8 && cfg->infer.kv_q4);
    c->q2 = cfg->model.kv_adapter ? 0 : (!c->int8 && !c->q4 && cfg->infer.kv_q2);
    c->rotate = c->q4 && cfg->infer.kv_rotate;
    c->recent = cfg->infer.kv_recent;
    c->sinks = cfg->infer.kv_sinks;
    c->evict = cfg->infer.kv_evict;
    c->ew = cfg->infer.kv_evict_window;
    c->lr = (cfg->model.kv_adapter ? 0 : (!c->int8 && !c->q4 && !c->q2 && cfg->infer.kv_lowrank > 0))
                ? (cfg->infer.kv_lowrank < hd ? cfg->infer.kv_lowrank : hd) : 0;
    c->fp16 = !c->int8 && !c->q4 && !c->q2 && !c->lr && cfg->infer.kv_fp16;
    c->G = G; c->hd = hd;
    c->n_slots = uenr_active_kv_layers(cfg);
    c->type = ai(L); c->donor = ai(L); c->shared = ai(L);
    if (!c->type || !c->donor || !c->shared) { uenr_kv_cache_free(c); return UENR_ERR_ALLOC; }
    for (int l = 0; l < L; l++) {
        c->type[l] = uenr_layer_is_sliding(cfg, l) ? 0 : 1;
        c->shared[l] = uenr_layer_is_shared(cfg, l);
        c->donor[l] = c->shared[l] ? uenr_layer_donor(cfg, l) : l;
    }
    c->slot_off = ai(L); c->slot_cap = ai(L);
    if (!c->slot_off || !c->slot_cap) { uenr_kv_cache_free(c); return UENR_ERR_ALLOC; }
    {
        int W = cfg->model.sliding_window;
        int total = 0;
        for (int l = 0; l < L; l++) {
            if (c->shared[l]) continue;
            int sc = cap;
            if (c->type[l] == 0 && W > 0 && W < cap) sc = W;
            else if (c->type[l] != 0 && c->evict &&
                     c->sinks + c->ew > 0 && c->sinks + c->ew < cap)
                sc = c->sinks + c->ew;
            c->slot_off[l] = total;
            c->slot_cap[l] = sc;
            total += sc;
        }
        for (int l = 0; l < L; l++) {
            if (!c->shared[l]) continue;
            c->slot_off[l] = c->slot_off[c->donor[l]];
            c->slot_cap[l] = c->slot_cap[c->donor[l]];
        }
        c->slot_rows = (size_t)total;
    }
    const size_t nrow = c->slot_rows * KV;
    if (c->int8) {
        c->k8 = (int8_t *)aligned_alloc(64, ((nrow + 63) & ~(size_t)63));
        c->v8 = (int8_t *)aligned_alloc(64, ((nrow + 63) & ~(size_t)63));
        c->ks = ac((size_t)c->n_slots * G);
        c->vs = ac((size_t)c->n_slots * G);
        if (c->k8) memset(c->k8, 0, nrow);
        if (c->v8) memset(c->v8, 0, nrow);
    } else if (c->q4) {
        c->k4 = (uint8_t *)aligned_alloc(64, (((nrow / 2) + 63) & ~(size_t)63));
        c->v4 = (uint8_t *)aligned_alloc(64, (((nrow / 2) + 63) & ~(size_t)63));
        c->k4s = ac(c->slot_rows * G);
        c->v4s = ac(c->slot_rows * G);
    } else if (c->q2) {
        c->k2 = (uint8_t *)aligned_alloc(64, (((nrow / 4) + 63) & ~(size_t)63));
        c->v2 = (uint8_t *)aligned_alloc(64, (((nrow / 4) + 63) & ~(size_t)63));
        c->k2s = ac(c->slot_rows * G);
        c->v2s = ac(c->slot_rows * G);
        if (c->k2) memset(c->k2, 0, nrow / 4);
        if (c->v2) memset(c->v2, 0, nrow / 4);
        c->krec = ac((size_t)c->n_slots * c->recent * KV);
        c->vrec = ac((size_t)c->n_slots * c->recent * KV);
        c->ksink = ac((size_t)c->n_slots * c->sinks * KV);
        c->vsink = ac((size_t)c->n_slots * c->sinks * KV);
        c->recpos = ai((size_t)c->n_slots * c->recent);
        if (c->recpos) memset(c->recpos, 0xFF, (size_t)c->n_slots * c->recent * sizeof(int));
    }
    if (c->lr) {
        c->klr = ac(c->slot_rows * G * c->lr);
        c->vlr = ac(c->slot_rows * G * c->lr);
        c->qlr = ac((size_t)cfg->model.heads * c->lr);
    }
    if (c->q4 || c->q2) {
        c->q8v = (int8_t *)aligned_alloc(64, (((size_t)C + 63) & ~(size_t)63));
        c->q8s = ac(cfg->model.heads);
        if (c->k4) memset(c->k4, 0, nrow / 2);
        if (c->v4) memset(c->v4, 0, nrow / 2);
    } else if (c->fp16) {
        c->k16 = (uint16_t *)aligned_alloc(64, (((nrow * sizeof(uint16_t)) + 63) & ~(size_t)63));
        c->v16 = (uint16_t *)aligned_alloc(64, (((nrow * sizeof(uint16_t)) + 63) & ~(size_t)63));
        if (c->k16) memset(c->k16, 0, nrow * sizeof(uint16_t));
        if (c->v16) memset(c->v16, 0, nrow * sizeof(uint16_t));
    } else if (cfg->infer.kv_offload) {
        static int off_id;
        char path[600];
        snprintf(path, sizeof(path), "%s.%d.%d", cfg->infer.kv_offload_file,
                 (int)getpid(), off_id++);
        size_t one = nrow * sizeof(float);
        size_t bytes = 2 * one;
        int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        void *map = MAP_FAILED;
        if (fd >= 0) {
            if (ftruncate(fd, (off_t)bytes) == 0)
                map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
        }
        if (map != MAP_FAILED) {
            c->kmap = map;
            c->vmap = (char *)map + one;
            c->mapsz = bytes;
            c->k = (float *)c->kmap;
            c->v = (float *)c->vmap;
        } else {
            c->k = ac(nrow);
            c->v = ac(nrow);
        }
    } else {
        c->k = ac(nrow);
        c->v = ac(nrow);
    }
    c->x = ac(C); c->xa = ac(C); c->q = ac(C); c->attn = ac(C);
    c->proj = ac(C); c->x2 = ac(C); c->xf = ac(C); c->eout = ac(C); c->ys = ac(C);
    c->krow = ac(KV); c->vrow = ac(KV);
    c->gbuf = ac(maxF); c->ubuf = ac(maxF);
    c->router_logits = ac(E);
    c->xfinal = ac(C); c->logits = ac(cfg->model.vocab_size);
    c->scores = ac(cap);
    c->xprev = ac(C);
    c->early = cfg->infer.early_exit;
    c->sparse_k = cfg->infer.kv_sparse_k;
    c->exit_delta = cfg->infer.early_exit_delta;
    c->vlist = ai(cap);
    c->topk_idx = ai(K); c->topk_val = ac(K);
    if ((!c->int8 && !c->fp16 && !c->q4 && !c->q2 && (!c->k || !c->v)) ||
        (c->int8 && (!c->k8 || !c->v8 || !c->ks || !c->vs)) ||
        (c->fp16 && (!c->k16 || !c->v16)) ||
        (c->lr && (!c->klr || !c->vlr || !c->qlr)) ||
        (c->q4 && (!c->k4 || !c->v4 || !c->k4s || !c->v4s)) ||
        (c->q2 && (!c->k2 || !c->v2 || !c->k2s || !c->v2s ||
                   (c->recent > 0 && (!c->krec || !c->vrec || !c->recpos)) ||
                   (c->sinks > 0 && (!c->ksink || !c->vsink)))) ||
        ((c->q4 || c->q2) && (!c->q8v || !c->q8s)) ||
        !c->x || !c->xa || !c->q || !c->attn || !c->proj ||
        !c->x2 || !c->xf || !c->eout || !c->ys || !c->krow || !c->vrow || !c->gbuf ||
        !c->ubuf || !c->router_logits || !c->xfinal || !c->logits ||
        !c->topk_idx || !c->topk_val || !c->scores || !c->vlist || !c->xprev) {
        uenr_kv_cache_free(c);
        return UENR_ERR_ALLOC;
    }
    return UENR_OK;
}

void uenr_kv_cache_free(UenrKVCache *c) {
    if (!c) return;
    if (c->kmap) munmap(c->kmap, c->mapsz);
    if (c->vmap) { /* second mapping is interior */ }
    if (!c->kmap) { free(c->k); free(c->v); }
    free(c->k8); free(c->v8); free(c->k16); free(c->v16);
    free(c->k4); free(c->v4); free(c->k4s); free(c->v4s);
    free(c->q8v); free(c->q8s);
    free(c->k2); free(c->v2); free(c->k2s); free(c->v2s);
    free(c->krec); free(c->vrec); free(c->recpos);
    free(c->ksink); free(c->vsink);
    free(c->klr); free(c->vlr); free(c->qlr);
    free(c->ks); free(c->vs); free(c->x); free(c->xa); free(c->q); free(c->attn);
    free(c->proj); free(c->x2); free(c->xf); free(c->eout); free(c->ys); free(c->krow); free(c->vrow);
    free(c->gbuf); free(c->ubuf); free(c->router_logits); free(c->xfinal); free(c->logits);
    free(c->scores); free(c->xprev); free(c->vlist);
    free(c->topk_idx); free(c->topk_val);
    free(c->type); free(c->donor); free(c->shared);
    free(c->slot_off); free(c->slot_cap);
    memset(c, 0, sizeof(*c));
}

void uenr_kv_cache_reset(UenrKVCache *c) { if (c) c->len = 0; }

int uenr_kv_cache_copy_prefix(UenrKVCache *dst, const UenrKVCache *src, int n) {
    if (!dst || !src || n < 0) return 0;
    if (dst->int8 || dst->fp16 || dst->q4 || dst->q2 || dst->lr || dst->kmap) return 0;
    if (!dst->k || !dst->v || !src->k || !src->v) return 0;
    for (int l = 0; l < dst->n_slots; l++) {
        if (n > dst->slot_cap[l]) return 0;
        size_t sb = (size_t)dst->slot_off[l] * dst->KV;
        memcpy(dst->k + sb, src->k + sb, (size_t)n * dst->KV * sizeof(float));
        memcpy(dst->v + sb, src->v + sb, (size_t)n * dst->KV * sizeof(float));
    }
    dst->len = n;
    return 1;
}

static inline float dotv(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
    }
    float s = uenr_v_hsum(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static void norm_rope_head(float *v, const float *w, const UenrRope *r, int pos,
                           int hd, float eps) {
    float ss = 0.0f;
    for (int d = 0; d < hd; d++) ss += v[d] * v[d];
    float rr = uenr_rsqrtf(ss / (float)hd + eps);
    for (int d = 0; d < hd; d++) v[d] *= rr * w[d];
    int half = hd / 2;
    const float *ct = r->cos_tab + (size_t)pos * r->half;
    const float *st = r->sin_tab + (size_t)pos * r->half;
    if (half == r->half) {
        for (int j = 0; j < half; j++) {
            float a = v[j], b = v[j + half];
            float cc = ct[j], ss2 = st[j];
            v[j] = a * cc - b * ss2;
            v[j + half] = a * ss2 + b * cc;
        }
    }
}

static void norm_v_head(float *v, int hd, float eps) {
    float ss = 0.0f;
    for (int d = 0; d < hd; d++) ss += v[d] * v[d];
    float r = uenr_rsqrtf(ss / (float)hd + eps);
    for (int d = 0; d < hd; d++) v[d] *= r;
}

static inline void f32_to_f16(uint16_t *dst, const float *src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(dst + i), h);
    }
    for (; i < n; i++) {
        __m128 f = _mm_load_ss(src + i);
        __m128i h = _mm_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = (uint16_t)_mm_extract_epi16(h, 0);
    }
}
static inline void f16_to_f32(float *dst, const uint16_t *src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
    for (; i < n; i++) {
        __m128i h = _mm_cvtsi32_si128((int)src[i]);
        __m128 f = _mm_cvtph_ps(h);
        dst[i] = _mm_cvtss_f32(f);
    }
}

static inline void f32_to_q4(uint8_t *dst, float *scale, const float *src, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = src[i] < 0.0f ? -src[i] : src[i];
        if (a > mx) mx = a;
    }
    float sc = mx > 0.0f ? mx / 7.0f : 1.0f;
    *scale = sc;
    for (int i = 0; i < n; i += 2) {
        int q0 = (int)(src[i] / sc + (src[i] >= 0.0f ? 0.5f : -0.5f));
        int q1 = (int)(src[i + 1] / sc + (src[i + 1] >= 0.0f ? 0.5f : -0.5f));
        if (q0 > 7) q0 = 7;
        if (q0 < -7) q0 = -7;
        if (q1 > 7) q1 = 7;
        if (q1 < -7) q1 = -7;
        dst[i / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
}
static inline void q4_to_f32(float *dst, const uint8_t *src, int n, float scale) {
    for (int i = 0; i < n; i += 2) {
        uint8_t b = src[i / 2];
        int q0 = (int)(b & 0xF);
        int q1 = (int)(b >> 4);
        if (q0 >= 8) q0 -= 16;
        if (q1 >= 8) q1 -= 16;
        dst[i] = (float)q0 * scale;
        dst[i + 1] = (float)q1 * scale;
    }
}


/* Vectorized int4 dot/accumulate. Packing: byte i = (src[2i] low, src[2i+1] high),
 * signed 4-bit. Handles n in 32-value chunks, scalar tail. */
static inline __m128i q4_nibbles_to_i8(__m128i b) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(b, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
    __m128i inter = _mm_unpacklo_epi8(lo, hi); /* first 8 bytes -> 16 values */
    /* sign-extend 4-bit to 8-bit: (x ^ 8) - 8 */
    inter = _mm_sub_epi8(_mm_xor_si128(inter, _mm_set1_epi8(8)), _mm_set1_epi8(8));
    return inter;
}
static inline __m256 q4_i8x8_to_f32(__m128i v) {
    __m128i lo = _mm_cvtepi8_epi32(v);
    __m128i hi = _mm_cvtepi8_epi32(_mm_srli_si128(v, 4));
    __m256 f = _mm256_castps128_ps256(_mm_cvtepi32_ps(lo));
    return _mm256_insertf128_ps(f, _mm_cvtepi32_ps(hi), 1);
}
static inline float dot_q4_f32(const float *q, const uint8_t *p, int n, float scale) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(p + i / 2));
        __m128i v16 = q4_nibbles_to_i8(b);
        __m128i hi8 = _mm_srli_si128(v16, 8);
        __m256 f0 = q4_i8x8_to_f32(v16);
        __m256 f1 = q4_i8x8_to_f32(hi8);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(f0, _mm256_loadu_ps(q + i)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(f1, _mm256_loadu_ps(q + i + 8)));
        /* for 32 values we need two more 128-bit halves from the upper 8 source bytes */
        __m128i b2 = _mm_loadl_epi64((const __m128i *)(p + i / 2 + 8));
        __m128i v16b = q4_nibbles_to_i8(b2);
        __m128i hi8b = _mm_srli_si128(v16b, 8);
        __m256 g0 = q4_i8x8_to_f32(v16b);
        __m256 g1 = q4_i8x8_to_f32(hi8b);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(g0, _mm256_loadu_ps(q + i + 16)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(g1, _mm256_loadu_ps(q + i + 24)));
    }
    float sum = uenr_v_hsum(acc) * scale;
    for (; i < n; i++) {
        uint8_t b = p[i / 2];
        int qv = (i & 1) ? (b >> 4) : (b & 0xF);
        if (qv >= 8) qv -= 16;
        sum += q[i] * (float)qv * scale;
    }
    return sum;
}
static inline void add_q4_f32(float *o, const uint8_t *p, int n, float scale, float w) {
    __m256 wv = _mm256_set1_ps(w * scale);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        for (int h = 0; h < 2; h++) {
            __m128i b = _mm_loadl_epi64((const __m128i *)(p + i / 2 + h * 8));
            __m128i v16 = q4_nibbles_to_i8(b);
            __m128i hi8 = _mm_srli_si128(v16, 8);
            __m256 f0 = _mm256_mul_ps(q4_i8x8_to_f32(v16), wv);
            __m256 f1 = _mm256_mul_ps(q4_i8x8_to_f32(hi8), wv);
            _mm256_storeu_ps(o + i + h * 16, _mm256_add_ps(_mm256_loadu_ps(o + i + h * 16), f0));
            _mm256_storeu_ps(o + i + h * 16 + 8, _mm256_add_ps(_mm256_loadu_ps(o + i + h * 16 + 8), f1));
        }
    }
    for (; i < n; i++) {
        uint8_t b = p[i / 2];
        int qv = (i & 1) ? (b >> 4) : (b & 0xF);
        if (qv >= 8) qv -= 16;
        o[i] += w * (float)qv * scale;
    }
}


static inline int q2_code(float x, float s) {
    return x > 0.33f * s ? 1 : (x < -0.33f * s ? 2 : 0);
}
static inline int q2_val(int c) { return c == 1 ? 1 : (c == 2 ? -1 : 0); }
static inline void f32_to_q2(uint8_t *dst, float *scale, const float *src, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = src[i] < 0.0f ? -src[i] : src[i];
        if (a > mx) mx = a;
    }
    *scale = mx > 0.0f ? mx : 1.0f;
    for (int i = 0; i < n; i += 4) {
        int c0 = q2_code(src[i], *scale);
        int c1 = q2_code(src[i + 1], *scale);
        int c2 = q2_code(src[i + 2], *scale);
        int c3 = q2_code(src[i + 3], *scale);
        dst[i / 4] = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }
}
static inline int q2_get(const uint8_t *p, int i) {
    return q2_val((p[i / 4] >> (2 * (i & 3))) & 3);
}
static inline const float *kv_recent_ptr(const UenrKVCache *c, int is_k,
                                         int slot, int j, int pos) {
    if (c->sinks > 0 && j < c->sinks)
        return (is_k ? c->ksink : c->vsink) + ((size_t)slot * c->sinks + j) * c->KV;
    if (c->recent > 0 && pos - j < c->recent &&
        c->recpos[(size_t)slot * c->recent + (j % c->recent)] == j) {
        size_t idx = (size_t)slot * c->recent + (j % c->recent);
        return (is_k ? c->krec : c->vrec) + idx * c->KV;
    }
    return NULL;
}
static inline int hsum_epi32(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(v);
}
static inline int dot_q4_i8(const int8_t *q, const uint8_t *p, int n) {
    __m128i acc = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi16(1);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(p + i / 2));
        __m128i k8 = q4_nibbles_to_i8(b);
        __m128i qv = _mm_loadu_si128((const __m128i *)(q + i));
        __m128i ks = _mm_sign_epi8(k8, qv);
        __m128i aq = _mm_abs_epi8(qv);
        __m128i p16 = _mm_maddubs_epi16(aq, ks);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(p16, ones));
    }
    int sum = hsum_epi32(acc);
    for (; i < n; i++) {
        uint8_t b = p[i / 2];
        int qv = (i & 1) ? (b >> 4) : (b & 0xF);
        if (qv >= 8) qv -= 16;
        sum += q[i] * qv;
    }
    return sum;
}
static void hadamard(float *v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float a = v[j], b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
    float s = uenr_rsqrtf((float)n);
    for (int i = 0; i < n; i++) v[i] *= s;
}

UenrResult uenr_model_forward_step(UenrModel *m, UenrGemm *g, UenrKVCache *c,
                                   int token, int pos, float *logits_out) {
    if (!m || !g || !c || !logits_out || token < 0 || token >= m->V)
        return UENR_ERR_INVALID;
    if (pos < 0 || pos >= c->cap) return UENR_ERR_BOUNDS;
    const int C = m->C, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, K = m->K;
    const float eps = m->cfg.model.rms_eps;
    const float scale = m->cfg.model.attn_scale > 0.0f
                            ? m->cfg.model.attn_scale : uenr_rsqrtf((float)hd);
    const int group = H / G;
    float *x = c->x, *xa = c->xa, *q = c->q, *krow = c->krow, *vrow = c->vrow;
    float *attn = c->attn, *proj = c->proj, *x2 = c->x2;
    float *xf = c->xf, *eout = c->eout, *ys = c->ys;

    memcpy(x, m->tok_emb + (size_t)token * C, (size_t)C * sizeof(float));
    const int q8 = m->q8_on;

    float last_delta = 1.0f;
    c->exit_tokens++;
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        UenrQLayer *ql = q8 ? &m->q_layers[l] : NULL;
        const int donor = c->donor[l];
        if (c->early && l == m->L - 1 && y->is_shared &&
            last_delta < c->exit_delta) {
            c->exit_skips++;
            continue;
        }
        memcpy(c->xprev, x, (size_t)C * sizeof(float));
        const int shared = c->shared[l];
        const int local = (c->type[l] == 0);
        const int slot = donor;
        const int scap = c->slot_cap[slot];
        const size_t sbase = (size_t)c->slot_off[slot] * KV;
        int widx;
        if (local) widx = pos % scap;
        else if (c->evict)
            widx = (pos < c->sinks + c->ew) ? pos : c->sinks + ((pos - c->sinks) % c->ew);
        else widx = pos;
        const int astart = (local && pos >= scap) ? pos - scap + 1 : 0;
        int nv = 0;
        int *vl = c->vlist;
        if (!local && c->evict) {
            for (int j = 0; j < c->sinks && j <= pos; j++) vl[nv++] = j;
            int st = pos - c->ew + 1;
            if (st < c->sinks) st = c->sinks;
            for (int j = st; j <= pos; j++) vl[nv++] = j;
        } else if (c->sparse_k > 0 && c->ew > 0 && pos + 1 > c->sparse_k + c->ew) {
            for (int j = 0; j < c->sinks && j <= pos; j++) vl[nv++] = j;
            int recent = pos - c->ew + 1;
            if (recent < c->sinks) recent = c->sinks;
            for (int j = recent; j <= pos; j++) vl[nv++] = j;
            int lo = c->sinks, hi = recent - 1;
            uint64_t st = (uint64_t)pos * 6364136223846793005ULL + 1442695040888963407ULL;
            for (int i = 0; i < c->sparse_k && i < hi - lo + 1; i++) {
                st = st * 6364136223846793005ULL + 1;
                vl[nv++] = lo + (int)((st >> 33) % (uint64_t)(hi - lo + 1));
            }
        } else {
            for (int j = astart; j <= pos; j++) vl[nv++] = j;
        }
        uenr_rmsnorm_fwd(xa, &c->rstd, x, y->rms_attn, 1, C, eps);
        if (q8) {
            uenr_gemv_q8(q, xa, ql->wq.q, ql->wq.scale, C, C);
            if (!shared) {
                uenr_gemv_q8(krow, xa, ql->wk.q, ql->wk.scale, KV, C);
                uenr_gemv_q8(vrow, xa, ql->wv.q, ql->wv.scale, KV, C);
            }
        } else {
            uenr_linear_fwd(g, q, xa, y->wq, 1, C, C);
            if (!shared) {
                uenr_linear_fwd(g, krow, xa, y->wk, 1, C, KV);
                uenr_linear_fwd(g, vrow, xa, y->wv, 1, C, KV);
            }
        }

        const UenrRope *rp = y->type ? &m->rope_global : &m->rope_local;
        for (int h = 0; h < H; h++)
            norm_rope_head(q + (size_t)h * hd, y->q_norm, rp, pos, hd, eps);
        if (c->rotate)
            for (int h = 0; h < H; h++) hadamard(q + (size_t)h * hd, hd);
        if (c->q4 || c->q2) {
            for (int h = 0; h < H; h++) {
                float *qh = q + (size_t)h * hd;
                float mx = 0.0f;
                for (int d = 0; d < hd; d++) { float a = qh[d] < 0 ? -qh[d] : qh[d]; if (a > mx) mx = a; }
                float qs = mx > 0.0f ? mx / 127.0f : 1.0f;
                c->q8s[h] = qs;
                for (int d = 0; d < hd; d++) {
                    int iv = (int)(qh[d] / qs + (qh[d] >= 0 ? 0.5f : -0.5f));
                    if (iv > 127) iv = 127;
                    if (iv < -127) iv = -127;
                    c->q8v[(size_t)h * hd + d] = (int8_t)iv;
                }
            }
        }
        if (c->lr) {
            for (int h = 0; h < H; h++) {
                hadamard(q + (size_t)h * hd, hd);
                memcpy(c->qlr + (size_t)h * c->lr, q + (size_t)h * hd,
                       (size_t)c->lr * sizeof(float));
            }
        }
        if (!shared) {
            for (int gg = 0; gg < G; gg++)
                norm_rope_head(krow + (size_t)gg * hd, y->k_norm, rp, pos, hd, eps);
            if (m->cfg.model.v_norm) {
                for (int gg = 0; gg < G; gg++)
                    norm_v_head(vrow + (size_t)gg * hd, hd, eps);
            }
        }

        if (!shared && c->rotate) {
            for (int gg = 0; gg < G; gg++) {
                hadamard(krow + (size_t)gg * hd, hd);
                hadamard(vrow + (size_t)gg * hd, hd);
            }
        }
        if (!shared && c->lr) {
            for (int gg = 0; gg < G; gg++) hadamard(krow + (size_t)gg * hd, hd);
            for (int gg = 0; gg < G; gg++) hadamard(vrow + (size_t)gg * hd, hd);
        }
        if (!shared && c->int8) {
            int8_t *kc8 = c->k8 + sbase + (size_t)widx * KV;
            int8_t *vc8 = c->v8 + sbase + (size_t)widx * KV;
            for (int gg = 0; gg < G; gg++) {
                const float *kh = krow + (size_t)gg * hd;
                const float *vh = vrow + (size_t)gg * hd;
                float kmax = 0.0f, vmax = 0.0f;
                for (int d = 0; d < hd; d++) {
                    float a = kh[d] < 0 ? -kh[d] : kh[d];
                    float b = vh[d] < 0 ? -vh[d] : vh[d];
                    if (a > kmax) kmax = a;
                    if (b > vmax) vmax = b;
                }
                float ks = kmax > 0 ? kmax / 127.0f : 1.0f;
                float vs = vmax > 0 ? vmax / 127.0f : 1.0f;
                c->ks[(size_t)slot * G + gg] = ks;
                c->vs[(size_t)slot * G + gg] = vs;
                for (int d = 0; d < hd; d++) {
                    float qk = kh[d] / ks;
                    float qv = vh[d] / vs;
                    int ik = (int)(qk + (qk >= 0 ? 0.5f : -0.5f));
                    int iv = (int)(qv + (qv >= 0 ? 0.5f : -0.5f));
                    if (ik > 127) ik = 127;
                    if (ik < -127) ik = -127;
                    if (iv > 127) iv = 127;
                    if (iv < -127) iv = -127;
                    kc8[(size_t)gg * hd + d] = (int8_t)ik;
                    vc8[(size_t)gg * hd + d] = (int8_t)iv;
                }
            }
        } else if (!shared && c->lr) {
            for (int gg = 0; gg < G; gg++) {
                size_t dst = ((size_t)c->slot_off[slot] + widx) * G * c->lr + (size_t)gg * c->lr;
                memcpy(c->klr + dst, krow + (size_t)gg * hd, (size_t)c->lr * sizeof(float));
                memcpy(c->vlr + dst, vrow + (size_t)gg * hd, (size_t)c->lr * sizeof(float));
            }
        } else if (!shared && c->q2) {
            for (int gg = 0; gg < G; gg++) {
                size_t row = (size_t)widx * KV + (size_t)gg * hd;
                size_t sidx = ((size_t)c->slot_off[slot] + widx) * G + gg;
                f32_to_q2(c->k2 + (sbase / 4) + row / 4, &c->k2s[sidx], krow + (size_t)gg * hd, hd);
                f32_to_q2(c->v2 + (sbase / 4) + row / 4, &c->v2s[sidx], vrow + (size_t)gg * hd, hd);
            }
            if (c->sinks > 0 && pos < c->sinks) {
                memcpy(c->ksink + ((size_t)slot * c->sinks + pos) * KV, krow, (size_t)KV * sizeof(float));
                memcpy(c->vsink + ((size_t)slot * c->sinks + pos) * KV, vrow, (size_t)KV * sizeof(float));
            }
            if (c->recent > 0) {
                int ri = pos % c->recent;
                size_t idx = (size_t)slot * c->recent + ri;
                memcpy(c->krec + idx * KV, krow, (size_t)KV * sizeof(float));
                memcpy(c->vrec + idx * KV, vrow, (size_t)KV * sizeof(float));
                c->recpos[idx] = pos;
            }
        } else if (!shared && c->q4) {
            for (int gg = 0; gg < G; gg++) {
                size_t row = (size_t)widx * KV + (size_t)gg * hd;
                size_t sidx = ((size_t)c->slot_off[slot] + widx) * G + gg;
                f32_to_q4(c->k4 + (sbase / 2) + row / 2, &c->k4s[sidx], krow + (size_t)gg * hd, hd);
                f32_to_q4(c->v4 + (sbase / 2) + row / 2, &c->v4s[sidx], vrow + (size_t)gg * hd, hd);
            }
        } else if (!shared && c->fp16) {
            f32_to_f16(c->k16 + sbase + (size_t)widx * KV, krow, KV);
            f32_to_f16(c->v16 + sbase + (size_t)widx * KV, vrow, KV);
        } else if (!shared) {
            float *kc = c->k + sbase + (size_t)widx * KV;
            float *vc = c->v + sbase + (size_t)widx * KV;
            memcpy(kc, krow, (size_t)KV * sizeof(float));
            memcpy(vc, vrow, (size_t)KV * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            int gg = h / group;
            const float *qh = q + (size_t)h * hd;
            float *oh = attn + (size_t)h * hd;
            if (!c->int8 && !c->q2 && !c->q4 && !c->lr && !c->fp16 &&
                !(shared && y->ka)) {
                /* Fused online softmax + V accumulation: one pass, no scores. */
                float mxo = -3.402823466e+38f, lsum = 0.0f;
                for (int d = 0; d < hd; d++) oh[d] = 0.0f;
                for (int vi = 0; vi < nv; vi++) {
                    int j = vl[vi];
                    int jj = local ? (j % scap)
                                   : (c->evict ? (j < c->sinks ? j
                                      : c->sinks + ((j - c->sinks) % c->ew)) : j);
                    const float *kj = c->k + sbase + (size_t)jj * KV + (size_t)gg * hd;
                    float scv = dotv(qh, kj, hd) * scale;
                    if (m->cfg.model.attn_softcap > 0.0f &&
                        (scv > 0.1f * m->cfg.model.attn_softcap ||
                         scv < -0.1f * m->cfg.model.attn_softcap))
                        scv = m->cfg.model.attn_softcap *
                              uenr_tanhf(scv / m->cfg.model.attn_softcap);
                    const float *vj = c->v + sbase + (size_t)jj * KV + (size_t)gg * hd;
                    if (scv > mxo) {
                        float alpha = uenr_expf(mxo - scv);
                        lsum = lsum * alpha + 1.0f;
                        for (int d = 0; d < hd; d++) oh[d] = oh[d] * alpha + vj[d];
                        mxo = scv;
                    } else {
                        float p = uenr_expf(scv - mxo);
                        lsum += p;
                        for (int d = 0; d < hd; d++) oh[d] += p * vj[d];
                    }
                }
                float invl = uenr_rcpf(lsum);
                for (int d = 0; d < hd; d++) oh[d] *= invl;
                continue;
            }
            float mx = -3.402823466e+38f;
            float *sc = c->scores;
            const size_t rowbase = sbase;
            const float ksc = c->int8 ? c->ks[(size_t)donor * G + gg] : 1.0f;
            const float vsc = c->int8 ? c->vs[(size_t)donor * G + gg] : 1.0f;
            for (int vi = 0; vi < nv; vi++) {
                int j = vl[vi];
                float acc = 0.0f;
                int jj = local ? (j % scap) : (c->evict ? (j < c->sinks ? j : c->sinks + ((j - c->sinks) % c->ew)) : j);
                if (shared && y->ka) {
                    const float *kj = c->k + rowbase + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++)
                        acc += qh[d] * (y->ka[(size_t)gg * hd + d] * kj[d] +
                                        y->kb[(size_t)gg * hd + d]);
                } else if (c->lr) {
                    const float *kl = c->klr + ((rowbase / KV + (size_t)jj) * G + gg) * c->lr;
                    const float *ql = c->qlr + (size_t)h * c->lr;
                    for (int d = 0; d < c->lr; d++) acc += ql[d] * kl[d];
                } else if (c->int8) {
                    const int8_t *kj = c->k8 + rowbase + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++) acc += qh[d] * ((float)kj[d] * ksc);
                } else if (c->q2) {
                    const float *rk = kv_recent_ptr(c, 1, donor, jj, pos);
                    if (rk) {
                        acc = dotv(qh, rk + (size_t)gg * hd, hd);
                    } else {
                        size_t rowoff = (size_t)jj * KV + (size_t)gg * hd;
                        int isum = 0;
                        for (int d = 0; d < hd; d++)
                            isum += c->q8v[(size_t)h * hd + d] * q2_get(c->k2 + rowbase / 4 + rowoff / 4, d);
                        acc = (float)isum * c->q8s[h] *
                              c->k2s[(rowbase / KV + (size_t)jj) * G + gg];
                    }
                } else if (c->q4) {
                    size_t rowoff = (size_t)jj * KV + (size_t)gg * hd;
                    int isum = dot_q4_i8(c->q8v + (size_t)h * hd,
                                         c->k4 + rowbase / 2 + rowoff / 2, hd);
                    acc = (float)isum * c->q8s[h] *
                          c->k4s[(rowbase / KV + (size_t)jj) * G + gg];
                } else if (c->fp16) {
                    float kf[1024];
                    f16_to_f32(kf, c->k16 + rowbase + (size_t)jj * KV + (size_t)gg * hd, hd);
                    acc = dotv(qh, kf, hd);
                } else {
                    const float *kj = c->k + rowbase + (size_t)jj * KV + (size_t)gg * hd;
                    acc = dotv(qh, kj, hd);
                }
                sc[j] = acc * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0.0f;
            for (int vi = 0; vi < nv; vi++) { int j = vl[vi]; sc[j] = uenr_expf(sc[j] - mx); sum += sc[j]; }
            float inv = uenr_rcpf(sum);
            if (shared && y->ka) {
                for (int d = 0; d < hd; d++) oh[d] = 0.0f;
                for (int vi = 0; vi < nv; vi++) {
                    int j = vl[vi];
                    float p = sc[j] * inv;
                    int jj = local ? (j % scap) : (c->evict ? (j < c->sinks ? j : c->sinks + ((j - c->sinks) % c->ew)) : j);
                    const float *vj = c->v + rowbase + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++)
                        oh[d] += p * (y->va[(size_t)gg * hd + d] * vj[d] +
                                      y->vb[(size_t)gg * hd + d]);
                }
            } else if (c->lr) {
                float la[1024];
                for (int d = 0; d < c->lr; d++) la[d] = 0.0f;
                for (int vi = 0; vi < nv; vi++) {
                    int j = vl[vi];
                    float p = sc[j] * inv;
                    int jj = local ? (j % scap) : (c->evict ? (j < c->sinks ? j : c->sinks + ((j - c->sinks) % c->ew)) : j);
                    const float *vlat = c->vlr + ((rowbase / KV + (size_t)jj) * G + gg) * c->lr;
                    for (int d = 0; d < c->lr; d++) la[d] += p * vlat[d];
                }
                float up[1024];
                for (int d = 0; d < hd; d++) up[d] = 0.0f;
                for (int d = 0; d < c->lr; d++) up[d] = la[d];
                hadamard(up, hd);
                for (int d = 0; d < hd; d++) oh[d] = up[d];
            } else if (c->q2) {
                int32_t ac2[64];
                for (int d = 0; d < hd; d++) { oh[d] = 0.0f; ac2[d] = 0; }
                for (int vi = 0; vi < nv; vi++) {
                    int j = vl[vi];
                    float p = sc[j] * inv;
                    int jj = local ? (j % scap) : (c->evict ? (j < c->sinks ? j : c->sinks + ((j - c->sinks) % c->ew)) : j);
                    const float *rv = kv_recent_ptr(c, 0, donor, jj, pos);
                    if (rv) {
                        for (int d = 0; d < hd; d++) oh[d] += p * rv[(size_t)gg * hd + d];
                    } else {
                        size_t rowoff = (size_t)jj * KV + (size_t)gg * hd;
                        float vs2 = c->v2s[(rowbase / KV + (size_t)jj) * G + gg];
                        for (int d = 0; d < hd; d++)
                            ac2[d] += (int32_t)(p * (float)q2_get(c->v2 + rowbase / 4 + rowoff / 4, d) * vs2 * 256.0f);
                    }
                }
                for (int d = 0; d < hd; d++) oh[d] += (float)ac2[d] * (1.0f / 256.0f);
            } else if (c->q4 && hd % 16 == 0) {
                __m128i acc[16];
                int na = hd / 4;
                for (int a = 0; a < na; a++) acc[a] = _mm_setzero_si128();
                for (int vi = 0; vi < nv; vi++) {
                    int j = vl[vi];
                    float p = sc[j] * inv;
                    int jj = local ? (j % scap) : (c->evict ? (j < c->sinks ? j : c->sinks + ((j - c->sinks) % c->ew)) : j);
                    float vsc4 = c->v4s[(rowbase / KV + (size_t)jj) * G + gg];
                    int pv = (int)(p * vsc4 * 256.0f + 0.5f);
                    if (pv > 32767) pv = 32767;
                    __m128i pp = _mm_set1_epi16((short)pv);
                    size_t rowoff = (size_t)jj * KV + (size_t)gg * hd;
                    for (int ch = 0; ch < hd / 16; ch++) {
                        __m128i b = _mm_loadl_epi64((const __m128i *)(c->v4 + rowbase / 2 + rowoff / 2 + ch * 8));
                        __m128i v8 = q4_nibbles_to_i8(b);
                        __m128i lo16 = _mm_cvtepi8_epi16(v8);
                        __m128i hi16 = _mm_cvtepi8_epi16(_mm_srli_si128(v8, 8));
                        __m128i plo = _mm_mullo_epi16(lo16, pp);
                        __m128i phi = _mm_mullo_epi16(hi16, pp);
                        int a0 = ch * 4;
                        acc[a0 + 0] = _mm_add_epi32(acc[a0 + 0], _mm_cvtepi16_epi32(plo));
                        acc[a0 + 1] = _mm_add_epi32(acc[a0 + 1], _mm_cvtepi16_epi32(_mm_srli_si128(plo, 8)));
                        acc[a0 + 2] = _mm_add_epi32(acc[a0 + 2], _mm_cvtepi16_epi32(phi));
                        acc[a0 + 3] = _mm_add_epi32(acc[a0 + 3], _mm_cvtepi16_epi32(_mm_srli_si128(phi, 8)));
                    }
                }
                int tmp[64];
                for (int a = 0; a < na; a++) _mm_storeu_si128((__m128i *)(tmp + a * 4), acc[a]);
                for (int d = 0; d < hd; d++) oh[d] = (float)tmp[d] * (1.0f / 256.0f);
            } else {
                for (int d = 0; d < hd; d++) oh[d] = 0.0f;
                for (int vi = 0; vi < nv; vi++) {
                    int j = vl[vi];
                    float p = sc[j] * inv;
                    int jj = local ? (j % scap) : (c->evict ? (j < c->sinks ? j : c->sinks + ((j - c->sinks) % c->ew)) : j);
                    if (c->int8) {
                        const int8_t *vj = c->v8 + rowbase + (size_t)jj * KV + (size_t)gg * hd;
                        for (int d = 0; d < hd; d++) oh[d] += p * ((float)vj[d] * vsc);
                    } else if (c->q4) {
                        size_t rowoff = (size_t)jj * KV + (size_t)gg * hd;
                        add_q4_f32(oh, c->v4 + rowbase / 2 + rowoff / 2, hd,
                                   c->v4s[(rowbase / KV + (size_t)jj) * G + gg], p);
                    } else if (c->fp16) {
                        float vf[1024];
                        f16_to_f32(vf, c->v16 + rowbase + (size_t)jj * KV + (size_t)gg * hd, hd);
                        for (int d = 0; d < hd; d++) oh[d] += p * vf[d];
                    } else {
                        const float *vj = c->v + rowbase + (size_t)jj * KV + (size_t)gg * hd;
                        for (int d = 0; d < hd; d++) oh[d] += p * vj[d];
                    }
                }
            }
            if (c->rotate) hadamard(oh, hd);
        }

        if (q8) uenr_gemv_q8(proj, attn, ql->wo.q, ql->wo.scale, C, C);
        else uenr_linear_fwd(g, proj, attn, y->wo, 1, C, C);
        uenr_rmsnorm_fwd(proj, &c->rstd, proj, y->post_attn, 1, C, eps);
        for (int d = 0; d < C; d++) x2[d] = x[d] + y->ls[d] * proj[d];

        uenr_rmsnorm_fwd(xf, &c->rstd, x2, y->rms_ffn, 1, C, eps);
        if (q8) uenr_gemv_q8(c->router_logits, xf, ql->router.q, ql->router.scale, E, C);
        else uenr_linear_fwd(g, c->router_logits, xf, y->router, 1, C, E);
        float mx = c->router_logits[0];
        for (int i = 1; i < E; i++) if (c->router_logits[i] > mx) mx = c->router_logits[i];
        float sum = 0.0f;
        for (int i = 0; i < E; i++) { c->router_logits[i] = uenr_expf(c->router_logits[i] - mx); sum += c->router_logits[i]; }
        float inv = uenr_rcpf(sum);
        for (int i = 0; i < E; i++) c->router_logits[i] *= inv;
        for (int s = 0; s < K; s++) {
            int best = -1;
            float bv = -1.0f;
            for (int i = 0; i < E; i++) {
                int taken = 0;
                for (int j = 0; j < s; j++) if (c->topk_idx[j] == i) { taken = 1; break; }
                if (!taken && c->router_logits[i] > bv) { bv = c->router_logits[i]; best = i; }
            }
            c->topk_idx[s] = best;
            c->topk_val[s] = bv;
        }

        for (int d = 0; d < C; d++) eout[d] = 0.0f;
        if (m->cfg.moe.enabled) {
            for (int s = 0; s < K; s++) {
                int e = c->topk_idx[s];
                if (q8) {
                    UenrQMat qg = { ql->expert_wg.q + (size_t)e * EF * C,
                                    ql->expert_wg.scale + (size_t)e * EF, EF, C };
                    UenrQMat qu = { ql->expert_wu.q + (size_t)e * EF * C,
                                    ql->expert_wu.scale + (size_t)e * EF, EF, C };
                    UenrQMat qd = { ql->expert_wd.q + (size_t)e * C * EF,
                                    ql->expert_wd.scale + (size_t)e * C, C, EF };
                    uenr_swiglu_fwd_q8_1(ys, xf, &qg, &qu, &qd, c->gbuf, c->ubuf);
                } else {
                    const float *Wg = y->expert_wg + (size_t)e * EF * C;
                    const float *Wu = y->expert_wu + (size_t)e * EF * C;
                    const float *Wd = y->expert_wd + (size_t)e * C * EF;
                    uenr_swiglu_fwd(g, 1, C, EF, xf, Wg, Wu, Wd, ys, c->gbuf, c->ubuf);
                }
                float gate = c->topk_val[s];
                for (int d = 0; d < C; d++) eout[d] += gate * ys[d];
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                if (q8) uenr_swiglu_fwd_q8_1(ys, xf, &ql->shared_wg, &ql->shared_wu,
                                             &ql->shared_wd, c->gbuf, c->ubuf);
                else uenr_swiglu_fwd(g, 1, C, FF, xf, y->shared_wg, y->shared_wu,
                                     y->shared_wd, ys, c->gbuf, c->ubuf);
                for (int d = 0; d < C; d++) eout[d] += ys[d];
            }
        } else {
            if (q8) uenr_swiglu_fwd_q8_1(ys, xf, &ql->shared_wg, &ql->shared_wu,
                                         &ql->shared_wd, c->gbuf, c->ubuf);
            else uenr_swiglu_fwd(g, 1, C, FF, xf, y->shared_wg, y->shared_wu,
                                 y->shared_wd, ys, c->gbuf, c->ubuf);
            for (int d = 0; d < C; d++) eout[d] += ys[d];
        }
        uenr_rmsnorm_fwd(eout, &c->rstd, eout, y->post_ffn, 1, C, eps);
        for (int d = 0; d < C; d++) x[d] = x2[d] + eout[d];
        float dn = 0.0f, xn = 0.0f;
        for (int d = 0; d < C; d++) {
            float dx = x[d] - c->xprev[d];
            dn += dx * dx; xn += x[d] * x[d];
        }
        last_delta = uenr_sqrtf(dn / (xn + 1e-12f));
    }

    uenr_rmsnorm_fwd(c->xfinal, &c->rstd, x, m->rms_final, 1, C, eps);
    if (q8) uenr_gemv_q8(logits_out, c->xfinal, m->q_lm_head.q,
                         m->q_lm_head.scale, m->V, C);
    else uenr_linear_fwd(g, logits_out, c->xfinal, m->lm_head, 1, C, m->V);
    c->len = pos + 1;
    return UENR_OK;
}

UenrResult uenr_model_forward_span(UenrModel *m, UenrGemm *g, UenrKVCache *c,
                                   const int *tokens, int k, int pos,
                                   float *logits_out) {
    if (!m || !g || !c || !tokens || !logits_out || k <= 0 || pos < 0 ||
        pos + k > c->cap) return UENR_ERR_INVALID;
    if (m->q8_on || c->int8 || c->fp16 || c->q4 || c->q2 || c->lr) return UENR_ERR_INVALID; /* fallback: sequential */
    const int C = m->C, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, K = m->K, V = m->V;
    const float eps = m->cfg.model.rms_eps;
    const float scale = m->cfg.model.attn_scale > 0.0f
                            ? m->cfg.model.attn_scale : uenr_rsqrtf((float)hd);
    const int group = H / G;
    const size_t kC = (size_t)k * C, kKV = (size_t)k * KV;
    int maxF = EF > FF ? EF : FF;

    float *x = ac(kC), *xa = ac(kC), *q = ac(kC), *krow = ac(kKV), *vrow = ac(kKV);
    float *attn = ac(kC), *proj = ac(kC), *x2 = ac(kC), *xf = ac(kC);
    float *eout = ac(kC), *ys = ac(kC), *gxs = ac(kC), *gys = ac(kC);
    float *router = ac((size_t)k * E), *gbuf = ac((size_t)k * maxF);
    float *ubuf = ac((size_t)k * maxF), *scratch = ac((size_t)k * maxF);
    float *sc = (float *)malloc((size_t)c->cap * sizeof(float));
    int *ecnt = ai(E), *eoff = ai(E + 1), *elist = ai(k * K + 1), *cursor = ai(E + 1);
    int *eidx = ai(k * K + 1);
    float *egate = ac((size_t)k * K + 1);
    float *xfinal = ac(kC);
    if (!x || !xa || !q || !krow || !vrow || !attn || !proj || !x2 || !xf ||
        !eout || !ys || !gxs || !gys || !router || !gbuf || !ubuf || !scratch ||
        !sc || !ecnt || !eoff || !elist || !cursor || !eidx || !egate || !xfinal) {
        free(x); free(xa); free(q); free(krow); free(vrow); free(attn);
        free(proj); free(x2); free(xf); free(eout); free(ys); free(gxs); free(gys);
        free(router); free(gbuf); free(ubuf); free(scratch); free(sc);
        free(ecnt); free(eoff); free(elist); free(cursor); free(eidx); free(egate);
        free(xfinal);
        return UENR_ERR_ALLOC;
    }

    for (int i = 0; i < k; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= V) tok = 0;
        memcpy(x + (size_t)i * C, m->tok_emb + (size_t)tok * C, (size_t)C * sizeof(float));
    }

    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        const int donor = c->donor[l], shared = c->shared[l];
        const int local = (c->type[l] == 0);
        const int slot = donor;
        const int scap = c->slot_cap[slot];
        const size_t sbase = (size_t)c->slot_off[slot] * KV;
        if (k > scap) goto span_fail;

        uenr_rmsnorm_fwd(xa, (float *)ys, x, y->rms_attn, k, C, eps);
        uenr_linear_fwd(g, q, xa, y->wq, k, C, C);
        if (!shared) {
            uenr_linear_fwd(g, krow, xa, y->wk, k, C, KV);
            uenr_linear_fwd(g, vrow, xa, y->wv, k, C, KV);
        }
        const UenrRope *rope_l = y->type ? &m->rope_global : &m->rope_local;
        for (int i = 0; i < k; i++) {
            int p = pos + i;
            for (int h = 0; h < H; h++)
                norm_rope_head(q + (size_t)i * C + (size_t)h * hd, y->q_norm, rope_l, p, hd, eps);
            if (!shared) {
                for (int gg = 0; gg < G; gg++) {
                    float *kh = krow + (size_t)i * KV + (size_t)gg * hd;
                    norm_rope_head(kh, y->k_norm, rope_l, p, hd, eps);
                    if (m->cfg.model.v_norm) norm_v_head(vrow + (size_t)i * KV + (size_t)gg * hd, hd, eps);
                }
            }
        }
        if (!shared) {
            for (int i = 0; i < k; i++) {
                int widx = local ? ((pos + i) % scap) : (pos + i);
                memcpy(c->k + sbase + (size_t)widx * KV, krow + (size_t)i * KV, (size_t)KV * sizeof(float));
                memcpy(c->v + sbase + (size_t)widx * KV, vrow + (size_t)i * KV, (size_t)KV * sizeof(float));
            }
        }

        for (int i = 0; i < k; i++) {
            int p = pos + i;
            int astart = (local && p >= scap) ? p - scap + 1 : 0;
            for (int h = 0; h < H; h++) {
                int gg = h / group;
                const float *qh = q + (size_t)i * C + (size_t)h * hd;
                float *oh = attn + (size_t)i * C + (size_t)h * hd;
                float mx = -3.402823466e+38f;
                for (int j = astart; j <= p; j++) {
                    int jj = local ? (j % scap) : j;
                    float acc = dotv(qh, c->k + sbase + (size_t)jj * KV + (size_t)gg * hd, hd);
                    acc *= scale;
                    if (m->cfg.model.attn_softcap > 0.0f &&
                        (acc > 0.1f * m->cfg.model.attn_softcap || acc < -0.1f * m->cfg.model.attn_softcap))
                        acc = m->cfg.model.attn_softcap * uenr_tanhf(acc / m->cfg.model.attn_softcap);
                    sc[j] = acc;
                    if (acc > mx) mx = acc;
                }
                float sum = 0.0f;
                for (int j = astart; j <= p; j++) { sc[j] = uenr_expf(sc[j] - mx); sum += sc[j]; }
                float inv = uenr_rcpf(sum);
                for (int d = 0; d < hd; d++) oh[d] = 0.0f;
                for (int j = astart; j <= p; j++) {
                    int jj = local ? (j % scap) : j;
                    float pr = sc[j] * inv;
                    const float *vj = c->v + sbase + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++) oh[d] += pr * vj[d];
                }
            }
        }

        uenr_linear_fwd(g, proj, attn, y->wo, k, C, C);
        uenr_rmsnorm_fwd(proj, (float *)ys, proj, y->post_attn, k, C, eps);
        for (size_t i = 0; i < kC; i++) x2[i] = x[i] + y->ls[i % C] * proj[i];
        uenr_rmsnorm_fwd(xf, (float *)ys, x2, y->rms_ffn, k, C, eps);

        uenr_linear_fwd(g, router, xf, y->router, k, C, E);
        memset(eout, 0, kC * sizeof(float));
        memset(ecnt, 0, (size_t)E * sizeof(int));
        for (int t = 0; t < k; t++) {
            float *r = router + (size_t)t * E;
            float mx = r[0];
            for (int e = 1; e < E; e++) if (r[e] > mx) mx = r[e];
            float sum = 0.0f;
            for (int e = 0; e < E; e++) { r[e] = uenr_expf(r[e] - mx); sum += r[e]; }
            float inv = uenr_rcpf(sum);
            for (int e = 0; e < E; e++) r[e] *= inv;
            for (int s = 0; s < K; s++) {
                int best = -1; float bv = -1.0f;
                for (int e = 0; e < E; e++) {
                    int taken = 0;
                    for (int j = 0; j < s; j++) if (eidx[t * K + j] == e) { taken = 1; break; }
                    if (!taken && r[e] > bv) { bv = r[e]; best = e; }
                }
                eidx[t * K + s] = best;
                egate[t * K + s] = bv;
                ecnt[best]++;
            }
        }
        if (m->cfg.moe.enabled) {
            eoff[0] = 0;
            for (int e = 0; e < E; e++) eoff[e + 1] = eoff[e] + ecnt[e];
            memcpy(cursor, eoff, (size_t)E * sizeof(int));
            for (int t = 0; t < k; t++)
                for (int s = 0; s < K; s++) elist[cursor[eidx[t * K + s]]++] = t * K + s;
            for (int e = 0; e < E; e++) {
                int ne = ecnt[e];
                if (ne <= 0) continue;
                for (int a = 0; a < ne; a++) {
                    int qx = elist[eoff[e] + a];
                    int t = qx / K;
                    memcpy(gxs + (size_t)a * C, xf + (size_t)t * C, (size_t)C * sizeof(float));
                }
                const float *Wg = y->expert_wg + (size_t)e * EF * C;
                const float *Wu = y->expert_wu + (size_t)e * EF * C;
                const float *Wd = y->expert_wd + (size_t)e * C * EF;
                uenr_swiglu_fwd(g, ne, C, EF, gxs, Wg, Wu, Wd, gys, gbuf, ubuf);
                for (int a = 0; a < ne; a++) {
                    int qx = elist[eoff[e] + a];
                    int t = qx / K;
                    float gate = egate[qx];
                    for (int d = 0; d < C; d++)
                        eout[(size_t)t * C + d] += gate * gys[(size_t)a * C + d];
                }
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                uenr_swiglu_fwd(g, k, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                                ys, gbuf, ubuf);
                for (size_t i = 0; i < kC; i++) eout[i] += ys[i];
            }
        } else {
            uenr_swiglu_fwd(g, k, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                            ys, gbuf, ubuf);
            for (size_t i = 0; i < kC; i++) eout[i] += ys[i];
        }
        uenr_rmsnorm_fwd(eout, (float *)ys, eout, y->post_ffn, k, C, eps);
        for (size_t i = 0; i < kC; i++) x[i] = x2[i] + eout[i];
    }

    uenr_rmsnorm_fwd(xfinal, (float *)ys, x, m->rms_final, k, C, eps);
    uenr_linear_fwd(g, logits_out, xfinal, m->lm_head, k, C, V);
    if (m->cfg.model.final_softcap > 0.0f) {
        const float scv = m->cfg.model.final_softcap;
        for (size_t i = 0; i < (size_t)k * V; i++) {
            float z = logits_out[i];
            if (z > 0.1f * scv || z < -0.1f * scv)
                logits_out[i] = scv * uenr_tanhf(z / scv);
        }
    }
    c->len = pos + k;
    free(x); free(xa); free(q); free(krow); free(vrow); free(attn);
    free(proj); free(x2); free(xf); free(eout); free(ys); free(gxs); free(gys);
    free(router); free(gbuf); free(ubuf); free(scratch); free(sc);
    free(ecnt); free(eoff); free(elist); free(cursor); free(eidx); free(egate);
    free(xfinal);
    return UENR_OK;

span_fail:
    free(x); free(xa); free(q); free(krow); free(vrow); free(attn);
    free(proj); free(x2); free(xf); free(eout); free(ys); free(gxs); free(gys);
    free(router); free(gbuf); free(ubuf); free(scratch); free(sc);
    free(ecnt); free(eoff); free(elist); free(cursor); free(eidx); free(egate);
    free(xfinal);
    return UENR_ERR_BOUNDS;
}

UenrResult uenr_model_forward_multi(UenrModel *m, UenrGemm *g,
                                    UenrKVCache **caches, int B,
                                    const int *tokens, const int *poses,
                                    float *logits_out) {
    if (!m || !g || !caches || !tokens || !poses || !logits_out || B <= 0)
        return UENR_ERR_INVALID;
    if (m->q8_on) return UENR_ERR_INVALID;
    int maxcap = 0;
    for (int b = 0; b < B; b++) {
        if (!caches[b] || caches[b]->int8 || caches[b]->fp16 || caches[b]->q4 || caches[b]->q2 || caches[b]->lr || poses[b] < 0 || poses[b] >= caches[b]->cap)
            return UENR_ERR_INVALID;
        if (caches[b]->cap > maxcap) maxcap = caches[b]->cap;
    }
    const int C = m->C, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, K = m->K, V = m->V;
    const float eps = m->cfg.model.rms_eps;
    const float scale = m->cfg.model.attn_scale > 0.0f
                            ? m->cfg.model.attn_scale : uenr_rsqrtf((float)hd);
    const int group = H / G;
    const size_t BC = (size_t)B * C, BKV = (size_t)B * KV;
    int maxF = EF > FF ? EF : FF;

    float *x = ac(BC), *xa = ac(BC), *q = ac(BC), *krow = ac(BKV), *vrow = ac(BKV);
    float *attn = ac(BC), *proj = ac(BC), *x2 = ac(BC), *xf = ac(BC), *eout = ac(BC);
    float *ys = ac(BC), *gxs = ac(BC), *gys = ac(BC), *xfinal = ac(BC);
    float *router = ac((size_t)B * E), *gbuf = ac((size_t)B * maxF);
    float *ubuf = ac((size_t)B * maxF);
    float *sc = (float *)malloc((size_t)maxcap * sizeof(float));
    int *ecnt = ai(E), *eoff = ai(E + 1), *elist = ai(B * K + 1), *cursor = ai(E + 1);
    int *eidx = ai(B * K + 1);
    float *egate = ac((size_t)B * K + 1);
    if (!x || !xa || !q || !krow || !vrow || !attn || !proj || !x2 || !xf || !eout ||
        !ys || !gxs || !gys || !xfinal || !router || !gbuf || !ubuf || !sc ||
        !ecnt || !eoff || !elist || !cursor || !eidx || !egate)
        goto multi_fail;

    for (int b = 0; b < B; b++) {
        int tok = tokens[b];
        if (tok < 0 || tok >= V) tok = 0;
        memcpy(x + (size_t)b * C, m->tok_emb + (size_t)tok * C, (size_t)C * sizeof(float));
    }

    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        const int shared = uenr_layer_is_shared(&m->cfg, l);
        uenr_rmsnorm_fwd(xa, ys, x, y->rms_attn, B, C, eps);
        uenr_linear_fwd(g, q, xa, y->wq, B, C, C);
        if (!shared) {
            uenr_linear_fwd(g, krow, xa, y->wk, B, C, KV);
            uenr_linear_fwd(g, vrow, xa, y->wv, B, C, KV);
        }
        const UenrRope *rope_l = (y->type ? &m->rope_global : &m->rope_local);
        for (int b = 0; b < B; b++) {
            int p = poses[b];
            for (int h = 0; h < H; h++)
                norm_rope_head(q + (size_t)b * C + (size_t)h * hd, y->q_norm, rope_l, p, hd, eps);
            if (!shared) {
                for (int gg = 0; gg < G; gg++) {
                    norm_rope_head(krow + (size_t)b * KV + (size_t)gg * hd, y->k_norm, rope_l, p, hd, eps);
                    if (m->cfg.model.v_norm)
                        norm_v_head(vrow + (size_t)b * KV + (size_t)gg * hd, hd, eps);
                }
            }
        }
        if (!shared) {
            for (int b = 0; b < B; b++) {
                UenrKVCache *c = caches[b];
                int pos = poses[b];
                int donor = c->donor[l], local = (c->type[l] == 0);
                int scap = c->slot_cap[donor];
                size_t sbase = (size_t)c->slot_off[donor] * KV;
                if (pos >= scap && !local) { /* global: cap == max_seq */ }
                int widx = local ? (pos % scap) : pos;
                memcpy(c->k + sbase + (size_t)widx * KV, krow + (size_t)b * KV, (size_t)KV * sizeof(float));
                memcpy(c->v + sbase + (size_t)widx * KV, vrow + (size_t)b * KV, (size_t)KV * sizeof(float));
                c->len = pos + 1;
            }
        }

        for (int b = 0; b < B; b++) {
            UenrKVCache *c = caches[b];
            int pos = poses[b];
            int donor = c->donor[l], local = (c->type[l] == 0);
            int scap = c->slot_cap[donor];
            size_t sbase = (size_t)c->slot_off[donor] * KV;
            int astart = (local && pos >= scap) ? pos - scap + 1 : 0;
            for (int h = 0; h < H; h++) {
                int gg = h / group;
                const float *qh = q + (size_t)b * C + (size_t)h * hd;
                float *oh = attn + (size_t)b * C + (size_t)h * hd;
                float mx = -3.402823466e+38f;
                for (int j = astart; j <= pos; j++) {
                    int jj = local ? (j % scap) : j;
                    float acc = dotv(qh, c->k + sbase + (size_t)jj * KV + (size_t)gg * hd, hd) * scale;
                    if (m->cfg.model.attn_softcap > 0.0f &&
                        (acc > 0.1f * m->cfg.model.attn_softcap || acc < -0.1f * m->cfg.model.attn_softcap))
                        acc = m->cfg.model.attn_softcap * uenr_tanhf(acc / m->cfg.model.attn_softcap);
                    sc[j] = acc;
                    if (acc > mx) mx = acc;
                }
                float sum = 0.0f;
                for (int j = astart; j <= pos; j++) { sc[j] = uenr_expf(sc[j] - mx); sum += sc[j]; }
                float inv = uenr_rcpf(sum);
                for (int d = 0; d < hd; d++) oh[d] = 0.0f;
                for (int j = astart; j <= pos; j++) {
                    int jj = local ? (j % scap) : j;
                    float pr = sc[j] * inv;
                    const float *vj = c->v + sbase + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++) oh[d] += pr * vj[d];
                }
            }
        }

        uenr_linear_fwd(g, proj, attn, y->wo, B, C, C);
        uenr_rmsnorm_fwd(proj, ys, proj, y->post_attn, B, C, eps);
        for (size_t i = 0; i < BC; i++) x2[i] = x[i] + y->ls[i % C] * proj[i];
        uenr_rmsnorm_fwd(xf, ys, x2, y->rms_ffn, B, C, eps);

        uenr_linear_fwd(g, router, xf, y->router, B, C, E);
        memset(eout, 0, BC * sizeof(float));
        memset(ecnt, 0, (size_t)E * sizeof(int));
        for (int t = 0; t < B; t++) {
            float *r = router + (size_t)t * E;
            float mx = r[0];
            for (int e = 1; e < E; e++) if (r[e] > mx) mx = r[e];
            float sum = 0.0f;
            for (int e = 0; e < E; e++) { r[e] = uenr_expf(r[e] - mx); sum += r[e]; }
            float inv = uenr_rcpf(sum);
            for (int e = 0; e < E; e++) r[e] *= inv;
            for (int s = 0; s < K; s++) {
                int best = -1; float bv = -1.0f;
                for (int e = 0; e < E; e++) {
                    int taken = 0;
                    for (int j = 0; j < s; j++) if (eidx[t * K + j] == e) { taken = 1; break; }
                    if (!taken && r[e] > bv) { bv = r[e]; best = e; }
                }
                eidx[t * K + s] = best;
                egate[t * K + s] = bv;
                ecnt[best]++;
            }
        }
        if (m->cfg.moe.enabled) {
            eoff[0] = 0;
            for (int e = 0; e < E; e++) eoff[e + 1] = eoff[e] + ecnt[e];
            memcpy(cursor, eoff, (size_t)E * sizeof(int));
            for (int t = 0; t < B; t++)
                for (int s = 0; s < K; s++) elist[cursor[eidx[t * K + s]]++] = t * K + s;
            for (int e = 0; e < E; e++) {
                int ne = ecnt[e];
                if (ne <= 0) continue;
                for (int a = 0; a < ne; a++) {
                    int qx = elist[eoff[e] + a];
                    int t = qx / K;
                    memcpy(gxs + (size_t)a * C, xf + (size_t)t * C, (size_t)C * sizeof(float));
                }
                const float *Wg = y->expert_wg + (size_t)e * EF * C;
                const float *Wu = y->expert_wu + (size_t)e * EF * C;
                const float *Wd = y->expert_wd + (size_t)e * C * EF;
                uenr_swiglu_fwd(g, ne, C, EF, gxs, Wg, Wu, Wd, gys, gbuf, ubuf);
                for (int a = 0; a < ne; a++) {
                    int qx = elist[eoff[e] + a];
                    int t = qx / K;
                    float gate = egate[qx];
                    for (int d = 0; d < C; d++)
                        eout[(size_t)t * C + d] += gate * gys[(size_t)a * C + d];
                }
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                uenr_swiglu_fwd(g, B, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                                ys, gbuf, ubuf);
                for (size_t i = 0; i < BC; i++) eout[i] += ys[i];
            }
        } else {
            uenr_swiglu_fwd(g, B, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                            ys, gbuf, ubuf);
            for (size_t i = 0; i < BC; i++) eout[i] += ys[i];
        }
        uenr_rmsnorm_fwd(eout, ys, eout, y->post_ffn, B, C, eps);
        for (size_t i = 0; i < BC; i++) x[i] = x2[i] + eout[i];
    }

    uenr_rmsnorm_fwd(xfinal, ys, x, m->rms_final, B, C, eps);
    uenr_linear_fwd(g, logits_out, xfinal, m->lm_head, B, C, V);
    if (m->cfg.model.final_softcap > 0.0f) {
        const float scv = m->cfg.model.final_softcap;
        for (size_t i = 0; i < (size_t)B * V; i++) {
            float z = logits_out[i];
            if (z > 0.1f * scv || z < -0.1f * scv)
                logits_out[i] = scv * uenr_tanhf(z / scv);
        }
    }
    free(x); free(xa); free(q); free(krow); free(vrow); free(attn); free(proj);
    free(x2); free(xf); free(eout); free(ys); free(gxs); free(gys); free(xfinal);
    free(router); free(gbuf); free(ubuf); free(sc);
    free(ecnt); free(eoff); free(elist); free(cursor); free(eidx); free(egate);
    return UENR_OK;

multi_fail:
    free(x); free(xa); free(q); free(krow); free(vrow); free(attn); free(proj);
    free(x2); free(xf); free(eout); free(ys); free(gxs); free(gys); free(xfinal);
    free(router); free(gbuf); free(ubuf); free(sc);
    free(ecnt); free(eoff); free(elist); free(cursor); free(eidx); free(egate);
    return UENR_ERR_ALLOC;
}
