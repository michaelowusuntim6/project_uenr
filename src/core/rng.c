#include "uenr/rng.h"
#include "uenr/math.h"

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void uenr_rng_seed(UenrRng *r, uint64_t seed) {
    uint64_t x = seed;
    r->s0 = splitmix64(&x);
    r->s1 = splitmix64(&x);
    if ((r->s0 | r->s1) == 0) r->s0 = 1;
}

uint64_t uenr_rng_u64(UenrRng *r) {
    uint64_t x = r->s0;
    const uint64_t y = r->s1;
    r->s0 = y;
    x ^= x << 23;
    r->s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return r->s1 + y;
}

float uenr_rng_f32(UenrRng *r) {
    /* 24 significant bits -> [0, 1) */
    return (float)((uenr_rng_u64(r) >> 40) * (1.0 / 16777216.0));
}

float uenr_rng_normal(UenrRng *r) {
    float u1 = uenr_rng_f32(r);
    float u2 = uenr_rng_f32(r);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float rad = uenr_sqrtf(-2.0f * uenr_logf(u1));
    const float two_pi = 6.283185307179586f;
    return rad * uenr_cosf(two_pi * u2);
}

int uenr_rng_range(UenrRng *r, int n) {
    if (n <= 1) return 0;
    return (int)(uenr_rng_u64(r) % (uint64_t)n);
}

void uenr_rng_shuffle(UenrRng *r, int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = uenr_rng_range(r, i + 1);
        int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}
