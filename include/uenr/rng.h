#ifndef UENR_RNG_H
#define UENR_RNG_H

#include <stdint.h>

/* xorshift128+ : fast, deterministic, no allocation. */
typedef struct {
    uint64_t s0, s1;
} UenrRng;

void     uenr_rng_seed(UenrRng *r, uint64_t seed);
uint64_t uenr_rng_u64(UenrRng *r);
float    uenr_rng_f32(UenrRng *r);          /* [0, 1) */
float    uenr_rng_normal(UenrRng *r);       /* N(0,1) */
int      uenr_rng_range(UenrRng *r, int n); /* [0, n) */
void     uenr_rng_shuffle(UenrRng *r, int *idx, int n);

#endif /* UENR_RNG_H */
