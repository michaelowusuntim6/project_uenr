#ifndef UENR_MATH_H
#define UENR_MATH_H

#include "uenr/simd.h"

/*
 * FMA-free formulas tuned for Ivy Bridge (AUDIT.md section 4).
 *
 * Every scalar function is written in mul/add (Horner) form so the compiler
 * emits vmulps/vaddps pairs rather than assuming an FMA that does not exist.
 * No libm transcendentals are used anywhere in this header's implementations.
 */

float uenr_expf(float x);
float uenr_logf(float x);
float uenr_sinf(float x);
float uenr_cosf(float x);
float uenr_sigmoidf(float x);
float uenr_siluf(float x);
float uenr_tanhf(float x);
float uenr_geluf(float x);
float uenr_rsqrtf(float x);
float uenr_rcpf(float x);
float uenr_sqrtf(float x);

/* Vector versions: 8-wide AVX (or 4-wide SSE fallback). */
uenr_v8f uenr_v_exp(uenr_v8f x);
uenr_v8f uenr_v_sigmoid(uenr_v8f x);
uenr_v8f uenr_v_silu(uenr_v8f x);
uenr_v8f uenr_v_tanh(uenr_v8f x);
uenr_v8f uenr_v_gelu(uenr_v8f x);

#endif /* UENR_MATH_H */
