#include "uenr/ops.h"
#include "uenr/math.h"

/* Process-wide clamp shared by the AVX1/AVX2 SwiGLU copies and quantized
 * decode.  The model initializes it from training.swiglu_clamp. */
float uenr_swiglu_clamp_value = 0.0f;

void uenr_swiglu_set_clamp(float c) {
    uenr_swiglu_clamp_value = c > 0.0f ? c : 0.0f;
}

float uenr_swiglu_activation(float gate, float up) {
    float c = uenr_swiglu_clamp_value;
    float g = gate;
    float u = up;
    if (c > 0.0f) {
        if (g > c) g = c;
        if (u > c) u = c;
        if (u < -c) u = -c;
    }
    return uenr_siluf(g) * u;
}
