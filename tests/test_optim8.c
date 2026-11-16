#include "uenr/ops.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { N = 256 };
    float p1[N], p2[N], g[N], m[N], v[N], ms[N / UENR_ADAM8_BLOCK + 1];
    int8_t m8[N];
    for (int i = 0; i < N; i++) { p1[i] = 1.0f + 0.01f * i; p2[i] = p1[i]; m[i] = v[i] = 0.0f; m8[i] = 0; }
    memset(ms, 0, sizeof(ms));
    for (int s = 1; s <= 200; s++) {
        for (int i = 0; i < N; i++) g[i] = p1[i] * 0.5f + 0.1f * sinf((float)s + (float)i);
        uenr_adamw_step(p1, g, m, v, N, s, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, NULL);
        uenr_adamw_step_m8(p2, g, m8, ms, v, N, s, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    }
    float worst = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = fabsf(p1[i] - p2[i]);
        if (e > worst) worst = e;
        if (!isfinite(p2[i])) { printf("FAIL non-finite int8 Adam\n"); return 1; }
    }
    printf("int8-m Adam vs fp32: float p0=%.4f int8 p0=%.4f worst|diff|=%.4f\n",
           p1[0], p2[0], worst);
    int ok = worst < 0.05f && p2[0] < 0.5f;
    printf(ok ? "all passed\n" : "FAIL int8 Adam diverges\n");
    return ok ? 0 : 1;
}
