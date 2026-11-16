#include "uenr/math.h"

#include <math.h>
#include <stdio.h>

static int check(const char *name, float got, float want, float tol) {
    float err = fabsf(got - want);
    if (err > tol) {
        printf("FAIL %-8s got=%g want=%g err=%g\n", name, got, want, err);
        return 1;
    }
    printf("ok   %-8s got=%g\n", name, got);
    return 0;
}

int main(void) {
    int fails = 0;
    fails += check("exp(0)", uenr_expf(0.0f), 1.0f, 1e-5f);
    fails += check("exp(1)", uenr_expf(1.0f), 2.7182818f, 1e-4f);
    fails += check("exp(-2)", uenr_expf(-2.0f), 0.13533528f, 1e-4f);
    fails += check("sigmoid(0)", uenr_sigmoidf(0.0f), 0.5f, 1e-5f);
    fails += check("silu(1)", uenr_siluf(1.0f), 0.7310586f, 1e-4f);
    fails += check("rsqrt(4)", uenr_rsqrtf(4.0f), 0.5f, 1e-6f);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
