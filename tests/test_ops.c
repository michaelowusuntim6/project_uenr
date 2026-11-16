#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ok(int cond, const char *what) {
    if (!cond) { printf("FAIL %s\n", what); fails++; }
    else printf("ok   %s\n", what);
}

static void test_rmsnorm(void) {
    enum { T = 4, D = 12 };
    float x[T * D], w[D], y[T * D], rstd[T], dx[T * D], dy[T * D];
    float dw[D], r[D * T];
    UenrRng rng;
    uenr_rng_seed(&rng, 99);
    for (int i = 0; i < T * D; i++) x[i] = uenr_rng_normal(&rng);
    for (int d = 0; d < D; d++) w[d] = 0.5f + uenr_rng_f32(&rng);
    for (int i = 0; i < T * D; i++) { r[i] = uenr_rng_normal(&rng); dy[i] = r[i]; }
    memset(dw, 0, sizeof(dw));

    uenr_rmsnorm_fwd(y, rstd, x, w, T, D, 1e-6f);

    float maxerr = 0.0f;
    for (int t = 0; t < T; t++) {
        float ss = 0.0f;
        for (int d = 0; d < D; d++) ss += x[t * D + d] * x[t * D + d];
        float rv = 1.0f / sqrtf(ss / D + 1e-6f);
        for (int d = 0; d < D; d++) {
            float want = x[t * D + d] * rv * w[d];
            float e = fabsf(y[t * D + d] - want);
            if (e > maxerr) maxerr = e;
        }
    }
    ok(maxerr < 1e-5f, "rmsnorm forward matches reference");

    float loss0 = 0.0f;
    for (int i = 0; i < T * D; i++) loss0 += y[i] * r[i];
    uenr_rmsnorm_bwd(dx, dw, dy, x, w, rstd, T, D);

    float fy[T * D], gr[T * D];
    const float eps = 1e-3f;
    float worst = 0.0f;
    for (int i = 0; i < T * D; i += 7) {
        float old = x[i];
        x[i] = old + eps;
        uenr_rmsnorm_fwd(fy, rstd, x, w, T, D, 1e-6f);
        float lp = 0.0f;
        for (int k = 0; k < T * D; k++) lp += fy[k] * r[k];
        x[i] = old - eps;
        uenr_rmsnorm_fwd(fy, rstd, x, w, T, D, 1e-6f);
        float lm = 0.0f;
        for (int k = 0; k < T * D; k++) lm += fy[k] * r[k];
        x[i] = old;
        float num = (lp - lm) / (2 * eps);
        float e = fabsf(num - dx[i]);
        (void)gr;
        if (e > worst) worst = e;
    }
    (void)loss0;
    ok(worst < 2e-3f, "rmsnorm backward matches finite differences");
}

static void test_rope(void) {
    UenrRope r;
    int hd = 8, max_seq = 16;
    ok(uenr_rope_init(&r, max_seq, hd / 2, 10000.0f) == UENR_OK, "rope init");
    int T = 12, H = 2;
    float *x = (float *)malloc((size_t)T * H * hd * sizeof(float));
    float *d = (float *)malloc((size_t)T * H * hd * sizeof(float));
    UenrRng rng;
    uenr_rng_seed(&rng, 3);
    for (int i = 0; i < T * H * hd; i++) x[i] = uenr_rng_normal(&rng);
    memcpy(d, x, (size_t)T * H * hd * sizeof(float));

    uenr_rope_apply(&r, d, T, 12, H, hd);
    float changed = 0.0f;
    for (int i = 0; i < T * H * hd; i++) changed += fabsf(d[i] - x[i]);
    ok(changed > 1.0f, "rope rotates values");

    uenr_rope_apply_bwd(&r, d, T, 12, H, hd);
    float worst = 0.0f;
    for (int i = 0; i < T * H * hd; i++) {
        float e = fabsf(d[i] - x[i]);
        if (e > worst) worst = e;
    }
    ok(worst < 1e-4f, "rope forward+backward is identity");
    free(x); free(d);
    uenr_rope_destroy(&r);
}

static void test_attn(void) {
    int B = 2, S = 5, H = 4, G = 2, hd = 4;
    int T = B * S, KV = G * hd;
    float *q = (float *)malloc((size_t)T * H * hd * sizeof(float));
    float *k = (float *)malloc((size_t)T * KV * sizeof(float));
    float *v = (float *)malloc((size_t)T * KV * sizeof(float));
    float *out = (float *)malloc((size_t)T * H * hd * sizeof(float));
    float *ref = (float *)malloc((size_t)T * H * hd * sizeof(float));
    float *probs = (float *)malloc((size_t)B * H * S * S * sizeof(float));
    UenrRng rng;
    uenr_rng_seed(&rng, 5);
    for (int i = 0; i < T * H * hd; i++) q[i] = uenr_rng_normal(&rng);
    for (int i = 0; i < T * KV; i++) { k[i] = uenr_rng_normal(&rng); v[i] = uenr_rng_normal(&rng); }

    float scale = 1.0f / sqrtf((float)hd);
    uenr_attn_fwd(out, probs, q, k, v, B, S, H, G, hd, scale, NULL);

    float worst = 0.0f;
    int group = H / G;
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
        int g = h / group;
        for (int s = 0; s < S; s++) {
            float sc[16], mx = -1e30f, sum = 0.0f;
            for (int j = 0; j <= s; j++) {
                float d = 0.0f;
                for (int dd = 0; dd < hd; dd++)
                    d += q[((b * S + s) * H + h) * hd + dd] * k[((b * S + j) * G + g) * hd + dd];
                sc[j] = d * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            for (int j = 0; j <= s; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            for (int dd = 0; dd < hd; dd++) {
                float acc = 0.0f;
                for (int j = 0; j <= s; j++)
                    acc += (sc[j] / sum) * v[((b * S + j) * G + g) * hd + dd];
                int idx = ((b * S + s) * H + h) * hd + dd;
                float e = fabsf(acc - out[idx]);
                if (e > worst) worst = e;
            }
        }
    }
    ok(worst < 2e-5f, "GQA causal attention matches reference");
    free(q); free(k); free(v); free(out); free(ref); free(probs);
}

static void test_softmax_ce(void) {
    enum { T = 3, V = 9 };
    float logits[T * V], dlog[T * V];
    int targets[T] = { 1, 5, 8 };
    UenrRng rng;
    uenr_rng_seed(&rng, 11);
    for (int i = 0; i < T * V; i++) logits[i] = uenr_rng_normal(&rng) * 2.0f;

    float loss = uenr_softmax_ce_fwd(logits, targets, T, V, -100, dlog, 1, NULL);
    float want = 0.0f;
    for (int t = 0; t < T; t++) {
        float mx = logits[t * V];
        for (int i = 1; i < V; i++) if (logits[t * V + i] > mx) mx = logits[t * V + i];
        float sum = 0.0f;
        for (int i = 0; i < V; i++) sum += expf(logits[t * V + i] - mx);
        want += logf(sum) + mx - logits[t * V + targets[t]];
    }
    ok(fabsf(loss - want) < 2e-4f, "softmax+CE loss matches reference");

    float maxsum = 0.0f;
    for (int t = 0; t < T; t++) {
        float s = 0.0f;
        for (int i = 0; i < V; i++) s += dlog[t * V + i];
        if (fabsf(s) > maxsum) maxsum = fabsf(s);
    }
    ok(maxsum < 1e-5f, "dlogits rows sum to zero");

    /* finite difference of one logit */
    int t = 1, i = 5;
    float eps = 1e-3f;
    float old = logits[t * V + i];
    logits[t * V + i] = old + eps;
    float lp = uenr_softmax_ce_fwd(logits, targets, T, V, -100, NULL, 0, NULL);
    logits[t * V + i] = old - eps;
    float lm = uenr_softmax_ce_fwd(logits, targets, T, V, -100, NULL, 0, NULL);
    logits[t * V + i] = old;
    float num = (lp - lm) / (2 * eps);
    ok(fabsf(num - dlog[t * V + i]) < 2e-3f, "softmax+CE gradient matches finite difference");
}

static void test_schedule(void) {
    UenrConfig cfg;
    uenr_config_load("config/uenr.toml", &cfg);
    int total = uenr_schedule_plan_steps(&cfg, 1000000);
    ok(total >= cfg.sched.min_steps_before_stop && total <= cfg.train.max_steps,
       "schedule step budget clamps");
    ok(uenr_schedule_lr(&cfg, 0, total, 0) == 0.0f, "lr starts at zero");
    float mid = uenr_schedule_lr(&cfg, cfg.train.warmup_steps / 2, total, 0);
    ok(mid > 0.0f && mid < cfg.train.lr, "warmup is linear");
    float end = uenr_schedule_lr(&cfg, total, total, 0);
    ok(end > 0.0f && end <= cfg.train.lr, "lr ends positive");

    UenrSchedState s;
    uenr_sched_state_init(&s);
    for (int i = 0; i < 10; i++) uenr_sched_state_update(&s, &cfg, 5.0f);
    ok(s.ema > 4.9f, "ema tracks loss");
}

int main(void) {
    test_rmsnorm();
    test_rope();
    test_attn();
    test_softmax_ce();
    test_schedule();
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
