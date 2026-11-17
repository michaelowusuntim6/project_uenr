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

static void test_rope_scaling(void) {
    UenrRope a, b;
    ok(uenr_rope_init(&a, 64, 8, 10000.0f) == UENR_OK, "rope init");
    ok(uenr_rope_init_ex(&b, 64, 8, 10000.0f, 2.0f, 0, 32.0f, 1.0f, 1.0f) == UENR_OK,
       "linear rope init");
    /* scale=2 halves every angle: cos_b(p,j) = cos_a(p/2,j) */
    float worst = 0.0f;
    for (int p = 0; p < 64; p += 2) {
        int q = p / 2;
        for (int j = 0; j < 8; j++) {
            float e = fabsf(b.cos_tab[(size_t)p * 8 + j] - a.cos_tab[(size_t)q * 8 + j]);
            if (e > worst) worst = e;
        }
    }
    ok(worst < 5e-3f, "linear RoPE scale halves position");
    uenr_rope_destroy(&a); uenr_rope_destroy(&b);
    UenrRope y;
    ok(uenr_rope_init_ex(&y, 64, 8, 10000.0f, 4.0f, 1, 32.0f, 1.0f, 1.0f) == UENR_OK,
       "yarn rope init");
    /* high-frequency (small j) dims are untouched by the YaRN ramp */
    UenrRope base;
    uenr_rope_init(&base, 64, 8, 10000.0f);
    float e0 = fabsf(y.cos_tab[8 + 0] - base.cos_tab[8 + 0]);
    ok(e0 < 5e-3f, "YaRN leaves high-frequency dims unscaled");
    uenr_rope_destroy(&y); uenr_rope_destroy(&base);

    /* partial RoPE: unrotated half stays identity */
    UenrRope p;
    ok(uenr_rope_init_partial(&p, 8, 8, 10000.0f, 0.25f, 1.0f, 0, 32.0f, 1.0f, 1.0f) == UENR_OK,
       "partial rope init");
    float u[16];
    for (int i = 0; i < 16; i++) u[i] = (float)(i + 1);
    float before[16]; memcpy(before, u, sizeof(u));
    uenr_rope_apply(&p, u, 1, 1, 1, 16);
    float unchanged = 0.0f;
    for (int i = 2; i < 8; i++) unchanged += fabsf(u[i] - before[i]);
    for (int i = 10; i < 16; i++) unchanged += fabsf(u[i] - before[i]);
    ok(unchanged < 1e-6f, "partial RoPE leaves non-rotary dims unchanged");
    uenr_rope_destroy(&p);
}

static void test_vnorm(void) {
    enum { T = 3, D = 4 };
    float x[T * D], y[T * D], d[T * D], dx[T * D], rstd[T];
    UenrRng rng; uenr_rng_seed(&rng, 91);
    for (int i = 0; i < T * D; i++) { x[i] = uenr_rng_normal(&rng); d[i] = uenr_rng_normal(&rng); }
    uenr_rmsnorm_noscale_fwd(y, rstd, x, T, D, 1e-6f);
    float worst_rms = 0.0f;
    for (int t = 0; t < T; t++) {
        float ss = 0.0f;
        for (int i = 0; i < D; i++) ss += y[t * D + i] * y[t * D + i];
        float rms = sqrtf(ss / (float)D);
        if (fabsf(rms - 1.0f) > worst_rms) worst_rms = fabsf(rms - 1.0f);
    }
    ok(worst_rms < 1e-3f, "V-norm yields unit RMS per head");
    uenr_rmsnorm_noscale_bwd(dx, d, x, rstd, T, D);
    float worst = 0.0f, eps = 1e-3f;
    for (int i = 0; i < D; i++) {
        float old = x[i];
        x[i] = old + eps; float lp = 0.0f;
        { float xt[T * D], yt[T * D], rt[T]; memcpy(xt, x, sizeof(xt));
          uenr_rmsnorm_noscale_fwd(yt, rt, xt, T, D, 1e-6f);
          for (int k = 0; k < T * D; k++) lp += d[k] * yt[k]; }
        x[i] = old - eps; float lm = 0.0f;
        { float xt[T * D], yt[T * D], rt[T]; memcpy(xt, x, sizeof(xt));
          uenr_rmsnorm_noscale_fwd(yt, rt, xt, T, D, 1e-6f);
          for (int k = 0; k < T * D; k++) lm += d[k] * yt[k]; }
        x[i] = old;
        float num = (lp - lm) / (2 * eps);
        float e = fabsf(num - dx[i]);
        if (e > worst) worst = e;
    }
    ok(worst < 2e-3f, "V-norm backward matches finite differences");
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
    uenr_attn_fwd(out, probs, q, k, v, B, S, H, G, hd, scale, 0, 0.0f, NULL);

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

    /* sliding window: each query sees only the last W positions */
    int W = 3;
    uenr_attn_fwd(out, probs, q, k, v, B, S, H, G, hd, scale, W, 0.0f, NULL);
    worst = 0.0f;
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
        int g = h / group;
        for (int s = 0; s < S; s++) {
            int j0 = s >= W ? s - W + 1 : 0;
            float sc[16], mx = -1e30f, sum = 0.0f;
            for (int j = j0; j <= s; j++) {
                float d = 0.0f;
                for (int dd = 0; dd < hd; dd++)
                    d += q[((b * S + s) * H + h) * hd + dd] * k[((b * S + j) * G + g) * hd + dd];
                sc[j] = d * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            for (int j = j0; j <= s; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            for (int dd = 0; dd < hd; dd++) {
                float acc = 0.0f;
                for (int j = j0; j <= s; j++)
                    acc += (sc[j] / sum) * v[((b * S + j) * G + g) * hd + dd];
                int idx = ((b * S + s) * H + h) * hd + dd;
                float e = fabsf(acc - out[idx]);
                if (e > worst) worst = e;
            }
            for (int j = 0; j < j0; j++) {
                int idx = (((size_t)b * H + h) * S + s) * S + j;
                if (probs[idx] != 0.0f) worst = 1e9f;
            }
        }
    }
    ok(worst < 2e-5f, "sliding-window attention matches reference");
    free(q); free(k); free(v); free(out); free(ref); free(probs);
}

static void test_flash_attn(void) {
    int B = 2, S = 5, H = 4, G = 2, hd = 4;
    int T = B * S;
    float *q = malloc((size_t)T * H * hd * sizeof(float));
    float *k = malloc((size_t)T * G * hd * sizeof(float));
    float *v = malloc((size_t)T * G * hd * sizeof(float));
    float *o1 = malloc((size_t)T * H * hd * sizeof(float));
    float *o2 = malloc((size_t)T * H * hd * sizeof(float));
    float *dout = malloc((size_t)T * H * hd * sizeof(float));
    float *p = malloc((size_t)B * H * S * S * sizeof(float));
    float *lse = malloc((size_t)B * H * S * sizeof(float));
    float *dq1 = calloc((size_t)T * H * hd, sizeof(float));
    float *dk1 = calloc((size_t)T * G * hd, sizeof(float));
    float *dv1 = calloc((size_t)T * G * hd, sizeof(float));
    float *dq2 = calloc((size_t)T * H * hd, sizeof(float));
    float *dk2 = calloc((size_t)T * G * hd, sizeof(float));
    float *dv2 = calloc((size_t)T * G * hd, sizeof(float));
    UenrRng rng; uenr_rng_seed(&rng, 77);
    for (int i = 0; i < T * H * hd; i++) { q[i] = uenr_rng_normal(&rng); dout[i] = uenr_rng_normal(&rng); }
    for (int i = 0; i < T * G * hd; i++) { k[i] = uenr_rng_normal(&rng); v[i] = uenr_rng_normal(&rng); }
    float scale = 1.0f / sqrtf((float)hd);
    uenr_attn_fwd(o1, p, q, k, v, B, S, H, G, hd, scale, 0, 0.0f, NULL);
    uenr_flash_attn_fwd(o2, lse, q, k, v, B, S, H, G, hd, scale, 0, 0.0f, NULL);
    float worst = 0.0f;
    for (int i = 0; i < T * H * hd; i++) { float e = fabsf(o1[i] - o2[i]); if (e > worst) worst = e; }
    ok(worst < 2e-5f, "flash attention forward matches stored-softmax");
    UenrResult abr = uenr_attn_bwd(dq1, dk1, dv1, dout, q, k, v, p, B, S, H, G, hd,
                                   scale, 0, 0.0f, NULL, NULL);
    ok(abr == UENR_OK, "attention backward reports success");
    uenr_flash_attn_bwd(dq2, dk2, dv2, dout, q, k, v, o2, lse, B, S, H, G, hd, scale, 0, 0.0f, NULL);
    float werr = 0.0f;
    for (int i = 0; i < T * H * hd; i++) { float e = fabsf(dq1[i] - dq2[i]); if (e > werr) werr = e; }
    for (int i = 0; i < T * G * hd; i++) { float e = fabsf(dk1[i] - dk2[i]); if (e > werr) werr = e; float f = fabsf(dv1[i] - dv2[i]); if (f > werr) werr = f; }
    ok(werr < 2e-4f, "flash attention backward matches stored-softmax");
    /* Supplied scratch must produce exactly the same gradients and must not
     * touch the caller's buffer. */
    float *scratch = malloc((size_t)S * sizeof(float));
    memset(dq1, 0, (size_t)T * H * hd * sizeof(float));
    memset(dk1, 0, (size_t)T * G * hd * sizeof(float));
    memset(dv1, 0, (size_t)T * G * hd * sizeof(float));
    abr = uenr_attn_bwd(dq1, dk1, dv1, dout, q, k, v, p, B, S, H, G, hd,
                        scale, 0, 0.0f, scratch, NULL);
    float serr = 0.0f;
    for (int i = 0; i < T * H * hd; i++) { float e = fabsf(dq1[i] - dq2[i]); if (e > serr) serr = e; }
    for (int i = 0; i < T * G * hd; i++) {
        float e = fabsf(dk1[i] - dk2[i]); if (e > serr) serr = e;
        float f = fabsf(dv1[i] - dv2[i]); if (f > serr) serr = f;
    }
    ok(abr == UENR_OK && serr < 2e-4f,
       "attention backward reuses supplied scratch without changing gradients");
    free(scratch);
    free(q); free(k); free(v); free(o1); free(o2); free(dout); free(p); free(lse);
    free(dq1); free(dk1); free(dv1); free(dq2); free(dk2); free(dv2);
}

static void test_softmax_ce(void) {
    enum { T = 3, V = 9 };
    float logits[T * V], dlog[T * V];
    int targets[T] = { 1, 5, 8 };
    UenrRng rng;
    uenr_rng_seed(&rng, 11);
    for (int i = 0; i < T * V; i++) logits[i] = uenr_rng_normal(&rng) * 2.0f;

    float loss = uenr_softmax_ce_fwd(logits, targets, T, V, -100, dlog, 1, NULL, 0.0f);
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

    /* in-place CE: dlogits aliases logits */
    {
        float logits2[T * V], dlog2[T * V];
        for (int i = 0; i < T * V; i++) logits2[i] = logits[i];
        float l2 = uenr_softmax_ce_fwd(logits2, targets, T, V, -100, logits2, 1, NULL, 0.0f);
        for (int i = 0; i < T * V; i++) dlog2[i] = logits2[i];
        float maxe = 0.0f;
        for (int i = 0; i < T * V; i++) { float e = fabsf(dlog2[i] - dlog[i]); if (e > maxe) maxe = e; }
        ok(fabsf(l2 - loss) < 2e-4f && maxe < 1e-6f, "softmax+CE works in place");
    }

    /* finite difference of one logit */
    int t = 1, i = 5;
    float eps = 1e-3f;
    float old = logits[t * V + i];
    logits[t * V + i] = old + eps;
    float lp = uenr_softmax_ce_fwd(logits, targets, T, V, -100, NULL, 0, NULL, 0.0f);
    logits[t * V + i] = old - eps;
    float lm = uenr_softmax_ce_fwd(logits, targets, T, V, -100, NULL, 0, NULL, 0.0f);
    logits[t * V + i] = old;
    float num = (lp - lm) / (2 * eps);
    ok(fabsf(num - dlog[t * V + i]) < 2e-3f, "softmax+CE gradient matches finite difference");

    /* z-loss: penalize logsumexp^2 and check its gradient */
    {
        float zc = 0.01f;
        float zloss = uenr_softmax_ce_fwd(logits, targets, T, V, -100, dlog, 1, NULL, zc);
        logits[t * V + i] = old + eps;
        float zp = uenr_softmax_ce_fwd(logits, targets, T, V, -100, NULL, 0, NULL, zc);
        logits[t * V + i] = old - eps;
        float zm = uenr_softmax_ce_fwd(logits, targets, T, V, -100, NULL, 0, NULL, zc);
        logits[t * V + i] = old;
        float znum = (zp - zm) / (2 * eps);
        ok(zloss > loss && fabsf(znum - dlog[t * V + i]) < 2e-3f,
           "z-loss adds quadratic penalty with matching gradient");
    }
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
    test_rope_scaling();
    test_vnorm();
    test_attn();
    test_flash_attn();
    test_softmax_ce();
    test_schedule();
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
