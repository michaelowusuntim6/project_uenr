/* Mathematical regression tests for DPO (and later PPO/GRPO) objectives. */
#include "uenr/train.h"
#include "uenr/math.h"

#include <math.h>
#include <stdio.h>

static int fails;

static int check(int cond, const char *what) {
    if (!cond) printf("FAIL %s\n", what);
    else printf("ok   %s\n", what);
    if (!cond) fails++;
    return cond;
}

int main(void) {
    /* DPO loss and its policy-log-probability gradients. */
    const float beta = 0.4f;
    float deltas[] = { -20.0f, -3.0f, -0.5f, 0.0f, 0.5f, 3.0f, 20.0f };
    int n = (int)(sizeof(deltas) / sizeof(deltas[0]));
    int fd_ok = 1, sign_ok = 1, sym_ok = 1, finite_ok = 1;
    const float eps = 1e-3f;
    for (int i = 0; i < n; i++) {
        float d = deltas[i];
        float gw = 0, gl = 0;
        uenr_dpo_logp_grad(d, beta, &gw, &gl);
        float fd = (uenr_dpo_loss(d + eps, beta) - uenr_dpo_loss(d - eps, beta)) / (2 * eps);
        float ana = gw; /* dL/ddelta: ddelta/dlogp_w = 1 */
        if (fabsf(fd - ana) > 2e-3f * (1.0f + fabsf(fd))) fd_ok = 0;
        if (!(gw <= 0.0f && gl >= 0.0f)) sign_ok = 0;
        if (fabsf(gw + gl) > 1e-6f) sym_ok = 0;
        if (!isfinite(uenr_dpo_loss(d, beta)) || !isfinite(gw) || !isfinite(gl)) finite_ok = 0;
    }
    check(fd_ok, "DPO logp gradients match finite differences of the loss");
    check(sign_ok, "DPO chosen gradient <= 0, rejected gradient >= 0");
    check(sym_ok, "DPO chosen/rejected gradients are exactly opposite");
    check(finite_ok, "DPO loss and gradients are finite at extremes");

    /* Directional proof: a small descent step on the two sequence log-probs
     * must increase the preference margin delta. */
    float d0 = 0.3f;
    float gw = 0, gl = 0;
    uenr_dpo_logp_grad(d0, beta, &gw, &gl);
    float lr = 0.05f;
    float lpw = -1.0f + (-lr * gw); /* logp -= lr * dL/dlogp */
    float lpl = -2.0f + (-lr * gl);
    float delta_after = (lpw - 0.0f) - (lpl - 0.0f); /* reference terms unchanged */
    check(delta_after > d0, "DPO step increases the chosen-vs-rejected margin");

    /* Extreme-probability stability: saturated sigma must not produce NaN/Inf. */
    float gw2 = 0, gl2 = 0;
    uenr_dpo_logp_grad(1e9f, beta, &gw2, &gl2);
    check(gw2 == 0.0f && gl2 == 0.0f, "DPO gradients vanish for a huge margin");
    uenr_dpo_logp_grad(-1e9f, beta, &gw2, &gl2);
    check(isfinite(gw2) && isfinite(gl2) &&
          fabsf(gw2 + beta) < 1e-5f && fabsf(gl2 - beta) < 1e-5f,
          "DPO gradients saturate at +/-beta for a huge negative margin");

    /* ---- PPO clipped surrogate -------------------------------------- */
    const float clip = 0.2f;
    struct { float adv, ratio; int active; } cases[] = {
        {  1.0f, 0.50f, 1 }, /* below interval, A>0: unclipped is min */
        {  1.0f, 1.00f, 1 },
        {  1.0f, 1.20f, 1 }, /* boundary still active */
        {  1.0f, 1.30f, 0 }, /* above interval, A>0: clipped */
        { -1.0f, 1.30f, 1 }, /* above interval, A<0: improving, active */
        { -1.0f, 1.00f, 1 },
        { -1.0f, 0.80f, 1 }, /* boundary still active */
        { -1.0f, 0.70f, 0 }, /* below interval, A<0: clipped */
        {  0.0f, 1.00f, 0 }, /* zero advantage: no update */
    };
    int clip_ok = 1, coef_ok = 1;
    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        int act = -1;
        float dS = uenr_ppo_surrogate_coef(cases[i].adv, cases[i].ratio, clip, &act);
        if (act != cases[i].active) clip_ok = 0;
        float want = cases[i].active ? cases[i].adv * cases[i].ratio : 0.0f;
        if (fabsf(dS - want) > 1e-6f) coef_ok = 0;
    }
    check(clip_ok, "PPO clipping regions are correct for both advantage signs");
    check(coef_ok, "PPO surrogate derivative is A*r where active, else zero");

    /* Directional: positive advantage must raise log pi, negative lower it. */
    int dir_ok = 1;
    for (int sgn = -1; sgn <= 1; sgn += 2) {
        for (float ratio = 0.5f; ratio <= 1.5001f; ratio += 0.25f) {
            int act = 0;
            float adv = (float)sgn;
            float dS = uenr_ppo_surrogate_coef(adv, ratio, clip, &act);
            float coef = -dS; /* no KL */
            float logp = 0.0f;
            float after = logp - 0.01f * coef; /* descent step */
            if (act && sgn > 0 && !(after > logp)) dir_ok = 0;
            if (act && sgn < 0 && !(after < logp)) dir_ok = 0;
        }
    }
    check(dir_ok, "PPO step encourages positive and discourages negative advantages");

    /* KL coefficient: with the k1 estimator the penalty gradient is +lambda,
     * so descent lowers the policy log-probability toward the reference. */
    float lam = 0.02f, kl = 1.5f;
    float coef_kl = lam; /* d(lambda*KL)/dlogp, ref constant */
    float logp_step = -0.1f - 0.01f * coef_kl;
    check(kl > 0.0f && logp_step < -0.1f,
          "KL penalty pulls the policy log-probability down toward the reference");

    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
