/* DPO/RL checkpoint policy: fail closed instead of using random reference
 * weights; explicit fallback from a missing configured reference to the base
 * model; invalid/incompatible references are rejected. */
#include "uenr/model.h"
#include "uenr/train.h"
#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static int check(int cond, const char *what) {
    if (!cond) printf("FAIL %s\n", what);
    else printf("ok   %s\n", what);
    if (!cond) fails++;
    return cond;
}

static void config_tiny(UenrConfig *c) {
    memset(c, 0, sizeof(*c));
    if (uenr_config_load("config/uenr.toml", c) != UENR_OK) {
        printf("FAIL config\n");
        exit(1);
    }
    c->model.vocab_size = 512;
    c->model.hidden = 32;
    c->model.layers = 2;
    c->model.heads = 2;
    c->model.kv_heads = 1;
    c->model.head_dim = 16;
    c->model.ffn = 32;
    c->model.kv_shared_layers = 0;
    c->model.max_seq = 64;
    c->model.attn_softcap = 0.0f;
    c->model.final_softcap = 0.0f;
    c->moe.enabled = 0;
    c->moe.shared_expert = 0;
    c->train.seq_len = 16;
    c->train.batch_size = 2;
    c->train.grad_accum = 1;
}

static int save_weights(const UenrConfig *cfg, const char *path,
                        float base, int step) {
    UenrRng rng;
    uenr_rng_seed(&rng, 5);
    UenrModel m;
    if (uenr_model_init(&m, cfg, &rng) != UENR_OK) return 0;
    for (size_t i = 0; i < m.alloc_floats; i++)
        m.params[i] = base + (float)(i % 71) * 0.001f;
    int ok = uenr_model_save(&m, path, step, 0) == UENR_OK;
    uenr_model_free(&m);
    return ok;
}

static int matches_base(const UenrModel *m, float base) {
    for (size_t i = 0; i < m->alloc_floats; i += 257)
        if (fabsf(m->params[i] - (base + (float)(i % 71) * 0.001f)) > 1e-6f)
            return 0;
    return 1;
}

int main(void) {
    const char *dir = "/tmp/uenr_ref_test";
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
    if (system(cmd) != 0) return 1;

    UenrConfig cfg;
    config_tiny(&cfg);
    char refp[256], basep[256], badp[256], missp[256], shap[256];
    snprintf(refp, sizeof(refp), "%s/ref.bin", dir);
    snprintf(basep, sizeof(basep), "%s/base.bin", dir);
    snprintf(badp, sizeof(badp), "%s/bad.bin", dir);
    snprintf(missp, sizeof(missp), "%s/missing.bin", dir);
    snprintf(shap, sizeof(shap), "%s/wrong_shape.bin", dir);
    if (!save_weights(&cfg, refp, 0.5f, 4) || !save_weights(&cfg, basep, 0.25f, 2)) {
        printf("FAIL save fixtures\n");
        return 1;
    }
    FILE *f = fopen(badp, "wb");
    if (f) { fputs("garbage", f); fclose(f); }
    UenrConfig other = cfg;
    other.model.vocab_size = 768;
    other.model.hidden = 48;
    other.model.heads = 3;
    other.model.head_dim = 16;
    other.model.ffn = 48;
    if (!save_weights(&other, shap, 0.9f, 1)) { printf("FAIL shape fixture\n"); return 1; }

    UenrModel ref;

    /* 1. configured reference exists and is valid */
    snprintf(cfg.dpo.ref_checkpoint, UENR_MAX_STR, "%s", refp);
    snprintf(cfg.paths.model_out, UENR_MAX_STR, "%s", basep);
    UenrResult r = uenr_align_load_reference(&ref, &cfg);
    check(r == UENR_OK && matches_base(&ref, 0.5f),
          "uses the configured reference checkpoint");
    uenr_model_free(&ref);

    /* 2. configured reference missing: explicit base fallback */
    snprintf(cfg.dpo.ref_checkpoint, UENR_MAX_STR, "%s", missp);
    r = uenr_align_load_reference(&ref, &cfg);
    check(r == UENR_OK && matches_base(&ref, 0.25f),
          "missing configured reference falls back to the base checkpoint");
    uenr_model_free(&ref);

    /* 3. neither reference nor base exists: fail closed */
    snprintf(cfg.paths.model_out, UENR_MAX_STR, "%s", missp);
    r = uenr_align_load_reference(&ref, &cfg);
    check(r != UENR_OK, "no reference and no base fails instead of random init");

    /* 4. configured reference exists but is corrupt: reject, no silent fallback */
    snprintf(cfg.dpo.ref_checkpoint, UENR_MAX_STR, "%s", badp);
    snprintf(cfg.paths.model_out, UENR_MAX_STR, "%s", basep);
    r = uenr_align_load_reference(&ref, &cfg);
    check(r != UENR_OK, "corrupt configured reference is rejected");
    uenr_model_free(&ref);

    /* 5. configured reference has an incompatible shape */
    snprintf(cfg.dpo.ref_checkpoint, UENR_MAX_STR, "%s", shap);
    r = uenr_align_load_reference(&ref, &cfg);
    check(r != UENR_OK, "incompatible reference checkpoint is rejected");
    uenr_model_free(&ref);

    /* 6. required policy checkpoint never falls back to random weights */
    snprintf(cfg.paths.model_out, UENR_MAX_STR, "%s", missp);
    UenrModel pol;
    r = uenr_align_load_required(&pol, &cfg, cfg.paths.model_out, 1, "test policy");
    check(r != UENR_OK, "missing policy checkpoint fails instead of random init");
    uenr_model_free(&pol);

    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
    if (system(cmd) != 0) return 1;
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
