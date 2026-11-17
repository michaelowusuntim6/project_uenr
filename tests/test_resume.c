/* Checkpoint discovery: periodic steps are found when no final model exists,
 * headers (not filenames) decide the newest checkpoint, corrupt files are
 * skipped, and the selected checkpoint restores weights + optimizer state. */
#include "uenr/model.h"
#include "uenr/train.h"
#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
    c->train.adafactor = 1;
}

int main(void) {
    const char *dir = "/tmp/uenr_resume_test";
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
    if (system(cmd) != 0) return 1;

    UenrConfig cfg;
    config_tiny(&cfg);
    UenrRng rng;
    uenr_rng_seed(&rng, 3);
    UenrModel m;
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK ||
        uenr_model_opt_init(&m, 1) != UENR_OK) {
        printf("FAIL model init\n");
        return 1;
    }
    for (size_t i = 0; i < m.alloc_floats; i++) m.params[i] = 0.25f + (float)(i % 89) * 0.001f;
    for (size_t i = 0; i < m.ada_n; i++) m.ada[i] = 0.75f;

    char p3[256], p12[256], pbad[256], phead[256];
    snprintf(p3, sizeof(p3), "%s/step_000003.bin", dir);
    snprintf(p12, sizeof(p12), "%s/step_000012.bin", dir);
    snprintf(pbad, sizeof(pbad), "%s/step_000099.bin", dir);
    snprintf(phead, sizeof(phead), "%s/step_999999.bin", dir);
    if (uenr_model_save(&m, p3, 3, 1) != UENR_OK ||
        uenr_model_save(&m, p12, 12, 1) != UENR_OK ||
        uenr_model_save(&m, phead, 5, 1) != UENR_OK) {
        printf("FAIL save\n");
        return 1;
    }
    FILE *fb = fopen(pbad, "wb");
    if (fb) { fputs("not a checkpoint", fb); fclose(fb); }

    char picked[512] = {0};
    int hstep = -1;
    int found = uenr_latest_checkpoint(dir, picked, sizeof(picked), &hstep);
    check(found >= 0, "finds a periodic checkpoint without a final model");
    check(hstep == 12, "selects the newest checkpoint by header step");
    check(strstr(picked, "step_000012.bin") != NULL,
          "ignores higher-numbered corrupt and stale header files");

    /* Loading the selected checkpoint restores weights and optimizer state. */
    UenrRng r2;
    uenr_rng_seed(&r2, 3);
    UenrModel m2;
    int step = 0;
    int restored = uenr_model_init(&m2, &cfg, &r2) == UENR_OK &&
                   uenr_model_opt_init(&m2, 1) == UENR_OK &&
                   uenr_model_load(&m2, picked, &step, 1) == UENR_OK;
    check(restored && step == 12, "checkpoint header step restores the step counter");
    int same = restored;
    for (size_t i = 0; i < m2.alloc_floats && same; i += 997)
        if (fabsf(m2.params[i] - (0.25f + (float)(i % 89) * 0.001f)) > 1e-6f) same = 0;
    check(same, "weights restored from the periodic checkpoint");
    int opt_ok = restored;
    for (size_t i = 0; i < m2.ada_n && opt_ok; i += 293)
        if (fabsf(m2.ada[i] - 0.75f) > 1e-6f) opt_ok = 0;
    check(opt_ok, "optimizer state restored from the periodic checkpoint");
    uenr_model_free(&m2);
    uenr_model_free(&m);

    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
    if (system(cmd) != 0) return 1;
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
