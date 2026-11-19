/* Optimizer state: save/restore round trip, idempotent trainer initialization,
 * step-counter preservation and mismatched-kind rejection. */
#include "uenr/model.h"
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

static void fill(float *p, size_t n, float base) {
    for (size_t i = 0; i < n; i++) p[i] = base + (float)(i % 97) * 0.001f;
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
    c->model.flash_attn = 0;
    c->moe.enabled = 0;
    c->moe.shared_expert = 0;
    c->train.seq_len = 16;
    c->train.batch_size = 2;
    c->train.grad_accum = 1;
}

static void set_opt(UenrConfig *c, int kind) {
    c->train.optim8 = kind == 4;
    c->train.muon = kind == 3;
    c->train.adafactor = kind == 2;
    c->train.muon_ns_iters = 0; /* legacy row update; NS tested separately */
}

static void fill_state(UenrModel *m, int kind) {
    if (kind == 1) { fill(m->adam_m, m->alloc_floats, 0.11f);
                     fill(m->adam_v, m->alloc_floats, 0.22f); }
    else if (kind == 2) fill(m->ada, m->ada_n, 0.33f);
    else if (kind == 3) fill(m->muon_m, m->alloc_floats, 0.44f);
    else if (kind == 4) {
        for (size_t i = 0; i < m->alloc_floats; i++) m->adam_m8[i] = (int8_t)(i % 251);
        fill(m->adam_ms, m->adam_nblocks, 0.55f);
        fill(m->adam_v, m->alloc_floats, 0.66f);
    }
}

static int state_matches(const UenrModel *m, int kind) {
    if (kind == 2) {
        for (size_t i = 0; i < m->ada_n; i += 4093) {
            float want = 0.33f + (float)(i % 97) * 0.001f;
            if (fabsf(m->ada[i] - want) > 1e-7f) return 0;
        }
        return 1;
    }
    for (size_t i = 0; i < m->alloc_floats; i += 4093) {
        float want;
        if (kind == 1) { want = 0.11f + (float)(i % 97) * 0.001f;
                         if (fabsf(m->adam_m[i] - want) > 1e-7f) return 0;
                         want = 0.22f + (float)(i % 97) * 0.001f;
                         if (fabsf(m->adam_v[i] - want) > 1e-7f) return 0; }
        else if (kind == 3) { want = 0.44f + (float)(i % 97) * 0.001f;
                              if (fabsf(m->muon_m[i] - want) > 1e-7f) return 0; }
        else if (kind == 4) {
            if (m->adam_m8[i] != (int8_t)(i % 251)) return 0;
            want = 0.66f + (float)(i % 97) * 0.001f;
            if (fabsf(m->adam_v[i] - want) > 1e-7f) return 0;
        }
    }
    if (kind == 4)
        for (size_t i = 0; i < m->adam_nblocks; i += 97) {
            float want = 0.55f + (float)(i % 97) * 0.001f;
            if (fabsf(m->adam_ms[i] - want) > 1e-7f) return 0;
        }
    return 1;
}

static const char *opt_name(int kind) {
    return kind == 1 ? "adam" : kind == 2 ? "adafactor" :
           kind == 3 ? "muon" : "optim8";
}

static int roundtrip(int kind) {
    UenrConfig cfg;
    config_tiny(&cfg);
    set_opt(&cfg, kind);
    UenrRng rng;
    uenr_rng_seed(&rng, 11);
    UenrModel m;
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK) { printf("FAIL init\n"); return 0; }
    if (uenr_model_opt_init(&m, 1) != UENR_OK) { printf("FAIL opt init %s\n", opt_name(kind)); return 0; }
    fill(m.params, m.alloc_floats, 0.5f);
    fill_state(&m, kind);
    char path[128];
    snprintf(path, sizeof(path), "/tmp/uenr_optim_%s.bin", opt_name(kind));
    int ok = 1;
    if (uenr_model_save(&m, path, 7, 1) != UENR_OK) ok = 0;
    uenr_model_free(&m);

    uenr_rng_seed(&rng, 11);
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK) ok = 0;
    /* align.c-style pre-allocation before load, then load restores moments */
    if (ok && uenr_model_opt_init(&m, 1) != UENR_OK) ok = 0;
    int step = 0;
    if (ok && uenr_model_load(&m, path, &step, 1) != UENR_OK) ok = 0;
    ok = ok && step == 7;
    /* trainer's second opt_init must be idempotent */
    void *p_m = kind == 1 ? (void *)m.adam_m : kind == 2 ? (void *)m.ada :
               kind == 3 ? (void *)m.muon_m : (void *)m.adam_m8;
    void *p_v = kind == 2 ? NULL : (void *)m.adam_v;
    if (ok && uenr_model_opt_init(&m, 1) != UENR_OK) ok = 0;
    void *p_m2 = kind == 1 ? (void *)m.adam_m : kind == 2 ? (void *)m.ada :
                kind == 3 ? (void *)m.muon_m : (void *)m.adam_m8;
    void *p_v2 = kind == 2 ? NULL : (void *)m.adam_v;
    ok = ok && p_m == p_m2 && p_v == p_v2;
    ok = ok && state_matches(&m, kind);
    for (size_t i = 0; i < m.alloc_floats; i += 1009)
        if (fabsf(m.params[i] - (0.5f + (float)(i % 97) * 0.001f)) > 1e-6f) ok = 0;

    /* One more optimizer step must run from the restored state. */
    if (ok) {
        for (size_t i = 0; i < m.alloc_floats; i++) m.grad[i] = 0.01f * ((float)(i % 7) - 3.0f);
        uenr_model_adamw(&m, step + 1, 1e-3f, 0.9f, 0.99f, 1e-8f, 0.01f, NULL);
        int finite = 1;
        for (size_t i = 0; i < m.alloc_floats; i++)
            if (!isfinite(m.params[i])) { finite = 0; break; }
        ok = ok && finite;
    }

    /* Mismatched optimizer kind: a fresh model configured for a different
     * optimizer must reject the checkpoint before mutating its weights. */
    if (ok) {
        UenrConfig other;
        config_tiny(&other);
        set_opt(&other, kind == 1 ? 2 : 1);
        UenrModel m2;
        UenrRng r2;
        uenr_rng_seed(&r2, 11);
        if (uenr_model_init(&m2, &other, &r2) != UENR_OK ||
            uenr_model_opt_init(&m2, 1) != UENR_OK) {
            ok = 0;
        } else {
            float before = m2.params[0];
            UenrResult mism = uenr_model_load(&m2, path, &step, 1);
            ok = ok && mism == UENR_ERR_SHAPE && m2.params[0] == before;
            uenr_model_free(&m2);
        }
    }

    uenr_model_free(&m);
    remove(path);
    return ok;
}

static void test_muon_ns(void) {
    enum { R = 12, C = 8 };
    float x[R * C], out[R * C];
    unsigned seed = 12345u;
    for (int i = 0; i < R * C; i++) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = ((float)((seed >> 8) & 0xffff) / 65535.0f - 0.5f) * 3.0f;
    }
    int ok = uenr_muon_orthogonalize(out, x, R, C, 5);
    check(ok, "muon NS: orthogonalization returns success");
    int finite = 1;
    float mass = 0.0f;
    for (int i = 0; i < R * C; i++) {
        if (!isfinite(out[i])) finite = 0;
        mass += out[i] * out[i];
    }
    check(finite && mass > 0.1f && mass < 100.0f,
          "muon NS: iterate stays finite with O(1) mass");
    /* The iteration is transpose-equivariant: orthogonalizing a wide matrix
     * must equal the transpose of orthogonalizing its tall counterpart. */
    float wt[C * R], ot[C * R];
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) wt[j * R + i] = x[i * C + j];
    ok = uenr_muon_orthogonalize(ot, wt, C, R, 5) && ok;
    float maxdiff = 0.0f;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            float d = fabsf(out[i * C + j] - ot[j * R + i]);
            if (d > maxdiff) maxdiff = d;
        }
    check(ok && maxdiff < 1e-5f, "muon NS: tall and wide paths agree by transpose");
}

int main(void) {
    test_muon_ns();
    for (int kind = 1; kind <= 4; kind++) {
        char what[96];
        int ok = roundtrip(kind);
        snprintf(what, sizeof(what), "%s: state save/restore + idempotent init", opt_name(kind));
        check(ok, what);
    }
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
