#include "uenr/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void expect(int cond, const char *what) {
    if (!cond) { printf("FAIL %s\n", what); fails++; }
    else printf("ok   %s\n", what);
}

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[got] = '\0';
    *n = got;
    return b;
}

/* Create a config file consisting of the shipped config plus `extra`, load it
 * and require rejection. */
static void reject_with(const char *extra, const char *what) {
    const char *tmp = "/tmp/uenr_cfg_bad.toml";
    size_t n = 0;
    char *base = slurp("config/uenr.toml", &n);
    FILE *f = fopen(tmp, "wb");
    int ok = base && f;
    if (ok) {
        fwrite(base, 1, n, f);
        fputs(extra, f);
        fclose(f);
    }
    free(base);
    UenrConfig c;
    UenrResult r = ok ? uenr_config_load(tmp, &c) : UENR_ERR_IO;
    expect(r != UENR_OK, what);
    remove(tmp);
}

static void reject_file(const char *text, const char *what) {
    const char *tmp = "/tmp/uenr_cfg_bad.toml";
    FILE *f = fopen(tmp, "wb");
    if (!f) { printf("FAIL write %s\n", what); fails++; return; }
    fputs(text, f);
    fclose(f);
    UenrConfig c;
    UenrResult r = uenr_config_load(tmp, &c);
    expect(r != UENR_OK, what);
    remove(tmp);
}

int main(void) {
    UenrConfig c;
    UenrResult r = uenr_config_load("config/uenr.toml", &c);
    if (!uenr_ok(r)) {
        printf("FAIL load err=%d\n", (int)r);
        return 1;
    }
    expect(c.model.hidden == 192, "hidden parsed");
    expect(c.model.layers == 4, "layers parsed");
    expect(c.model.heads == 6, "heads parsed");
    expect(c.model.heads * c.model.head_dim == c.model.hidden, "derived shape valid");
    expect(c.moe.enabled == 1, "moe flag parsed");
    expect(c.moe.num_experts == 24, "expert count parsed");
    expect(c.moe.experts_per_tok == 1, "top-k parsed");
    expect(c.train.batch_size > 0, "batch size parsed");
    expect(c.lora.n_targets == 4, "string array parsed");
    expect(!strcmp(c.lora.targets[0], "q_proj"), "string array content");
    expect(c.sched.kind[0] != '\0', "string parsed");

    /* Valid strict overrides, including an escaped quote. */
    {
        const char *tmp = "/tmp/uenr_cfg_good.toml";
        size_t n = 0;
        char *base = slurp("config/uenr.toml", &n);
        FILE *f = base ? fopen(tmp, "wb") : NULL;
        if (f) {
            fwrite(base, 1, n, f);
            fputs("[run]\nname = \"a\\\"b\"\nresume = true\n"
                  "[training]\nlr = 5.0e-4\n", f);
            fclose(f);
        }
        free(base);
        r = f ? uenr_config_load(tmp, &c) : UENR_ERR_IO;
        expect(r == UENR_OK && !strcmp(c.run.name, "a\"b") && c.run.resume == 1 &&
               c.train.lr > 4.9e-4f && c.train.lr < 5.1e-4f,
               "valid overrides and escapes accepted");
        remove(tmp);
    }

    reject_with("[run]\nunknown_key = 1\n", "unknown key rejected");
    reject_with("[unknown_section]\nx = 1\n", "unknown section rejected");
    reject_with("[run]\nresume = yes\n", "malformed boolean rejected");
    reject_with("[run]\nresume = TRUE\n", "uppercase boolean rejected");
    reject_with("[run]\nthreads = 2x\n", "trailing junk in integer rejected");
    reject_with("[run]\nthreads = 99999999999999999999\n", "integer overflow rejected");
    reject_with("[training]\nlr = 1e999\n", "float overflow rejected");
    reject_with("[training]\nlr = nan\n", "NaN float rejected");
    reject_with("[run]\nname = \"unterminated\n", "unterminated string rejected");
    reject_with("[run]\nname = \"x\" junk\n", "trailing junk after string rejected");
    reject_with("[run]\nname = \"bad\\qescape\"\n", "unknown string escape rejected");
    reject_with("[lora]\ntargets = [\"q_proj\", 3]\n", "malformed string array rejected");
    reject_with("[model]\nheads = 5\n", "invalid range/derived shape rejected");
    reject_with("[model]\nlayers = 2\nkv_shared_layers = 1\nattn_pattern = \"SG\"\n",
                "shared layer without a same-type donor rejected");
    reject_with("[training]\nbeta1 = 2.0\n", "beta out of range rejected");
    reject_file("[run\nname = \"x\"\n", "malformed section header rejected");
    reject_file("threads = 2\n", "key outside a section rejected");
    reject_file("[run]\nname \"x\"\n", "missing equals rejected");

    /* Every shipped profile must satisfy the strict parser and validation. */
    {
        const char *profiles[] = { "config/uenr.toml", "config/scale.toml",
                                   "config/draft.toml", "config/tokenizer16k.toml" };
        for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); i++) {
            char what[128];
            snprintf(what, sizeof(what), "shipped profile loads: %s", profiles[i]);
            expect(uenr_config_load(profiles[i], &c) == UENR_OK, what);
        }
    }

    if (fails == 0) printf("config ok\n");
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
