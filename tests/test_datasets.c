/* Per-phase dataset discovery: multiple paths, deterministic preparation,
 * LoRA fallback to SFT, phase filtering and parser acceptance. */
#include "uenr/dataset.h"
#include "uenr/uenr.h"

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

static const char *TMP = "/tmp/uenr_datasets_test";

static void write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("FAIL write %s\n", path); fails++; return; }
    fputs(text, f);
    fclose(f);
}

static int count_lines(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int n = 0, c, seen = 0, last = 0;
    while ((c = fgetc(f)) != EOF) {
        seen = 1; last = c;
        if (c == '\n') n++;
    }
    fclose(f);
    if (!seen) return 0;
    return last == '\n' ? n : n + 1;
}

int main(void) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", TMP, TMP);
    if (system(cmd) != 0) return 1;

    char sft1[256], sft2[256], prefs[256], rl[256], out_sft[256];
    char out_lora[256], out_pairs[256], out_rl[256], out_train[256];
    char out_val[256], rep[256];
    snprintf(sft1, sizeof(sft1), "%s/sft1.jsonl", TMP);
    snprintf(sft2, sizeof(sft2), "%s/sft2.jsonl", TMP);
    snprintf(prefs, sizeof(prefs), "%s/prefs.jsonl", TMP);
    snprintf(rl, sizeof(rl), "%s/rl.jsonl", TMP);
    snprintf(out_sft, sizeof(out_sft), "%s/out_sft.jsonl", TMP);
    snprintf(out_lora, sizeof(out_lora), "%s/out_lora.jsonl", TMP);
    snprintf(out_pairs, sizeof(out_pairs), "%s/out_pairs.jsonl", TMP);
    snprintf(out_rl, sizeof(out_rl), "%s/out_rl.jsonl", TMP);
    snprintf(out_train, sizeof(out_train), "%s/train.bin", TMP);
    snprintf(out_val, sizeof(out_val), "%s/val.bin", TMP);
    snprintf(rep, sizeof(rep), "%s/duplicate_section.toml", TMP);

    write_text(sft1, "{\"prompt\":\"p1\",\"completion\":\"c1\"}\n"
                     "{\"prompt\":\"p2\",\"completion\":\"c2\"}\n");
    write_text(sft2, "{\"prompt\":\"p3\",\"completion\":\"c3\"}\n"
                     "{\"prompt\":\"p4\",\"completion\":\"c4\"}\n");
    write_text(prefs, "{\"prompt\":\"p\",\"chosen\":\"a\",\"rejected\":\"b\"}\n"
                      "{\"prompt\":\"q\",\"chosen\":\"c\",\"rejected\":\"d\"}\n");
    write_text(rl, "{\"prompt\":\"r1\"}\n{\"prompt\":\"r2\"}\n{\"prompt\":\"r3\"}\n");

    UenrConfig cfg;
    if (uenr_config_load("config/uenr.toml", &cfg) != UENR_OK) {
        printf("FAIL config\n"); return 1;
    }
    snprintf(cfg.paths.sft_jsonl, sizeof(cfg.paths.sft_jsonl), "%s", out_sft);
    snprintf(cfg.paths.lora_jsonl, sizeof(cfg.paths.lora_jsonl), "%s", out_lora);
    snprintf(cfg.paths.pairs_jsonl, sizeof(cfg.paths.pairs_jsonl), "%s", out_pairs);
    snprintf(cfg.paths.rl_jsonl, sizeof(cfg.paths.rl_jsonl), "%s", out_rl);
    snprintf(cfg.paths.train_bin, sizeof(cfg.paths.train_bin), "%s", out_train);
    snprintf(cfg.paths.val_bin, sizeof(cfg.paths.val_bin), "%s", out_val);
    cfg.datasets.sft.n_paths = 2;
    snprintf(cfg.datasets.sft.paths[0], UENR_MAX_STR, "%s", sft1);
    snprintf(cfg.datasets.sft.paths[1], UENR_MAX_STR, "%s", sft2);
    snprintf(cfg.datasets.sft.format, sizeof(cfg.datasets.sft.format),
             "jsonl_prompt_completion");
    cfg.datasets.sft.max_samples = 0;
    cfg.datasets.sft.shuffle_seed = 42;
    cfg.datasets.lora.n_paths = 0;
    snprintf(cfg.datasets.lora.format, sizeof(cfg.datasets.lora.format), "inherit");
    cfg.datasets.dpo.n_paths = 1;
    snprintf(cfg.datasets.dpo.paths[0], UENR_MAX_STR, "%s", prefs);
    snprintf(cfg.datasets.dpo.format, sizeof(cfg.datasets.dpo.format),
             "jsonl_prompt_chosen_rejected");
    cfg.datasets.dpo.shuffle_seed = 43;
    cfg.datasets.rl.n_paths = 1;
    snprintf(cfg.datasets.rl.paths[0], UENR_MAX_STR, "%s", rl);
    snprintf(cfg.datasets.rl.format, sizeof(cfg.datasets.rl.format), "jsonl_prompt_only");
    cfg.datasets.rl.shuffle_seed = 44;

    check(uenr_data_prepare_phase(&cfg, NULL, "sft") == UENR_OK,
          "prepare --phase sft succeeds without a tokenizer");
    check(count_lines(out_sft) == 4, "multiple SFT paths concatenate to 4 samples");
    check(count_lines(out_train) < 0,
          "phase sft does not produce the pretrain UDAT file");

    check(uenr_data_prepare_phase(&cfg, NULL, "lora") == UENR_OK,
          "prepare --phase lora succeeds");
    check(count_lines(out_lora) == 4, "empty LoRA paths fall back to SFT sources");

    check(uenr_data_prepare_phase(&cfg, NULL, "dpo") == UENR_OK,
          "prepare --phase dpo succeeds");
    check(count_lines(out_pairs) == 2, "DPO preference records prepared");
    check(uenr_data_prepare_phase(&cfg, NULL, "rl") == UENR_OK,
          "prepare --phase rl succeeds");
    check(count_lines(out_rl) == 3, "RL prompt records prepared");

    /* Cap and deterministic ordering: two runs with the same seed agree. */
    cfg.datasets.sft.max_samples = 2;
    check(uenr_data_prepare_phase(&cfg, NULL, "sft") == UENR_OK, "SFT cap run");
    char *first = NULL;
    {
        FILE *f = fopen(out_sft, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            first = malloc((size_t)n + 1);
            if (first) { size_t got = fread(first, 1, (size_t)n, f); first[got] = '\0'; }
            fclose(f);
        }
    }
    check(count_lines(out_sft) == 2, "max_samples caps the prepared SFT set");
    check(uenr_data_prepare_phase(&cfg, NULL, "sft") == UENR_OK, "SFT deterministic rerun");
    int same = 1;
    if (first) {
        FILE *f = fopen(out_sft, "rb");
        char buf[8192]; size_t got = f ? fread(buf, 1, sizeof(buf) - 1, f) : 0;
        if (f) fclose(f);
        buf[got] = '\0';
        same = strcmp(buf, first) == 0;
        free(first);
    }
    check(same, "shuffle_seed produces deterministic output");

    /* Parser accepts repeated [datasets.sft] sections and multiple paths. */
    char *base = NULL;
    {
        FILE *f = fopen("config/uenr.toml", "rb");
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        base = malloc((size_t)n + 1);
        size_t got = fread(base, 1, (size_t)n, f); base[got] = '\0'; fclose(f);
    }
    FILE *cf = fopen(rep, "wb");
    if (!cf) return 1;
    fputs(base, cf);
    fputs("\n[datasets.sft]\npaths = [\"a.jsonl\", \"b.jsonl\"]\n", cf);
    fclose(cf);
    free(base);
    UenrConfig parsed;
    check(uenr_config_load(rep, &parsed) == UENR_OK &&
          parsed.datasets.sft.n_paths == 2, "strict parser accepts multiple dataset paths");

    snprintf(cmd, sizeof(cmd), "rm -rf %s", TMP);
    if (system(cmd) != 0) return 1;
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
