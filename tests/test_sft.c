/* SFT data-path regression tests: autoregressive label alignment, assistant
 * span masking, tool/user masking, truncation, padding and ignore index. */
#include "uenr/dataset.h"
#include "uenr/tokenizer.h"
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

/* Assistant content and the <end> closing an assistant turn are the only
 * tokens the model may be asked to predict. */
static int verify_sample(const UenrTokenizer *t, const UenrConfig *cfg,
                         const UenrSample *s, const char *tag) {
    const int ign = cfg->tok.ignore_index;
    int ok = 1, supervised = 0, bad_shift = 0;
    int acontent = 0, aend = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->labels[i] == ign) continue;
        supervised++;
        if (i + 1 >= s->n) { bad_shift = 1; continue; }
        if (s->labels[i] != s->ids[i + 1]) bad_shift = 1;
        if (s->ids[i + 1] == cfg->tok.special_end) aend = 1;
        else acontent = 1;
    }
    /* user / tool / system / role marker positions must be masked */
    int i = 0;
    while (i < s->n) {
        int rid = s->ids[i];
        if ((rid == cfg->tok.special_user || rid == cfg->tok.special_tool_result ||
             rid == cfg->tok.special_system)) {
            int j = i + 1;
            while (j < s->n && s->ids[j] != cfg->tok.special_end) j++;
            for (int k = i; k <= j && k < s->n; k++)
                if (s->labels[k] != ign) ok = 0;
            i = j;
        }
        i++;
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "%s: every label is the following token", tag);
    check(bad_shift == 0, msg);
    snprintf(msg, sizeof(msg), "%s: assistant content supervised", tag);
    check(acontent && supervised > 1, msg);
    snprintf(msg, sizeof(msg), "%s: assistant end supervised", tag);
    check(aend, msg);
    snprintf(msg, sizeof(msg), "%s: user/tool/system spans masked", tag);
    check(ok, msg);
    snprintf(msg, sizeof(msg), "%s: final label ignored", tag);
    check(s->n > 0 && s->labels[s->n - 1] == ign, msg);
    return ok;
}

int main(void) {
    UenrConfig cfg;
    if (uenr_config_load("config/uenr.toml", &cfg) != UENR_OK) {
        printf("FAIL config\n");
        return 1;
    }
    cfg.model.vocab_size = 512;
    cfg.tok.train_bytes = 60000;

    const char *txt = "/tmp/uenr_sft_sample.txt";
    FILE *f = fopen(txt, "w");
    if (!f) return 1;
    for (int i = 0; i < 800; i++)
        fprintf(f, "Once upon a time there was a little robot number %d who liked "
                   "to count apples and oranges in the garden. The robot said hello "
                   "to the sun and the moon and the stars every single day.\n", i);
    fclose(f);

    const char *art = "/tmp/uenr_sft_tok.bin";
    if (uenr_tok_train(&cfg, txt, art, NULL, NULL) != UENR_OK) {
        printf("FAIL tokenizer train\n");
        return 1;
    }
    UenrTokenizer *t = NULL;
    if (uenr_tok_load(&cfg, art, &t) != UENR_OK) { printf("FAIL tokenizer load\n"); return 1; }

    const char *jsonl = "/tmp/uenr_sft_pairs.jsonl";
    f = fopen(jsonl, "w");
    if (!f) return 1;
    fprintf(f, "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"},"
               "{\"role\":\"assistant\",\"content\":\"Hello there\"}]}\n");
    fprintf(f, "{\"messages\":[{\"role\":\"system\",\"content\":\"You help\"},"
               "{\"role\":\"user\",\"content\":\"Weather?\"},"
               "{\"role\":\"assistant\",\"content\":\"<|tool_call|>{\\\"name\\\":\\\"weather\\\"}<|end|>\"}]}\n");
    fprintf(f, "{\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Accra?\"},"
               "{\"role\":\"assistant\",\"content\":\"<|tool_call|>{\\\"city\\\":\\\"Accra\\\"}<|end|>\"},"
               "{\"role\":\"tool\",\"content\":\"{\\\"temp\\\":25}\"},"
               "{\"role\":\"assistant\",\"content\":\"It is 25 C.\"}]}\n");
    fclose(f);

    UenrSFTData d;
    if (uenr_sft_load(&cfg, t, jsonl, &d) != UENR_OK) {
        printf("FAIL sft load\n");
        return 1;
    }
    check(d.count == 3, "loads all three conversations");
    for (int i = 0; i < d.count; i++) {
        char tag[32];
        snprintf(tag, sizeof(tag), "sample%d", i);
        verify_sample(t, &cfg, &d.samples[i], tag);
    }

    /* Truncation and padding: every non-ignored label is still the token that
     * follows its input position; padding carries ignore_index. */
    const int seq = 12, nseq = 6, pad = cfg.tok.pad_id;
    int *in = malloc((size_t)seq * nseq * sizeof(int));
    int *lb = malloc((size_t)seq * nseq * sizeof(int));
    UenrRng rng;
    uenr_rng_seed(&rng, 7);
    if (uenr_sft_batch(&d, &rng, seq, nseq, in, lb, pad, cfg.tok.ignore_index) != 0) {
        printf("FAIL sft batch\n");
        return 1;
    }
    int bad = 0, pad_bad = 0;
    for (int r = 0; r < nseq; r++) {
        int *ri = in + (size_t)r * seq;
        int *rl = lb + (size_t)r * seq;
        int copy = 0;
        while (copy < seq && ri[copy] != pad) copy++;
        for (int i = 0; i < seq; i++) {
            if (rl[i] == cfg.tok.ignore_index) continue;
            if (rl[i] < 0 || rl[i] >= cfg.model.vocab_size) bad = 1;
            if (i + 1 < copy && rl[i] != ri[i + 1]) bad = 1;
        }
        for (int i = copy; i < seq; i++)
            if (rl[i] != cfg.tok.ignore_index) pad_bad = 1;
    }
    check(bad == 0, "truncated batch keeps valid next-token targets");
    check(pad_bad == 0, "padding is ignore_index");
    free(in); free(lb);
    uenr_sft_free(&d);

    /* Configurable ignore index must flow through load and batching. */
    cfg.tok.ignore_index = -7;
    if (uenr_sft_load(&cfg, t, jsonl, &d) != UENR_OK) { printf("FAIL reload\n"); return 1; }
    int ign_ok = 1, any_ign = 0;
    for (int i = 0; i < d.count; i++) {
        for (int k = 0; k < d.samples[i].n; k++) {
            int l = d.samples[i].labels[k];
            if (l == -7) any_ign = 1;
            else if (l == -100) ign_ok = 0;
        }
    }
    check(any_ign && ign_ok, "configured ignore index used by loader");
    uenr_sft_free(&d);

    uenr_tok_free(t);
    remove(jsonl);
    remove(art);
    remove(txt);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
