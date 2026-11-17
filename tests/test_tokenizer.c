#include "uenr/tokenizer.h"
#include "uenr/uenr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(int cond, const char *what) {
    if (!cond) printf("FAIL %s\n", what);
    else printf("ok   %s\n", what);
    return cond ? 0 : 1;
}

int main(void) {
    UenrConfig cfg;
    if (uenr_config_load("config/uenr.toml", &cfg) != UENR_OK) {
        printf("FAIL config\n");
        return 1;
    }
    cfg.model.vocab_size = 512;
    cfg.tok.train_bytes = 60000;

    const char *txt = "/tmp/uenr_tok_sample.txt";
    FILE *f = fopen(txt, "w");
    if (!f) return 1;
    for (int i = 0; i < 800; i++) {
        fprintf(f, "Once upon a time there was a little robot number %d who liked to count "
                   "apples and oranges in the garden. The robot said hello to the sun and "
                   "the moon and the stars every single day.\n", i);
    }
    fclose(f);

    const char *art = "/tmp/uenr_tok_test.bin";
    if (uenr_tok_train(&cfg, txt, art, NULL, NULL) != UENR_OK) {
        printf("FAIL train\n");
        return 1;
    }
    UenrTokenizer *t = NULL;
    if (uenr_tok_load(&cfg, art, &t) != UENR_OK) {
        printf("FAIL load\n");
        return 1;
    }

    int fails = 0;
    fails += check(uenr_tok_vocab(t) == 512, "vocab size");
    fails += check(uenr_tok_id(t, UENR_ROLE_BOS) == cfg.tok.bos_id, "bos id");
    fails += check(uenr_tok_id(t, UENR_ROLE_ASSISTANT) == cfg.tok.special_assistant, "assistant id");

    const char *msg = "Hello, little robot! Once upon a time 12345.";
    int *ids = NULL, n = 0;
    if (uenr_tok_encode(t, msg, &ids, &n) != UENR_OK) { printf("FAIL encode\n"); return 1; }
    fails += check(n > 0 && n < (int)strlen(msg), "bpe compresses");
    char *back = NULL;
    size_t blen = 0;
    if (uenr_tok_decode(t, ids, n, &back, &blen) != UENR_OK) { printf("FAIL decode\n"); return 1; }
    fails += check(strcmp(back, msg) == 0, "round-trip");
    if (strcmp(back, msg) != 0) printf("  got: '%s'\n", back);
    free(back);
    free(ids);

    {
        int *a = NULL, *b = NULL, an = 0, bn = 0;
        int ra = uenr_tok_encode(t, "caf\xC3\xA9", &a, &an);
        int rb = uenr_tok_encode(t, "cafe\xCC\x81", &b, &bn);
        int same = ra == UENR_OK && rb == UENR_OK && an == bn;
        for (int i = 0; same && i < an; i++) if (a[i] != b[i]) same = 0;
        fails += check(same, "NFC-lite composes base+combining mark");
        free(a); free(b);
    }

    int *cids = NULL, *labels = NULL, cn = 0;
    const int ign = cfg.tok.ignore_index;
    if (uenr_chat_sft(t, "You are helpful.", "Tell me a story.",
                      "Sure.<|tool_call|>story()<|end|>", ign,
                      &cids, &labels, &cn) != UENR_OK) {
        printf("FAIL chat_sft\n");
        return 1;
    }
    int has_tool = 0, supervised_tool = 0, supervised_user = 0;
    int shifted = 1, last_ignored = (cn > 0 && labels[cn - 1] == ign);
    int role_context = 0, end_supervised = 0;
    int arole_pos = -1, tstart = -1, tend = -1;
    for (int i = 0; i < cn; i++) {
        if (cids[i] == cfg.tok.special_assistant && arole_pos < 0) arole_pos = i;
        if (cids[i] == cfg.tok.special_tool_call) { has_tool = 1; tstart = i; }
        if (cids[i] == cfg.tok.special_end && tstart >= 0 && tend < 0) tend = i;
        /* every supervised label is the *next* token (never a same-position
         * target, even though equal adjacent values may coincide) */
        if (labels[i] != ign) {
            if (i + 1 >= cn || labels[i] != cids[i + 1]) shifted = 0;
        }
    }
    /* no label may point at any token inside the user span (marker, text, end) */
    {
        int ustart = -1, uend = -1;
        for (int i = 0; i < cn; i++) {
            if (cids[i] == cfg.tok.special_user && ustart < 0) ustart = i;
            else if (ustart >= 0 && cids[i] == cfg.tok.special_end && uend < 0) uend = i;
        }
        if (ustart > 0 && uend >= ustart)
            for (int t = ustart - 1; t < uend && t < cn; t++)
                if (labels[t] != ign) supervised_user = 1;
    }
    if (arole_pos >= 0 && arole_pos + 1 < cn)
        role_context = labels[arole_pos] == cids[arole_pos + 1] && labels[arole_pos] != ign;
    if (tstart > 0)
        supervised_tool = labels[tstart - 1] == cfg.tok.special_tool_call;
    if (tend > 0)
        end_supervised = labels[tend - 1] == cfg.tok.special_end;
    fails += check(has_tool, "tool_call markup atomic");
    fails += check(shifted, "labels are next-token targets");
    fails += check(last_ignored, "final token has no target");
    fails += check(role_context, "assistant role marker predicts first content token");
    fails += check(supervised_tool, "tool_call supervised in assistant span");
    fails += check(end_supervised, "assistant end marker supervised");
    fails += check(!supervised_user, "user span masked");
    free(cids);
    free(labels);

    char *sys = NULL;
    int *iids = NULL, in = 0;
    if (uenr_chat_infer(t, "sys", "hi", &iids, &in) != UENR_OK) { printf("FAIL chat_infer\n"); return 1; }
    fails += check(in >= 7, "infer template built");
    free(iids);
    (void)sys;

    uenr_tok_free(t);
    remove(txt);
    remove(art);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
