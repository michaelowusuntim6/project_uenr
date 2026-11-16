#include "uenr/reward.h"
#include "uenr/dataset.h"
#include "uenr/io.h"
#include "uenr/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define URWD_MAGIC 0x44575255u /* "URWD" */

UenrResult uenr_reward_load(UenrRewardModel *r, const char *path) {
    if (!r || !path) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    if (size < 5 * sizeof(uint32_t)) { free(data); return UENR_ERR_INVALID; }
    uint32_t *h = (uint32_t *)data;
    if (h[0] != URWD_MAGIC || h[1] != 1) { free(data); return UENR_ERR_INVALID; }
    int C = (int)h[2];
    if (size != 5 * sizeof(uint32_t) + sizeof(float) + (size_t)C * sizeof(float)) {
        free(data);
        return UENR_ERR_INVALID;
    }
    r->C = C;
    r->w = (float *)malloc((size_t)C * sizeof(float));
    if (!r->w) { free(data); return UENR_ERR_ALLOC; }
    memcpy(&r->bias, (unsigned char *)data + 5 * sizeof(uint32_t), sizeof(float));
    memcpy(r->w, (unsigned char *)data + 5 * sizeof(uint32_t) + sizeof(float),
           (size_t)C * sizeof(float));
    free(data);
    return UENR_OK;
}

void uenr_reward_free(UenrRewardModel *r) {
    if (!r) return;
    free(r->w);
    r->w = NULL;
    r->C = 0;
    r->bias = 0.0f;
}

static UenrResult save_reward(const UenrRewardModel *r, const char *path) {
    size_t hdr = 5 * sizeof(uint32_t);
    size_t bytes = hdr + sizeof(float) + (size_t)r->C * sizeof(float);
    unsigned char *b = (unsigned char *)malloc(bytes);
    if (!b) return UENR_ERR_ALLOC;
    uint32_t *h = (uint32_t *)b;
    h[0] = URWD_MAGIC; h[1] = 1; h[2] = (uint32_t)r->C; h[3] = 0; h[4] = 0;
    memcpy(b + hdr, &r->bias, sizeof(float));
    memcpy(b + hdr + sizeof(float), r->w, (size_t)r->C * sizeof(float));
    UenrResult res = uenr_write_file_atomic(path, b, bytes);
    free(b);
    return res;
}

static UenrResult build_ids(const UenrConfig *cfg, const UenrTokenizer *tok,
                            const char *prompt, const char *completion,
                            int **ids_out, int *n_out, int *pn_out) {
    int *pids = NULL, pn = 0;
    if (uenr_chat_infer(tok, cfg->infer.system, prompt, &pids, &pn) != UENR_OK)
        return UENR_ERR_INVALID;
    if (pn >= cfg->model.max_seq) { free(pids); return UENR_ERR_INVALID; }
    int *cids = NULL, cn = 0;
    if (uenr_tok_encode(tok, completion, &cids, &cn) != UENR_OK) {
        free(pids);
        return UENR_ERR_INVALID;
    }
    int n = pn + cn + 1;
    if (n > cfg->model.max_seq) n = cfg->model.max_seq;
    int *ids = (int *)malloc((size_t)n * sizeof(int));
    if (!ids) { free(pids); free(cids); return UENR_ERR_ALLOC; }
    int k = 0;
    for (int i = 0; i < pn && k < n; i++) ids[k++] = pids[i];
    for (int i = 0; i < cn && k < n; i++) ids[k++] = cids[i];
    if (k < n) ids[k++] = uenr_tok_id(tok, UENR_ROLE_END);
    free(pids); free(cids);
    *ids_out = ids; *n_out = k; *pn_out = pn;
    return UENR_OK;
}

static void mean_pool(float *dst, const UenrActs *a, int pn, int n) {
    int C = a->C;
    for (int c = 0; c < C; c++) dst[c] = 0.0f;
    int start = pn < n ? pn : n - 1;
    int cnt = n - start;
    if (cnt <= 0) { cnt = 1; start = n - 1; }
    for (int t = start; t < n; t++)
        for (int c = 0; c < C; c++) dst[c] += a->xfinal[(size_t)t * C + c];
    float inv = 1.0f / (float)cnt;
    for (int c = 0; c < C; c++) dst[c] *= inv;
}

float uenr_reward_score_seq(UenrModel *base, UenrGemm *g, UenrActs *a,
                            const UenrRewardModel *r,
                            const int *ids, int n, int pn) {
    float *mean = (float *)malloc((size_t)base->C * sizeof(float));
    if (!mean) return 0.0f;
    int count = 0;
    uenr_model_forward(base, g, a, ids, n, n, NULL, -100, 0, &count);
    mean_pool(mean, a, pn, n);
    float s = r->bias;
    for (int c = 0; c < base->C; c++) s += r->w[c] * mean[c];
    free(mean);
    return s;
}

UenrResult uenr_reward_train(const UenrConfig *cfg) {
    UenrTokenizer *tok = NULL;
    UENR_TRY(uenr_tok_load(cfg, cfg->paths.tokenizer, &tok));
    UenrPairData pairs;
    if (uenr_pairs_load(cfg->paths.pairs_jsonl, &pairs) != UENR_OK) {
        uenr_tok_free(tok);
        return UENR_ERR_INVALID;
    }
    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 11);
    UenrModel base;
    if (uenr_model_init(&base, cfg, &rng) != UENR_OK) {
        uenr_pairs_free(&pairs); uenr_tok_free(tok); return UENR_ERR_ALLOC;
    }
    int step0 = 0;
    if (uenr_model_load(&base, cfg->paths.model_out, &step0, 0) != UENR_OK) {
        fprintf(stderr, "reward: cannot load base %s\n", cfg->paths.model_out);
        uenr_model_free(&base); uenr_pairs_free(&pairs); uenr_tok_free(tok);
        return UENR_ERR_IO;
    }
    UenrGemm g;
    uenr_gemm_init(&g, NULL);
    UenrActs a;
    if (uenr_acts_init(&a, cfg, cfg->model.max_seq, cfg->model.max_seq) != UENR_OK) {
        uenr_gemm_destroy(&g); uenr_model_free(&base);
        uenr_pairs_free(&pairs); uenr_tok_free(tok); return UENR_ERR_ALLOC;
    }
    UenrRewardModel rw;
    rw.C = base.C;
    rw.bias = 0.0f;
    rw.w = (float *)calloc((size_t)rw.C, sizeof(float));
    if (!rw.w) { uenr_acts_free(&a); uenr_gemm_destroy(&g); uenr_model_free(&base);
                 uenr_pairs_free(&pairs); uenr_tok_free(tok); return UENR_ERR_ALLOC; }

    float *mw = (float *)malloc((size_t)base.C * sizeof(float));
    float *ml = (float *)malloc((size_t)base.C * sizeof(float));
    if (!mw || !ml) { free(mw); free(ml); uenr_reward_free(&rw); uenr_acts_free(&a);
                      uenr_gemm_destroy(&g); uenr_model_free(&base);
                      uenr_pairs_free(&pairs); uenr_tok_free(tok); return UENR_ERR_ALLOC; }

    int steps = cfg->reward.steps;
    int correct = 0;
    float last_loss = 0.0f;
    for (int step = 1; step <= steps; step++) {
        const UenrPair *p = &pairs.pairs[uenr_rng_range(&rng, pairs.count)];
        int *wid = NULL, *lid = NULL; int wn = 0, ln = 0, wp = 0, lp = 0;
        if (build_ids(cfg, tok, p->prompt, p->chosen, &wid, &wn, &wp) != UENR_OK ||
            build_ids(cfg, tok, p->prompt, p->rejected, &lid, &ln, &lp) != UENR_OK) {
            free(wid); free(lid);
            continue;
        }
        uenr_model_forward(&base, &g, &a, wid, wn, wn, NULL, -100, 0, NULL);
        mean_pool(mw, &a, wp, wn);
        uenr_model_forward(&base, &g, &a, lid, ln, ln, NULL, -100, 0, NULL);
        mean_pool(ml, &a, lp, ln);
        float sw = rw.bias, sl = rw.bias;
        for (int c = 0; c < rw.C; c++) { sw += rw.w[c] * mw[c]; sl += rw.w[c] * ml[c]; }
        float d = sw - sl;
        float sig = 1.0f / (1.0f + expf(-d));
        float loss = -logf(sig + 1e-12f);
        float gsig = 1.0f - sig;
        for (int c = 0; c < rw.C; c++)
            rw.w[c] += cfg->reward.lr * gsig * (mw[c] - ml[c]);
        rw.bias += cfg->reward.lr * gsig;
        if (d > 0) correct++;
        last_loss = loss;
        free(wid); free(lid);
        if (step % 50 == 0 || step == steps)
            printf("[reward] step %d/%d loss %.4f acc %.2f\n",
                   step, steps, (double)last_loss, (double)correct / step);
    }
    UenrResult res = save_reward(&rw, cfg->reward.checkpoint);
    printf("[reward] saved %s\n", cfg->reward.checkpoint);
    free(mw); free(ml);
    uenr_reward_free(&rw);
    uenr_acts_free(&a); uenr_gemm_destroy(&g); uenr_model_free(&base);
    uenr_pairs_free(&pairs); uenr_tok_free(tok);
    return res;
}
