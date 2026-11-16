#include "uenr/eval.h"
#include "uenr/dataset.h"
#include "uenr/gemm.h"
#include "uenr/io.h"
#include "uenr/math.h"
#include "uenr/model.h"
#include "uenr/rng.h"
#include "uenr/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

UenrResult uenr_eval(const UenrConfig *cfg, const char *model_path,
                     const char *split, int batches, int seq, int nseq) {
    if (!cfg) return UENR_ERR_INVALID;
    if (!split) split = cfg->eval.split;
    if (batches <= 0) batches = cfg->eval.batches;
    if (seq <= 0) seq = cfg->eval.seq_len;
    if (nseq <= 0) nseq = cfg->eval.nseq;
    const char *path = (!split || !strcmp(split, "train")) ? cfg->paths.train_bin
                                                          : cfg->paths.val_bin;
    UenrTokenData data;
    { UenrResult rr = uenr_data_load(path, &data); if (rr != UENR_OK) { fprintf(stderr, "eval: data_load err %d\n", (int)rr); return rr; } }
    if (data.vocab != cfg->model.vocab_size) { uenr_data_free(&data); return UENR_ERR_SHAPE; }

    const char *mp = model_path ? model_path : cfg->paths.model_out;
    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 99);
    UenrModel m;
    { UenrResult rr = uenr_model_init(&m, cfg, &rng); if (rr != UENR_OK) { fprintf(stderr, "eval: model_init err %d\n", (int)rr); uenr_data_free(&data); return rr; } }
    int step = 0;
    if (uenr_model_load(&m, mp, &step, 0) != UENR_OK) {
        fprintf(stderr, "eval: cannot load %s\n", mp);
        uenr_model_free(&m); uenr_data_free(&data);
        return UENR_ERR_IO;
    }
    UenrGemm g;
    if (uenr_gemm_init(&g, NULL) != UENR_OK) {
        uenr_model_free(&m); uenr_data_free(&data); return UENR_ERR_ALLOC;
    }
    int T = nseq * seq;
    UenrActs a;
    if (uenr_acts_init(&a, cfg, T, seq) != UENR_OK) {
        fprintf(stderr, "eval: acts_init failed T=%d seq=%d\n", T, seq);
        uenr_gemm_destroy(&g); uenr_model_free(&m); uenr_data_free(&data);
        return UENR_ERR_ALLOC;
    }
    int *in = (int *)malloc((size_t)T * sizeof(int));
    int *tg = (int *)malloc((size_t)T * sizeof(int));
    if (!in || !tg) { free(in); free(tg); uenr_acts_free(&a); uenr_gemm_destroy(&g);
                      uenr_model_free(&m); uenr_data_free(&data); return UENR_ERR_ALLOC; }

    double loss = 0.0;
    long long count = 0;
    double t0 = now_s();
    for (int b = 0; b < batches; b++) {
        if (uenr_data_batch(&data, &rng, seq, nseq, in, tg) != 0) break;
        int c = 0;
        float l = uenr_model_forward(&m, &g, &a, in, T, seq, tg,
                                     cfg->tok.ignore_index, 0, &c);
        loss += l;
        count += c;
    }
    double dt = now_s() - t0;
    if (count == 0) count = 1;
    double mean = loss / (double)count;
    double ppl = (double)uenr_expf((float)mean);
    printf("[eval] %s: loss %.6f  ppl %.2f  tokens %lld  %.0f tok/s  step %d\n",
           path, mean, ppl, count, (double)count / (dt > 0 ? dt : 1e-9), step);

    UenrTokenizer *tok = NULL;
    if (uenr_tok_load(cfg, cfg->paths.tokenizer, &tok) == UENR_OK) {
        FILE *f = fopen(cfg->paths.data, "rb");
        if (f) {
            size_t want = 1u << 20;
            char *buf = (char *)malloc(want + 1);
            if (buf) {
                size_t got = fread(buf, 1, want, f);
                buf[got] = '\0';
                int *ids = NULL, n = 0;
                if (uenr_tok_encode(tok, buf, &ids, &n) == UENR_OK) {
                    char *back = NULL;
                    size_t blen = 0;
                    uenr_tok_decode(tok, ids, n, &back, &blen);
                    printf("[eval] tokenizer: %zu bytes -> %d tokens (%.2f bytes/token), round-trip %s\n",
                           got, n, n ? (double)got / n : 0.0,
                           (back && blen == got && memcmp(back, buf, got) == 0) ? "ok" : "differs");
                    free(back);
                    free(ids);
                }
                free(buf);
            }
            fclose(f);
        }
        uenr_tok_free(tok);
    }

    free(in); free(tg);
    uenr_acts_free(&a); uenr_gemm_destroy(&g); uenr_model_free(&m);
    uenr_data_free(&data);
    return UENR_OK;
}
