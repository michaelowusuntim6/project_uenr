#include "uenr/inference.h"
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

typedef struct { float p; int id; } ProbId;

static int cmp_desc(const void *a, const void *b) {
    const ProbId *x = (const ProbId *)a, *y = (const ProbId *)b;
    if (x->p < y->p) return 1;
    if (x->p > y->p) return -1;
    return x->id - y->id;
}

static int sample_next(const float *logits, int V, const UenrConfig *cfg,
                       UenrRng *rng, float *probs, ProbId *pairs, unsigned char *used) {
    if (cfg->infer.greedy) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < V; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    float temp = cfg->infer.temperature > 0.0f ? cfg->infer.temperature : 1.0f;
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs[i] = uenr_expf((logits[i] - mx) / temp);
        sum += probs[i];
    }
    float inv = uenr_rcpf(sum);
    for (int i = 0; i < V; i++) probs[i] *= inv;

    int K = cfg->infer.top_k;
    if (K > 0 && K < V) {
        memset(used, 0, (size_t)V);
        float thresh = 0.0f;
        for (int k = 0; k < K; k++) {
            int best = -1;
            float bv = -1.0f;
            for (int i = 0; i < V; i++)
                if (!used[i] && probs[i] > bv) { bv = probs[i]; best = i; }
            if (best < 0) break;
            used[best] = 1;
            thresh = bv;
        }
        float keep = 0.0f;
        for (int i = 0; i < V; i++) {
            if (probs[i] < thresh) probs[i] = 0.0f;
            keep += probs[i];
        }
        if (keep > 0.0f) {
            float k_inv = uenr_rcpf(keep);
            for (int i = 0; i < V; i++) probs[i] *= k_inv;
        }
    }

    float top_p = cfg->infer.top_p;
    if (top_p > 0.0f && top_p < 1.0f) {
        for (int i = 0; i < V; i++) { pairs[i].p = probs[i]; pairs[i].id = i; }
        qsort(pairs, (size_t)V, sizeof(ProbId), cmp_desc);
        float c = 0.0f;
        int cut = V;
        for (int i = 0; i < V; i++) {
            c += pairs[i].p;
            if (c >= top_p) { cut = i + 1; break; }
        }
        for (int i = 0; i < V; i++) probs[i] = 0.0f;
        float keep = 0.0f;
        for (int i = 0; i < cut; i++) { probs[pairs[i].id] = pairs[i].p; keep += pairs[i].p; }
        if (keep > 0.0f) {
            float k_inv = uenr_rcpf(keep);
            for (int i = 0; i < V; i++) probs[i] *= k_inv;
        }
    }

    float r = uenr_rng_f32(rng);
    float c = 0.0f;
    for (int i = 0; i < V; i++) {
        c += probs[i];
        if (r <= c) return i;
    }
    return V - 1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static UenrResult infer_one(const UenrConfig *cfg, const char *model_path,
                            UenrModel *m, UenrGemm *g, UenrActs *a,
                            UenrTokenizer *tok, UenrRng *rng,
                            const char *system, const char *prompt,
                            int max_new_tokens) {
    int *ids = NULL, n = 0;
    if (uenr_chat_infer(tok, system, prompt, &ids, &n) != UENR_OK)
        return UENR_ERR_INVALID;
    int max_seq = cfg->model.max_seq;
    if (n >= max_seq) {
        int drop = n - (max_seq - 1);
        memmove(ids, ids + drop, (size_t)(n - drop) * sizeof(int));
        n -= drop;
    }
    int gen_start = n;
    const int V = cfg->model.vocab_size;
    float *probs = (float *)malloc((size_t)V * sizeof(float));
    ProbId *pairs = (ProbId *)malloc((size_t)V * sizeof(ProbId));
    unsigned char *used = (unsigned char *)malloc((size_t)V);
    int use_kv = cfg->infer.kv_cache;
    float *step_logits = use_kv ? (float *)malloc((size_t)V * sizeof(float)) : NULL;
    UenrKVCache cache;
    memset(&cache, 0, sizeof(cache));
    if (!probs || !pairs || !used || (use_kv && !step_logits)) {
        free(ids); free(probs); free(pairs); free(used); free(step_logits);
        return UENR_ERR_ALLOC;
    }
    if (use_kv && uenr_kv_cache_init(&cache, cfg, max_seq) != UENR_OK) {
        use_kv = 0;
        free(step_logits);
        step_logits = NULL;
    }

    double t0 = now_s();
    int generated = 0;
    if (use_kv) {
        for (int i = 0; i < n; i++)
            uenr_model_forward_step(m, g, &cache, ids[i], i, step_logits);
        for (int step = 0; step < max_new_tokens && n < max_seq; step++) {
            int next = sample_next(step_logits, V, cfg, rng, probs, pairs, used);
            ids[n++] = next;
            generated++;
            if (next == cfg->tok.eos_id || next == cfg->tok.special_end) break;
            if (n >= max_seq) break;
            uenr_model_forward_step(m, g, &cache, next, n - 1, step_logits);
        }
    } else {
        for (int step = 0; step < max_new_tokens && n < max_seq; step++) {
            int count = 0;
            uenr_model_forward(m, g, a, ids, n, n, NULL, cfg->tok.ignore_index, 0, &count);
            const float *logits = a->logits + (size_t)(n - 1) * V;
            int next = sample_next(logits, V, cfg, rng, probs, pairs, used);
            ids[n++] = next;
            generated++;
            if (next == cfg->tok.eos_id || next == cfg->tok.special_end) break;
        }
    }
    double dt = now_s() - t0;
    if (dt <= 0) dt = 1e-9;

    int out_n = n - gen_start;
    if (out_n > 0 && (ids[n - 1] == cfg->tok.eos_id || ids[n - 1] == cfg->tok.special_end))
        out_n--;
    char *text = NULL;
    if (out_n > 0) uenr_tok_decode(tok, ids + gen_start, out_n, &text, NULL);
    printf("%s\n", text ? text : "");
    free(text);
    printf("[%d tokens, %.1f tok/s]\n", generated, (double)generated / dt);

    uenr_kv_cache_free(&cache);
    free(step_logits);
    free(ids); free(probs); free(pairs); free(used);
    return UENR_OK;
}

static UenrResult infer_setup(const UenrConfig *cfg, const char *model_path,
                              UenrModel *m, UenrGemm *g, UenrActs *a,
                              UenrTokenizer **tok_out, UenrRng *rng) {
    const char *mp = model_path ? model_path : cfg->paths.model_out;
    if (uenr_file_exists(mp) != UENR_OK) {
        fprintf(stderr, "infer: model %s not found (train first)\n", mp);
        return UENR_ERR_IO;
    }
    UenrRng init_rng;
    uenr_rng_seed(&init_rng, 1);
    UENR_TRY(uenr_model_init(m, cfg, &init_rng));
    int step = 0;
    UENR_TRY(uenr_model_load(m, mp, &step, 0));
    if (uenr_gemm_init(g, NULL) != UENR_OK) return UENR_ERR_ALLOC;
    if (uenr_acts_init(a, cfg, cfg->model.max_seq, cfg->model.max_seq) != UENR_OK)
        return UENR_ERR_ALLOC;
    UenrResult r = uenr_tok_load(cfg, cfg->paths.tokenizer, tok_out);
    if (r != UENR_OK) return r;
    uenr_rng_seed(rng, (uint64_t)cfg->run.seed + 7);
    return UENR_OK;
}

UenrResult uenr_infer(const UenrConfig *cfg, const char *model_path,
                      const char *system, const char *prompt,
                      int max_new_tokens, int greedy) {
    UenrConfig c = *cfg;
    if (greedy) c.infer.greedy = 1;
    if (max_new_tokens <= 0) max_new_tokens = c.infer.max_new_tokens;
    if (!system) system = c.infer.system;

    UenrModel m;
    UenrGemm g;
    UenrActs a;
    UenrTokenizer *tok = NULL;
    UenrRng rng;
    memset(&m, 0, sizeof(m));
    memset(&g, 0, sizeof(g));
    memset(&a, 0, sizeof(a));
    UenrResult r = infer_setup(&c, model_path, &m, &g, &a, &tok, &rng);
    if (r == UENR_OK) {
        printf("user> %s\nassistant> ", prompt);
        fflush(stdout);
        r = infer_one(&c, model_path, &m, &g, &a, tok, &rng, system, prompt, max_new_tokens);
    }
    uenr_tok_free(tok);
    uenr_acts_free(&a);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
    return r;
}

UenrResult uenr_infer_chat(const UenrConfig *cfg, const char *model_path,
                           int max_new_tokens, int greedy) {
    UenrConfig c = *cfg;
    if (greedy) c.infer.greedy = 1;
    if (max_new_tokens <= 0) max_new_tokens = c.infer.max_new_tokens;

    UenrModel m;
    UenrGemm g;
    UenrActs a;
    UenrTokenizer *tok = NULL;
    UenrRng rng;
    memset(&m, 0, sizeof(m));
    memset(&g, 0, sizeof(g));
    memset(&a, 0, sizeof(a));
    UenrResult r = infer_setup(&c, model_path, &m, &g, &a, &tok, &rng);
    if (r != UENR_OK) {
        uenr_tok_free(tok);
        uenr_acts_free(&a);
        uenr_gemm_destroy(&g);
        uenr_model_free(&m);
        return r;
    }
    printf("UENR chat. Type a line (empty line exits).\n");
    char *line = NULL;
    size_t cap = 0;
    for (;;) {
        printf("user> ");
        fflush(stdout);
        ssize_t n = getline(&line, &cap, stdin);
        if (n <= 0) break;
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0) break;
        printf("assistant> ");
        fflush(stdout);
        infer_one(&c, model_path, &m, &g, &a, tok, &rng, c.infer.system, line, max_new_tokens);
    }
    free(line);
    uenr_tok_free(tok);
    uenr_acts_free(&a);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
    return UENR_OK;
}
