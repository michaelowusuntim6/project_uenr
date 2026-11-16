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
typedef struct { int active, started, in_str, esc, depth, complete; } JsonState;

/* Incremental structural JSON check: balanced containers, valid strings,
 * no trailing non-space after the outer value closes. */
static int json_feed(JsonState *st, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        int ws = (c == ' ' || c == '\n' || c == '\r' || c == '\t');
        if (!st->started) {
            if (ws) continue;
            if (c == '{' || c == '[') { st->started = 1; st->depth = 1; continue; }
            return 0;
        }
        if (st->complete) { if (!ws) return 0; continue; }
        if (st->in_str) {
            if (st->esc) { st->esc = 0; continue; }
            if (c == '\\') { st->esc = 1; continue; }
            if (c == '"') { st->in_str = 0; continue; }
            if (c < 0x20) return 0;
        } else {
            if (c == '"') st->in_str = 1;
            else if (c == '{' || c == '[') st->depth++;
            else if (c == '}' || c == ']') {
                if (--st->depth == 0) st->complete = 1;
            } else if (c == ':' || c == ',' || c == '-' || c == '+' || c == '.' ||
                       (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                       (c >= 'A' && c <= 'Z') || ws) {
                /* scalar bytes */
            } else {
                return 0;
            }
        }
    }
    return 1;
}

int uenr_json_valid(const char *s, size_t n) {
    JsonState st = {0};
    if (!json_feed(&st, s, n)) return 0;
    return st.started && st.complete && !st.in_str && !st.esc && st.depth == 0;
}

static void apply_penalties(float *work, int V, const UenrConfig *cfg,
                            const int *hist, int nhist, unsigned char *seen) {
    float rp = cfg->infer.repetition_penalty;
    int win = cfg->infer.repetition_last_n;
    int start = (win > 0 && nhist > win) ? nhist - win : 0;
    if (rp > 0.0f && rp != 1.0f) {
        memset(seen, 0, (size_t)V);
        for (int i = start; i < nhist; i++) {
            int id = hist[i];
            if (id < 0 || id >= V || seen[id]) continue;
            seen[id] = 1;
            work[id] = (work[id] > 0.0f) ? work[id] / rp : work[id] * rp;
        }
    }
    int ng = cfg->infer.no_repeat_ngram;
    if (ng >= 2 && nhist >= ng) {
        int suf = nhist - (ng - 1);
        for (int i = 0; i + ng <= nhist; i++) {
            if (memcmp(hist + i, hist + suf, (size_t)(ng - 1) * sizeof(int)) == 0) {
                int b = hist[i + ng - 1];
                if (b >= 0 && b < V) work[b] = -3.402823466e+38f;
            }
        }
    }
}

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
    int K = cfg->infer.top_k;

    /* Fused path: select top-K logits first, then softmax/sort only K values.
     * Avoids V exp() calls and a V-wide probability buffer. */
    if (K > 0 && K < V) {
        memset(used, 0, (size_t)V);
        int m = 0;
        for (int k = 0; k < K; k++) {
            int best = -1;
            float bv = -3.402823466e+38f;
            for (int i = 0; i < V; i++)
                if (!used[i] && logits[i] > bv) { bv = logits[i]; best = i; }
            if (best < 0) break;
            used[best] = 1;
            pairs[m].id = best;
            pairs[m].p = logits[best] / temp;
            m++;
        }
        float mx = pairs[0].p;
        for (int i = 1; i < m; i++) if (pairs[i].p > mx) mx = pairs[i].p;
        float sum = 0.0f;
        for (int i = 0; i < m; i++) { pairs[i].p = uenr_expf(pairs[i].p - mx); sum += pairs[i].p; }
        float inv = uenr_rcpf(sum);
        for (int i = 0; i < m; i++) pairs[i].p *= inv;
        if (cfg->infer.min_p > 0.0f) {
            float maxp = 0.0f;
            for (int i = 0; i < m; i++) if (pairs[i].p > maxp) maxp = pairs[i].p;
            float cut = cfg->infer.min_p * maxp, keep = 0.0f;
            for (int i = 0; i < m; i++) {
                if (pairs[i].p < cut) pairs[i].p = 0.0f;
                keep += pairs[i].p;
            }
            if (keep > 0.0f) {
                float ki = uenr_rcpf(keep);
                for (int i = 0; i < m; i++) pairs[i].p *= ki;
            }
        }

        float top_p = cfg->infer.top_p;
        if (top_p > 0.0f && top_p < 1.0f) {
            qsort(pairs, (size_t)m, sizeof(ProbId), cmp_desc);
            float c = 0.0f;
            int cut = m;
            for (int i = 0; i < m; i++) { c += pairs[i].p; if (c >= top_p) { cut = i + 1; break; } }
            float keep = 0.0f;
            for (int i = 0; i < cut; i++) keep += pairs[i].p;
            if (keep > 0.0f) {
                float ki = uenr_rcpf(keep);
                for (int i = 0; i < cut; i++) pairs[i].p *= ki;
            }
            m = cut;
        }
        float r = uenr_rng_f32(rng);
        float c = 0.0f;
        for (int i = 0; i < m; i++) { c += pairs[i].p; if (r <= c) return pairs[i].id; }
        return pairs[m - 1].id;
    }

    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs[i] = uenr_expf((logits[i] - mx) / temp);
        sum += probs[i];
    }
    float inv = uenr_rcpf(sum);
    for (int i = 0; i < V; i++) probs[i] *= inv;

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

    if (cfg->infer.min_p > 0.0f) {
        float maxp = 0.0f;
        for (int i = 0; i < V; i++) if (probs[i] > maxp) maxp = probs[i];
        float cut = cfg->infer.min_p * maxp, keep = 0.0f;
        for (int i = 0; i < V; i++) {
            if (probs[i] < cut) probs[i] = 0.0f;
            keep += probs[i];
        }
        if (keep > 0.0f) {
            float ki = uenr_rcpf(keep);
            for (int i = 0; i < V; i++) probs[i] *= ki;
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

/* Penalize logits, then sample; in JSON mode reject tokens that break the
 * structural grammar and force valid structural bytes as a last resort. */
static int pick_next(const float *logits, float *work, int V, const UenrConfig *cfg,
                     UenrRng *rng, float *probs, ProbId *pairs, unsigned char *used,
                     unsigned char *seen, const int *hist, int nhist,
                     UenrTokenizer *tok, JsonState *js) {
    memcpy(work, logits, (size_t)V * sizeof(float));
    apply_penalties(work, V, cfg, hist, nhist, seen);
    if (!cfg->infer.json_constrained || !js->active)
        return sample_next(work, V, cfg, rng, probs, pairs, used);
    if (js->complete) return cfg->tok.special_end;
    for (int attempt = 0; attempt < 24; attempt++) {
        int id = sample_next(work, V, cfg, rng, probs, pairs, used);
        if (id == cfg->tok.eos_id || id == cfg->tok.special_end) {
            work[id] = -3.402823466e+38f;
            continue;
        }
        char *txt = NULL; size_t len = 0;
        int ok = 0;
        if (uenr_tok_decode(tok, &id, 1, &txt, &len) == UENR_OK) {
            JsonState cand = *js;
            ok = json_feed(&cand, txt, len);
            if (ok) *js = cand;
            free(txt);
        }
        if (ok) return id;
        work[id] = -3.402823466e+38f;
    }
    int b = js->in_str ? '"' : (!js->started ? '{' : '}');
    JsonState cand = *js;
    char ch = (char)b;
    if (json_feed(&cand, &ch, 1)) { *js = cand; return UENR_BYTE_BASE + b; }
    return cfg->tok.special_end;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Longest suffix match in the history; propose up to k following tokens. */
static int lookup_props(const int *ids, int n, int ng, int k, int *out) {
    int maxlen = ng < n ? ng : n;
    for (int len = maxlen; len >= 2; len--) {
        int suf = n - len;
        for (int s = suf - 1; s >= 0; s--) {
            if (memcmp(ids + s, ids + suf, (size_t)len * sizeof(int)) != 0) continue;
            int cnt = 0;
            for (int j = 0; j < k && s + len + j < n; j++)
                out[cnt++] = ids[s + len + j];
            if (cnt > 0) return cnt;
        }
    }
    return 0;
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
    unsigned char *seen = (unsigned char *)malloc((size_t)V);
    float *work = (float *)malloc((size_t)V * sizeof(float));
    int use_kv = cfg->infer.kv_cache;
    float *step_logits = use_kv ? (float *)malloc((size_t)V * sizeof(float)) : NULL;
    int draft_wanted = use_kv && cfg->infer.draft_model[0] && !m->q8_on;
    int spec_on = use_kv && cfg->infer.speculative && !m->q8_on;
    int span_k = cfg->infer.draft_k > cfg->infer.spec_k ? cfg->infer.draft_k : cfg->infer.spec_k;
    float *span_logits = (spec_on || draft_wanted)
        ? (float *)malloc((size_t)span_k * V * sizeof(float)) : NULL;
    int spec_prop = 0, spec_acc = 0;
    UenrModel dm; UenrGemm dg; UenrKVCache dcache;
    memset(&dm, 0, sizeof(dm)); memset(&dcache, 0, sizeof(dcache));
    int draft_on = 0;
    float *dlogits = NULL;
    UenrKVCache cache;
    memset(&cache, 0, sizeof(cache));
    if (!probs || !pairs || !used || !seen || !work || (use_kv && !step_logits) ||
        ((spec_on || draft_wanted) && !span_logits)) {
        free(ids); free(probs); free(pairs); free(used); free(seen); free(work);
    free(span_logits);
        free(step_logits);
        return UENR_ERR_ALLOC;
    }
    if (use_kv && uenr_kv_cache_init(&cache, cfg, max_seq) != UENR_OK) {
        use_kv = 0;
        free(step_logits);
        step_logits = NULL;
        free(span_logits);
        span_logits = NULL;
        spec_on = 0;
    }
    if (spec_on && cache.int8) spec_on = 0;
    if (draft_wanted && !cache.int8 && !cache.fp16 && !cache.q4 && !cache.q2 && !cache.lr) {
        UenrConfig dcfg;
        UenrResult drc = uenr_config_load(cfg->infer.draft_config, &dcfg);
        UenrResult fre = uenr_file_exists(cfg->infer.draft_model);
        if (drc == UENR_OK && fre == UENR_OK) {
            dcfg.model.max_seq = max_seq;
            dcfg.infer.kv_int8 = dcfg.infer.kv_fp16 = dcfg.infer.kv_q4 = 0;
            dcfg.infer.kv_q2 = 0; dcfg.infer.kv_lowrank = 0; dcfg.infer.kv_evict = 0;
            dlogits = malloc((size_t)V * sizeof(float));
            UenrResult ri = uenr_model_init(&dm, &dcfg, rng);
            UenrResult rl = ri == UENR_OK ? uenr_model_load(&dm, cfg->infer.draft_model, NULL, 0) : ri;
            UenrResult rg = rl == UENR_OK ? uenr_gemm_init(&dg, NULL) : rl;
            UenrResult rc = rg == UENR_OK ? uenr_kv_cache_init(&dcache, &dcfg, max_seq) : rg;
            if (dlogits && ri == UENR_OK && rl == UENR_OK && rg == UENR_OK && rc == UENR_OK) {
                int okd = 1;
                for (int pos = 0; pos < n; pos++)
                    if (uenr_model_forward_step(&dm, &dg, &dcache, ids[pos], pos, dlogits) != UENR_OK) { okd = 0; break; }
                draft_on = okd;
            }
        }
    }
    if (draft_on) spec_on = 0;
    if (!use_kv && a->T < max_seq) {
        uenr_acts_free(a);
        if (uenr_acts_init(a, cfg, max_seq, max_seq) != UENR_OK) {
            free(ids); free(probs); free(pairs); free(used); free(seen); free(work);
    free(span_logits);
            return UENR_ERR_ALLOC;
        }
    }

    double t0 = now_s();
    int generated = 0;
    JsonState js = {0};
    if (use_kv) {
        for (int i = 0; i < n; i++)
            uenr_model_forward_step(m, g, &cache, ids[i], i, step_logits);
        for (int step = 0; step < max_new_tokens && n < max_seq; step++) {
            int done = 0, emit_end = 0;
            if (draft_on) {
                int base = n;
                int kk = cfg->infer.draft_k;
                if (kk > max_seq - n) kk = max_seq - n;
                int props[8];
                for (int i = 0; i < kk; i++) {
                    int best = 0;
                    for (int v = 1; v < V; v++) if (dlogits[v] > dlogits[best]) best = v;
                    props[i] = best;
                    if (uenr_model_forward_step(&dm, &dg, &dcache, best, base + i, dlogits) != UENR_OK) {
                        kk = i; break;
                    }
                }
                if (kk > 0 && uenr_model_forward_span(m, g, &cache, props, kk, base,
                                                      span_logits) == UENR_OK) {
                    int a = 0, mismatch = 0;
                    const float *cur = step_logits;
                    for (int i = 0; i < kk; i++) {
                        if (generated >= max_new_tokens) { done = 1; break; }
                        int cand = pick_next(cur, work, V, cfg, rng, probs, pairs,
                                             used, seen, ids, n, tok, &js);
                        ids[n++] = cand;
                        generated++;
                        spec_prop++;
                        if (cand == props[i]) { a++; spec_acc++; } else mismatch = 1;
                        if (cfg->infer.json_constrained && cand == cfg->tok.special_tool_call) {
                            memset(&js, 0, sizeof(js));
                            js.active = 1;
                        }
                        if (cand == cfg->tok.eos_id || cand == cfg->tok.special_end) {
                            done = 1; emit_end = 1; break;
                        }
                        if (n >= max_seq) { done = 1; break; }
                        if (mismatch) {
                            uenr_model_forward_step(m, g, &cache, cand, n - 1, step_logits);
                            uenr_model_forward_step(&dm, &dg, &dcache, cand, n - 1, dlogits);
                            done = 1; break;
                        }
                        cur = span_logits + (size_t)i * V;
                    }
                    if (!done && generated < max_new_tokens) {
                        int extra = pick_next(span_logits + (size_t)(kk - 1) * V,
                                              work, V, cfg, rng, probs, pairs,
                                              used, seen, ids, n, tok, &js);
                        ids[n++] = extra;
                        generated++;
                        if (cfg->infer.json_constrained && extra == cfg->tok.special_tool_call) {
                            memset(&js, 0, sizeof(js));
                            js.active = 1;
                        }
                        if (extra != cfg->tok.eos_id && extra != cfg->tok.special_end &&
                            n < max_seq) {
                            uenr_model_forward_step(m, g, &cache, extra, n - 1, step_logits);
                            uenr_model_forward_step(&dm, &dg, &dcache, extra, n - 1, dlogits);
                        } else if (extra == cfg->tok.eos_id || extra == cfg->tok.special_end) {
                            emit_end = 1;
                        }
                        done = 1;
                    }
                    if (done) { if (emit_end) break; continue; }
                }
            }
            if (spec_on) {
                int props[8];
                int np = lookup_props(ids, n, cfg->infer.spec_ngram,
                                      cfg->infer.spec_k, props);
                int kk = np;
                if (kk > cfg->infer.spec_k) kk = cfg->infer.spec_k;
                if (kk > max_seq - n) kk = max_seq - n;
                if (kk > 0 &&
                    uenr_model_forward_span(m, g, &cache, props, kk, n,
                                            span_logits) == UENR_OK) {
                    int a = 0, mismatch = 0;
                    const float *cur = step_logits;
                    for (int i = 0; i < kk; i++) {
                        if (generated >= max_new_tokens) { done = 1; break; }
                        int cand = pick_next(cur, work, V, cfg, rng, probs, pairs,
                                             used, seen, ids, n, tok, &js);
                        ids[n++] = cand;
                        generated++;
                        spec_prop++;
                        if (cand == props[i]) a++; else mismatch = 1;
                        if (cand == props[i]) spec_acc++;
                        if (cfg->infer.json_constrained && cand == cfg->tok.special_tool_call) {
                            memset(&js, 0, sizeof(js));
                            js.active = 1;
                        }
                        if (cand == cfg->tok.eos_id || cand == cfg->tok.special_end) {
                            done = 1; emit_end = 1; break;
                        }
                        if (n >= max_seq) { done = 1; break; }
                        if (mismatch) {
                            uenr_model_forward_step(m, g, &cache, cand, n - 1, step_logits);
                            done = 1; break;
                        }
                        cur = span_logits + (size_t)i * V;
                    }
                    if (!done && generated < max_new_tokens) {
                        /* All k proposals accepted: take one bonus token. */
                        int extra = pick_next(span_logits + (size_t)(kk - 1) * V,
                                              work, V, cfg, rng, probs, pairs,
                                              used, seen, ids, n, tok, &js);
                        ids[n++] = extra;
                        generated++;
                        if (cfg->infer.json_constrained && extra == cfg->tok.special_tool_call) {
                            memset(&js, 0, sizeof(js));
                            js.active = 1;
                        }
                        if (extra != cfg->tok.eos_id && extra != cfg->tok.special_end &&
                            n < max_seq)
                            uenr_model_forward_step(m, g, &cache, extra, n - 1, step_logits);
                        else if (extra == cfg->tok.eos_id || extra == cfg->tok.special_end)
                            emit_end = 1;
                        done = 1;
                    }
                    if (done) { if (emit_end) break; continue; }
                }
            }
            int next = pick_next(step_logits, work, V, cfg, rng, probs, pairs,
                                 used, seen, ids, n, tok, &js);
            ids[n++] = next;
            generated++;
            if (next == cfg->tok.eos_id || next == cfg->tok.special_end) break;
            if (cfg->infer.json_constrained && next == cfg->tok.special_tool_call) {
                memset(&js, 0, sizeof(js));
                js.active = 1;
            }
            if (n >= max_seq) break;
            uenr_model_forward_step(m, g, &cache, next, n - 1, step_logits);
            if (draft_on) uenr_model_forward_step(&dm, &dg, &dcache, next, n - 1, dlogits);
        }
    } else {
        for (int step = 0; step < max_new_tokens && n < max_seq; step++) {
            int count = 0;
            uenr_model_forward(m, g, a, ids, n, n, NULL, cfg->tok.ignore_index, 0, &count);
            const float *logits = a->logits + (size_t)(n - 1) * V;
            int next = pick_next(logits, work, V, cfg, rng, probs, pairs,
                                 used, seen, ids, n, tok, &js);
            ids[n++] = next;
            generated++;
            if (next == cfg->tok.eos_id || next == cfg->tok.special_end) break;
            if (cfg->infer.json_constrained && next == cfg->tok.special_tool_call) {
                memset(&js, 0, sizeof(js));
                js.active = 1;
            }
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
    {
        char extra[96];
        extra[0] = '\0';
        if (cfg->infer.json_constrained && js.active)
            snprintf(extra, sizeof(extra), ", tool JSON %s",
                     js.complete ? "balanced" : "incomplete");
        if (spec_prop > 0) {
            size_t el = strlen(extra);
            snprintf(extra + el, sizeof(extra) - el, ", spec %d/%d",
                     spec_acc, spec_prop);
        }
        printf("[%d tokens, %.1f tok/s%s]\n", generated, (double)generated / dt, extra);
    }

    uenr_kv_cache_free(&cache);
    if (draft_on || dm.params) {
        uenr_kv_cache_free(&dcache);
        uenr_gemm_destroy(&dg);
        uenr_model_free(&dm);
    }
    free(dlogits);
    free(step_logits);
    free(ids); free(probs); free(pairs); free(used); free(seen); free(work);
    free(span_logits);
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
    if (cfg->infer.weight_int4) uenr_model_quantize_int4(m);
    else if (cfg->infer.weight_int8) uenr_model_quantize_int8(m);
    if (uenr_gemm_init(g, NULL) != UENR_OK) return UENR_ERR_ALLOC;
    {
        int at = cfg->infer.kv_cache ? 1 : cfg->model.max_seq;
        if (uenr_acts_init(a, cfg, at, at) != UENR_OK) return UENR_ERR_ALLOC;
    }
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

/* ------------------------------------------------------------------------- */
/* batched prompt-file inference                                             */
/* ------------------------------------------------------------------------- */

UenrResult uenr_infer_batch(const UenrConfig *cfg, const char *model_path,
                            const char *prompt_file, int max_new_tokens, int greedy) {
    if (!cfg || !prompt_file) return UENR_ERR_INVALID;
    UenrConfig c = *cfg;
    if (greedy) c.infer.greedy = 1;
    if (max_new_tokens <= 0) max_new_tokens = c.infer.max_new_tokens;
    const int maxseq = c.model.max_seq;
    const int V = c.model.vocab_size;
    int cap = c.infer.batch_max > 0 ? c.infer.batch_max : 8;

    FILE *f = fopen(prompt_file, "r");
    if (!f) { fprintf(stderr, "batch: cannot open %s\n", prompt_file); return UENR_ERR_IO; }
    char **prompts = calloc((size_t)cap, sizeof(char *));
    int B = 0;
    char *line = NULL;
    size_t lcap = 0;
    ssize_t ln;
    while (B < cap && (ln = getline(&line, &lcap, f)) >= 0) {
        while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
        prompts[B] = strdup(line);
        if (!prompts[B]) break;
        B++;
    }
    free(line);
    fclose(f);
    if (B == 0) { free(prompts); return UENR_ERR_INVALID; }

    UenrModel m; UenrGemm g; UenrActs a; UenrTokenizer *tok = NULL; UenrRng rng;
    memset(&m, 0, sizeof(m));
    UenrResult sr = infer_setup(&c, model_path, &m, &g, &a, &tok, &rng);
    if (sr != UENR_OK) {
        for (int b = 0; b < B; b++) free(prompts[b]);
        free(prompts);
        return sr;
    }

    int **ids = calloc((size_t)B, sizeof(int *));
    int *n = calloc((size_t)B, sizeof(int));
    int *plen = calloc((size_t)B, sizeof(int));
    int *done = calloc((size_t)B, sizeof(int));
    int *gen = calloc((size_t)B, sizeof(int));
    int *has_pending = calloc((size_t)B, sizeof(int));
    int *pending = calloc((size_t)B, sizeof(int));
    int *poses = calloc((size_t)B, sizeof(int));
    JsonState *js = calloc((size_t)B, sizeof(JsonState));
    UenrKVCache *caches = calloc((size_t)B, sizeof(UenrKVCache));
    UenrKVCache **cptr = calloc((size_t)B, sizeof(UenrKVCache *));
    float *logits = malloc((size_t)B * V * sizeof(float));
    float *mlog = malloc((size_t)B * V * sizeof(float));
    int *mtok = malloc((size_t)B * sizeof(int));
    int *map = malloc((size_t)B * sizeof(int));
    float *probs = malloc((size_t)V * sizeof(float));
    ProbId *pairs = malloc((size_t)V * sizeof(ProbId));
    unsigned char *used = malloc((size_t)V), *seen = malloc((size_t)V);
    float *work = malloc((size_t)V * sizeof(float));
    float *plog = malloc((size_t)V * sizeof(float));
    int ok = ids && n && plen && done && gen && has_pending && pending && poses &&
             js && caches && cptr && logits && mlog && mtok && map && probs && pairs &&
             used && seen && work && plog;
    if (!ok) { sr = UENR_ERR_ALLOC; goto out; }

    int lcp = 0;
    for (int b = 0; b < B; b++) {
        int *p = NULL; int pn = 0;
        if (uenr_chat_infer(tok, c.infer.system, prompts[b], &p, &pn) != UENR_OK) {
            sr = UENR_ERR_INVALID; goto out;
        }
        ids[b] = malloc((size_t)maxseq * sizeof(int));
        if (!ids[b]) { free(p); sr = UENR_ERR_ALLOC; goto out; }
        if (pn > maxseq) { memmove(p, p + (pn - maxseq), (size_t)maxseq * sizeof(int)); pn = maxseq; }
        memcpy(ids[b], p, (size_t)pn * sizeof(int));
        free(p);
        n[b] = pn; plen[b] = pn;
        if (uenr_kv_cache_init(&caches[b], &c, maxseq) != UENR_OK) {
            sr = UENR_ERR_ALLOC; goto out;
        }
        cptr[b] = &caches[b];
        if (b == 0) lcp = pn;
        else {
            int m = lcp < pn ? lcp : pn, k = 0;
            while (k < m && ids[b][k] == ids[0][k]) k++;
            lcp = k;
        }
    }

    double t0 = now_s();
    int prefix_ok = c.infer.prefix_cache && lcp > 0;
    if (prefix_ok) {
        for (int pos = 0; pos < lcp; pos++)
            uenr_model_forward_step(&m, &g, &caches[0], ids[0][pos], pos, plog);
        for (int b = 1; b < B && prefix_ok; b++)
            prefix_ok = uenr_kv_cache_copy_prefix(&caches[b], &caches[0], lcp);
    }
    if (!prefix_ok) lcp = 0;

    for (int b = 0; b < B; b++) {
        int start = lcp;
        if (n[b] == start && prefix_ok) {
            memcpy(logits + (size_t)b * V, plog, (size_t)V * sizeof(float));
        } else {
            for (int pos = start; pos < n[b]; pos++)
                uenr_model_forward_step(&m, &g, &caches[b], ids[b][pos], pos,
                                        logits + (size_t)b * V);
        }
    }

    int total_gen = 0;
    for (;;) {
        int np = 0;
        for (int b = 0; b < B; b++) {
            if (done[b]) continue;
            if (n[b] >= maxseq) { done[b] = 1; continue; }
            int next = pick_next(logits + (size_t)b * V, work, V, &c, &rng,
                                 probs, pairs, used, seen, ids[b], n[b], tok, &js[b]);
            ids[b][n[b]++] = next;
            gen[b]++; total_gen++;
            if (cfg->infer.json_constrained && next == cfg->tok.special_tool_call) {
                memset(&js[b], 0, sizeof(js[b]));
                js[b].active = 1;
            }
            if (next == cfg->tok.eos_id || next == cfg->tok.special_end ||
                gen[b] >= max_new_tokens || n[b] >= maxseq) {
                done[b] = 1;
                continue;
            }
            pending[b] = next;
            poses[b] = n[b] - 1;
            has_pending[b] = 1;
            mtok[np] = next;
            cptr[np] = &caches[b];
            map[np] = b;
            np++;
        }
        if (np == 0) break;
        int *pp = malloc((size_t)np * sizeof(int));
        if (!pp) { sr = UENR_ERR_ALLOC; goto out; }
        for (int i = 0; i < np; i++) pp[i] = poses[map[i]];
        for (int b = 0; b < B; b++) has_pending[b] = 0;
        sr = uenr_model_forward_multi(&m, &g, cptr, np, mtok, pp, mlog);
        free(pp);
        if (sr != UENR_OK) goto out;
        for (int i = 0; i < np; i++)
            memcpy(logits + (size_t)map[i] * V, mlog + (size_t)i * V,
                   (size_t)V * sizeof(float));
    }
    double dt = now_s() - t0;
    if (dt <= 0) dt = 1e-9;

    for (int b = 0; b < B; b++) {
        int out_n = n[b] - plen[b];
        if (out_n > 0 && (ids[b][n[b] - 1] == c.tok.eos_id ||
                          ids[b][n[b] - 1] == c.tok.special_end)) out_n--;
        char *text = NULL;
        if (out_n > 0) uenr_tok_decode(tok, ids[b] + plen[b], out_n, &text, NULL);
        printf("[%d] %s\n", b, text ? text : "");
        free(text);
        if (cfg->infer.json_constrained && js[b].active)
            printf("    tool JSON %s\n", js[b].complete ? "balanced" : "incomplete");
    }
    printf("[batch %d prompts, %d tokens, %.1f tok/s]\n", B, total_gen,
           (double)total_gen / dt);
    sr = UENR_OK;

out:
    for (int b = 0; b < B; b++) {
        if (ids && ids[b]) free(ids[b]);
        if (caches) uenr_kv_cache_free(&caches[b]);
        if (prompts[b]) free(prompts[b]);
    }
    free(prompts); free(ids); free(n); free(plen); free(done); free(gen);
    free(has_pending); free(pending); free(poses); free(js); free(caches);
    free(cptr); free(logits); free(mlog); free(mtok); free(map); free(probs); free(pairs);
    free(used); free(seen); free(work); free(plog);
    UenrActs *ap = &a; uenr_acts_free(ap);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
    return sr;
}

/* ------------------------------------------------------------------------- */
/* continuous batching: stream queued prompts into freed slots               */
/* ------------------------------------------------------------------------- */

UenrResult uenr_infer_continuous(const UenrConfig *cfg, const char *model_path,
                                 const char *prompt_file, int max_new_tokens,
                                 int greedy) {
    if (!cfg || !prompt_file) return UENR_ERR_INVALID;
    UenrConfig c = *cfg;
    if (greedy) c.infer.greedy = 1;
    if (max_new_tokens <= 0) max_new_tokens = c.infer.max_new_tokens;
    const int maxseq = c.model.max_seq, V = c.model.vocab_size;
    int cap = c.infer.batch_max > 0 ? c.infer.batch_max : 8;

    FILE *f = fopen(prompt_file, "r");
    if (!f) return UENR_ERR_IO;
    char **q = NULL;
    int P = 0, qcap = 0;
    char *line = NULL; size_t lcap = 0; ssize_t ln;
    while ((ln = getline(&line, &lcap, f)) >= 0) {
        while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
        if (P == qcap) {
            int nc = qcap ? qcap * 2 : 16;
            char **nq = realloc(q, (size_t)nc * sizeof(char *));
            if (!nq) { free(line); break; }
            q = nq; qcap = nc;
        }
        q[P] = strdup(line);
        if (!q[P]) break;
        P++;
    }
    free(line); fclose(f);
    if (P == 0) { free(q); return UENR_ERR_INVALID; }

    UenrModel m; UenrGemm g; UenrActs a; UenrTokenizer *tok = NULL; UenrRng rng;
    memset(&m, 0, sizeof(m));
    UenrResult sr = infer_setup(&c, model_path, &m, &g, &a, &tok, &rng);
    if (sr != UENR_OK) goto done;

    int **ids = calloc((size_t)cap, sizeof(int *));
    int *n = calloc((size_t)cap, sizeof(int));
    int *plen = calloc((size_t)cap, sizeof(int));
    int *gen = calloc((size_t)cap, sizeof(int));
    int *active = calloc((size_t)cap, sizeof(int));
    int *done = calloc((size_t)cap, sizeof(int));
    int *pidx = calloc((size_t)cap, sizeof(int));
    JsonState *js = calloc((size_t)cap, sizeof(JsonState));
    UenrKVCache *caches = calloc((size_t)cap, sizeof(UenrKVCache));
    UenrKVCache **cptr = calloc((size_t)cap, sizeof(UenrKVCache *));
    float *logits = malloc((size_t)cap * V * sizeof(float));
    float *mlog = malloc((size_t)cap * V * sizeof(float));
    int *mtok = malloc((size_t)cap * sizeof(int));
    int *map = malloc((size_t)cap * sizeof(int));
    int *poses = calloc((size_t)cap, sizeof(int));
    float *probs = malloc((size_t)V * sizeof(float));
    ProbId *pairs = malloc((size_t)V * sizeof(ProbId));
    unsigned char *used = malloc((size_t)V), *seen = malloc((size_t)V);
    float *work = malloc((size_t)V * sizeof(float));
    float *plog = malloc((size_t)V * sizeof(float));
    int **all_ids = calloc((size_t)P, sizeof(int *));
    int *all_n = calloc((size_t)P, sizeof(int));
    if (!ids || !n || !plen || !gen || !active || !done || !pidx || !js ||
        !caches || !cptr || !logits || !mlog || !mtok || !map || !poses ||
        !probs || !pairs || !used || !seen || !work || !plog || !all_ids || !all_n) {
        sr = UENR_ERR_ALLOC; goto out2;
    }
    for (int i = 0; i < P; i++) {
        int *p = NULL, pn = 0;
        if (uenr_chat_infer(tok, c.infer.system, q[i], &p, &pn) != UENR_OK) continue;
        if (pn > maxseq) { memmove(p, p + (pn - maxseq), (size_t)maxseq * sizeof(int)); pn = maxseq; }
        all_ids[i] = p; all_n[i] = pn;
    }
    int lcp = 0;
    for (int i = 0; i < P; i++) {
        if (!all_ids[i]) continue;
        if (lcp == 0) lcp = all_n[i];
        int m = lcp < all_n[i] ? lcp : all_n[i], k = 0;
        while (k < m && all_ids[i][k] == all_ids[0][k]) k++;
        lcp = k;
    }
    UenrKVCache prefix;
    memset(&prefix, 0, sizeof(prefix));
    int prefix_ok = 0;
    if (c.infer.prefix_cache && lcp > 0 && all_ids[0]) {
        if (uenr_kv_cache_init(&prefix, &c, maxseq) == UENR_OK) {
            prefix_ok = 1;
            for (int pos = 0; pos < lcp; pos++)
                uenr_model_forward_step(&m, &g, &prefix, all_ids[0][pos], pos, plog);
        }
    }
    int plen_lcp = lcp;

    int qnext = 0, alive = 0;
    int start_slot(int s) {
        if (qnext >= P) return 0;
        int idx = qnext++;
        int pn = all_n[idx];
        if (!all_ids[idx] || pn <= 0) return 0;
        if (!ids[s]) ids[s] = malloc((size_t)maxseq * sizeof(int));
        if (!ids[s]) return 0;
        memcpy(ids[s], all_ids[idx], (size_t)pn * sizeof(int));
        n[s] = pn; plen[s] = pn; gen[s] = 0; done[s] = 0; active[s] = 1;
        pidx[s] = idx; memset(&js[s], 0, sizeof(js[s]));
        uenr_kv_cache_free(&caches[s]);
        if (uenr_kv_cache_init(&caches[s], &c, maxseq) != UENR_OK) { active[s] = 0; return 0; }
        int st = 0;
        if (prefix_ok && plen_lcp > 0 &&
            uenr_kv_cache_copy_prefix(&caches[s], &prefix, plen_lcp))
            st = plen_lcp;
        if (st == pn) {
            memcpy(logits + (size_t)s * V, plog, (size_t)V * sizeof(float));
        } else {
            for (int pos = st; pos < pn; pos++)
                uenr_model_forward_step(&m, &g, &caches[s], ids[s][pos], pos,
                                        logits + (size_t)s * V);
        }
        return 1;
    }

    double t0 = now_s();
    int total_gen = 0;
    for (int s = 0; s < cap; s++) if (start_slot(s)) alive++;
    while (qnext < P || alive > 0) {
        int np = 0;
        for (int s = 0; s < cap; s++) {
            if (!active[s] || done[s]) continue;
            if (n[s] >= maxseq || gen[s] >= max_new_tokens) { done[s] = 1; continue; }
            int next = pick_next(logits + (size_t)s * V, work, V, &c, &rng,
                                 probs, pairs, used, seen, ids[s], n[s], tok, &js[s]);
            ids[s][n[s]++] = next;
            gen[s]++; total_gen++;
            if (c.infer.json_constrained && next == c.tok.special_tool_call) {
                memset(&js[s], 0, sizeof(js[s])); js[s].active = 1;
            }
            if (next == c.tok.eos_id || next == c.tok.special_end ||
                gen[s] >= max_new_tokens || n[s] >= maxseq) { done[s] = 1; continue; }
            mtok[np] = next; cptr[np] = &caches[s]; poses[np] = n[s] - 1;
            map[np] = s; np++;
        }
        if (np > 0) {
            sr = uenr_model_forward_multi(&m, &g, cptr, np, mtok, poses, mlog);
            if (sr == UENR_OK) {
                for (int i = 0; i < np; i++)
                    memcpy(logits + (size_t)map[i] * V, mlog + (size_t)i * V,
                           (size_t)V * sizeof(float));
            } else {
                for (int i = 0; i < np; i++)
                    uenr_model_forward_step(&m, &g, cptr[i], mtok[i], poses[i],
                                            logits + (size_t)map[i] * V);
                sr = UENR_OK;
            }
        }
        for (int s = 0; s < cap; s++) {
            if (!active[s] || !done[s]) continue;
            int out_n = n[s] - plen[s];
            if (out_n > 0 && (ids[s][n[s] - 1] == c.tok.eos_id ||
                              ids[s][n[s] - 1] == c.tok.special_end)) out_n--;
            char *text = NULL;
            if (out_n > 0) uenr_tok_decode(tok, ids[s] + plen[s], out_n, &text, NULL);
            printf("[%d] %s\n", pidx[s], text ? text : "");
            free(text);
            active[s] = 0; done[s] = 0;
            if (qnext < P) { if (start_slot(s)) continue; }
            alive--;
        }
        if (np == 0 && alive == 0) break;
        int any = 0;
        for (int s = 0; s < cap; s++) if (active[s] && !done[s]) any = 1;
        if (!any && qnext >= P) break;
    }
    {
        double dt = now_s() - t0;
        if (dt <= 0) dt = 1e-9;
        printf("[continuous %d prompts, %d slots, %d tokens, %.1f tok/s]\n",
               P, cap, total_gen, (double)total_gen / dt);
    }
    sr = UENR_OK;

out2:
    for (int s = 0; s < cap; s++) {
        if (ids && ids[s]) free(ids[s]);
        if (caches) uenr_kv_cache_free(&caches[s]);
    }
    for (int i = 0; i < P; i++) if (all_ids) free(all_ids[i]);
    free(all_ids); free(all_n);
    uenr_kv_cache_free(&prefix);
    free(ids); free(n); free(plen); free(gen); free(active); free(done); free(pidx);
    free(js); free(caches); free(cptr); free(logits); free(mlog); free(mtok);
    free(map); free(poses); free(probs); free(pairs); free(used); free(seen); free(work); free(plog);
    uenr_acts_free(&a);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
done:
    for (int i = 0; i < P; i++) free(q[i]);
    free(q);
    return sr;
}
