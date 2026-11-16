#include "uenr/dataset.h"
#include "uenr/io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UDAT_MAGIC 0x54414455u /* "UDAT" */

/* ------------------------------------------------------------------------- */
/* token buffer                                                              */
/* ------------------------------------------------------------------------- */

typedef struct {
    uint32_t *v;
    size_t n, cap;
} TokBuf;

static int tb_push(TokBuf *b, uint32_t x) {
    if (b->n == b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 65536;
        uint32_t *p = (uint32_t *)realloc(b->v, cap * sizeof(uint32_t));
        if (!p) return 0;
        b->v = p; b->cap = cap;
    }
    b->v[b->n++] = x;
    return 1;
}

static UenrResult write_token_bin(const char *path, const uint32_t *tok,
                                  size_t count, int vocab) {
    size_t bytes = 7 * sizeof(uint32_t) + count * sizeof(uint16_t);
    unsigned char *blob = (unsigned char *)malloc(bytes);
    if (!blob) return UENR_ERR_ALLOC;
    uint32_t *h = (uint32_t *)blob;
    h[0] = UDAT_MAGIC;
    h[1] = 1;
    h[2] = (uint32_t)(count & 0xffffffffu);
    h[3] = (uint32_t)(count >> 32);
    h[4] = (uint32_t)vocab;
    h[5] = 1; /* dtype: uint16 */
    h[6] = 0;
    uint16_t *dst = (uint16_t *)(blob + 7 * sizeof(uint32_t));
    for (size_t i = 0; i < count; i++) dst[i] = (uint16_t)tok[i];
    UenrResult r = uenr_write_file_atomic(path, blob, bytes);
    free(blob);
    return r;
}

UenrResult uenr_data_load(const char *path, UenrTokenData *out) {
    if (!path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    if (size < 7 * sizeof(uint32_t)) { free(data); return UENR_ERR_INVALID; }
    uint32_t *h = (uint32_t *)data;
    if (h[0] != UDAT_MAGIC || h[1] != 1 || h[5] != 1) { free(data); return UENR_ERR_INVALID; }
    size_t count = (size_t)h[2] | ((size_t)h[3] << 32);
    if (size != 7 * sizeof(uint32_t) + count * sizeof(uint16_t)) {
        free(data);
        return UENR_ERR_INVALID;
    }
    uint16_t *toks = (uint16_t *)malloc(count * sizeof(uint16_t));
    if (!toks) { free(data); return UENR_ERR_ALLOC; }
    memcpy(toks, (unsigned char *)data + 7 * sizeof(uint32_t), count * sizeof(uint16_t));
    out->tokens = toks;
    out->count = count;
    out->vocab = (int)h[4];
    free(data);
    return UENR_OK;
}

void uenr_data_free(UenrTokenData *d) {
    if (!d) return;
    free(d->tokens);
    d->tokens = NULL;
    d->count = 0;
}

int uenr_data_batch(const UenrTokenData *d, UenrRng *rng, int seq,
                    int nseq, int *inputs, int *targets) {
    if (!d || !rng || !inputs || !targets || seq <= 0 || nseq <= 0) return -1;
    if (d->count < (size_t)seq + 1) return -1;
    size_t max_start = d->count - (size_t)seq - 1;
    for (int s = 0; s < nseq; s++) {
        size_t start = (size_t)(uenr_rng_u64(rng) % (max_start + 1));
        const uint16_t *src = d->tokens + start;
        int *in = inputs + (size_t)s * seq;
        int *tg = targets + (size_t)s * seq;
        for (int i = 0; i < seq; i++) {
            in[i] = src[i];
            tg[i] = src[i + 1];
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* prepare                                                                   */
/* ------------------------------------------------------------------------- */

static void json_escape(FILE *f, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (c < 0x20) fprintf(f, "\\u%04x", c);
            else fputc(c, f);
        }
    }
}

static const char *trim_line(char *s, size_t *len) {
    while (*s && isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    *len = n;
    return s;
}

UenrResult uenr_data_prepare(const UenrConfig *cfg, const UenrTokenizer *tok) {
    if (!cfg || !tok) return UENR_ERR_INVALID;

    FILE *f = fopen(cfg->paths.data, "rb");
    if (!f) return UENR_ERR_IO;
    size_t want = (size_t)cfg->prep.max_bytes;
    char *raw = (char *)malloc(want + 1);
    if (!raw) { fclose(f); return UENR_ERR_ALLOC; }
    size_t got = fread(raw, 1, want, f);
    fclose(f);
    raw[got] = '\0';
    fprintf(stderr, "prepare: read %zu bytes\n", got);

    TokBuf tb = {0};
    size_t line_no = 0;
    char *p = raw;
    while (*p) {
        char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char saved = p[len];
        p[len] = '\0';
        size_t tlen = 0;
        const char *line = trim_line(p, &tlen);
        if (tlen > 0) {
            int *ids = NULL, n = 0;
            if (uenr_tok_encode(tok, line, &ids, &n) == UENR_OK && n > 0) {
                if (cfg->tok.add_bos && !tb_push(&tb, (uint32_t)uenr_tok_id(tok, UENR_ROLE_BOS))) goto oom;
                for (int i = 0; i < n; i++) if (!tb_push(&tb, (uint32_t)ids[i])) goto oom;
                if (cfg->tok.add_eos && !tb_push(&tb, (uint32_t)uenr_tok_id(tok, UENR_ROLE_EOS))) goto oom;
            }
            free(ids);
            line_no++;
        }
        p[len] = saved;
        if (!nl) break;
        p = nl + 1;
    }

    fprintf(stderr, "prepare: encoded %zu tokens from %zu lines\n", tb.n, line_no);
    if (tb.n == 0) { free(tb.v); free(raw); return UENR_ERR_INVALID; }

    size_t val_count = (size_t)((double)tb.n * (double)cfg->prep.val_percent / 100.0);
    size_t train_count = tb.n - val_count;
    if (train_count < (size_t)cfg->train.seq_len + 1) {
        val_count = 0;
        train_count = tb.n;
    }

    UenrResult r = write_token_bin(cfg->paths.train_bin, tb.v, train_count, uenr_tok_vocab(tok));
    if (UENR_OK == r)
        r = write_token_bin(cfg->paths.val_bin, tb.v + train_count, val_count, uenr_tok_vocab(tok));

    /* SFT + preference pairs from the first stories. */
    if (UENR_OK == r && cfg->prep.sft_stories > 0) {
        FILE *sf = fopen(cfg->paths.sft_jsonl, "w");
        FILE *pf = fopen(cfg->paths.pairs_jsonl, "w");
        if (sf && pf) {
            char *q = raw;
            int story = 0;
            char *prev_assistant = NULL;
            while (*q && story < cfg->prep.sft_stories) {
                char *nl = strchr(q, '\n');
                size_t len = nl ? (size_t)(nl - q) : strlen(q);
                char saved = q[len];
                q[len] = '\0';
                size_t tlen = 0;
                const char *line = trim_line(q, &tlen);
                if (tlen > (size_t)cfg->prep.sft_prompt_chars + 16) {
                    size_t cut = (size_t)cfg->prep.sft_prompt_chars;
                    if (cut > tlen / 2) cut = tlen / 2;
                    while (cut > 0 && line[cut] != ' ') cut--;
                    if (cut == 0) cut = (size_t)cfg->prep.sft_prompt_chars;
                    char prompt[1024];
                    if (cut >= sizeof(prompt)) cut = sizeof(prompt) - 1;
                    memcpy(prompt, line, cut);
                    prompt[cut] = '\0';
                    const char *assistant = line + cut;
                    while (*assistant == ' ') assistant++;
                    size_t alen = strlen(assistant);
                    if (alen > 600) alen = 600;
                    char *asst = (char *)malloc(alen + 1);
                    if (!asst) { q[len] = saved; break; }
                    memcpy(asst, assistant, alen);
                    asst[alen] = '\0';

                    fprintf(sf, "{\"messages\":[{\"role\":\"user\",\"content\":\"Continue the story: ");
                    json_escape(sf, prompt);
                    fprintf(sf, "\"},{\"role\":\"assistant\",\"content\":\"");
                    json_escape(sf, asst);
                    fprintf(sf, "\"}]}\n");

                    if (prev_assistant) {
                        fprintf(pf, "{\"prompt\":\"Continue the story: ");
                        json_escape(pf, prompt);
                        fprintf(pf, "\",\"chosen\":\"");
                        json_escape(pf, asst);
                        fprintf(pf, "\",\"rejected\":\"");
                        json_escape(pf, prev_assistant);
                        fprintf(pf, "\"}\n");
                        free(prev_assistant);
                    }
                    prev_assistant = asst;
                    story++;
                }
                q[len] = saved;
                if (!nl) break;
                q = nl + 1;
            }
            free(prev_assistant);
            fclose(sf);
            fclose(pf);
        } else {
            if (sf) fclose(sf);
            if (pf) fclose(pf);
        }
    }

    fprintf(stderr, "prepare: %zu lines, %zu tokens (train %zu, val %zu)\n",
            line_no, tb.n, train_count, val_count);
    free(tb.v);
    free(raw);
    return r;

oom:
    free(tb.v);
    free(raw);
    return UENR_ERR_ALLOC;
}

/* ------------------------------------------------------------------------- */
/* tiny JSON reader (for SFT / preference files)                             */
/* ------------------------------------------------------------------------- */

typedef struct {
    const char *p;
    size_t n, i;
} JParser;

static void jws(JParser *j) {
    while (j->i < j->n && isspace((unsigned char)j->p[j->i])) j->i++;
}

static int jhex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int jstring(JParser *j, char **out) {
    jws(j);
    if (j->i >= j->n || j->p[j->i] != '"') return 0;
    j->i++;
    size_t cap = 32, n = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return 0;
    while (j->i < j->n) {
        unsigned char c = (unsigned char)j->p[j->i++];
        if (c == '"') { buf[n] = '\0'; *out = buf; return 1; }
        char tmp[4];
        int tn = 1;
        tmp[0] = (char)c;
        if (c == '\\') {
            if (j->i >= j->n) break;
            char e = j->p[j->i++];
            switch (e) {
            case 'n': tmp[0] = '\n'; break;
            case 'r': tmp[0] = '\r'; break;
            case 't': tmp[0] = '\t'; break;
            case 'b': tmp[0] = '\b'; break;
            case 'f': tmp[0] = '\f'; break;
            case '"': tmp[0] = '"'; break;
            case '\\': tmp[0] = '\\'; break;
            case '/': tmp[0] = '/'; break;
            case 'u': {
                if (j->i + 4 > j->n) { free(buf); return 0; }
                int cp = 0;
                for (int k = 0; k < 4; k++) {
                    int h = jhex(j->p[j->i + k]);
                    if (h < 0) { free(buf); return 0; }
                    cp = cp * 16 + h;
                }
                j->i += 4;
                if (cp < 0x80) { tmp[0] = (char)cp; }
                else if (cp < 0x800) {
                    tmp[0] = (char)(0xC0 | (cp >> 6));
                    tmp[1] = (char)(0x80 | (cp & 0x3F));
                    tn = 2;
                } else {
                    tmp[0] = (char)(0xE0 | (cp >> 12));
                    tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    tmp[2] = (char)(0x80 | (cp & 0x3F));
                    tn = 3;
                }
                break;
            }
            default: tmp[0] = e; break;
            }
        }
        if (n + (size_t)tn + 1 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) { free(buf); return 0; }
            buf = nb;
        }
        for (int k = 0; k < tn; k++) buf[n++] = tmp[k];
    }
    free(buf);
    return 0;
}

static int jskip(JParser *j) {
    jws(j);
    if (j->i >= j->n) return 0;
    char c = j->p[j->i];
    if (c == '"') { char *s = NULL; int r = jstring(j, &s); free(s); return r; }
    if (c == '{' || c == '[') {
        char open = c, close = (c == '{') ? '}' : ']';
        j->i++;
        jws(j);
        if (j->i < j->n && j->p[j->i] == close) { j->i++; return 1; }
        for (;;) {
            if (open == '{') {
                char *k = NULL;
                if (!jstring(j, &k)) return 0;
                free(k);
                jws(j);
                if (j->i >= j->n || j->p[j->i] != ':') return 0;
                j->i++;
            }
            if (!jskip(j)) return 0;
            jws(j);
            if (j->i < j->n && j->p[j->i] == ',') { j->i++; continue; }
            if (j->i < j->n && j->p[j->i] == close) { j->i++; return 1; }
            return 0;
        }
    }
    while (j->i < j->n) {
        char d = j->p[j->i];
        if (d == ',' || d == '}' || d == ']' || isspace((unsigned char)d)) break;
        j->i++;
    }
    return 1;
}

typedef struct { char *role, *content; } Msg;

static void msgs_free(Msg *m, int n) {
    for (int i = 0; i < n; i++) { free(m[i].role); free(m[i].content); }
    free(m);
}

static int parse_messages(const char *line, Msg **out, int *out_n) {
    JParser j = { line, strlen(line), 0 };
    jws(&j);
    if (j.i >= j.n || j.p[j.i] != '{') return 0;
    j.i++;
    Msg *msgs = NULL;
    int nmsg = 0;

    jws(&j);
    if (j.i < j.n && j.p[j.i] == '}') { j.i++; *out = NULL; *out_n = 0; return 1; }

    for (;;) {
        char *key = NULL;
        if (!jstring(&j, &key)) { msgs_free(msgs, nmsg); return 0; }
        jws(&j);
        if (j.i >= j.n || j.p[j.i] != ':') { free(key); msgs_free(msgs, nmsg); return 0; }
        j.i++;
        if (strcmp(key, "messages") == 0) {
            free(key);
            jws(&j);
            if (j.i >= j.n || j.p[j.i] != '[') { msgs_free(msgs, nmsg); return 0; }
            j.i++;
            jws(&j);
            if (j.i < j.n && j.p[j.i] == ']') { j.i++; }
            else {
                for (;;) {
                    jws(&j);
                    if (j.i >= j.n || j.p[j.i] != '{') { msgs_free(msgs, nmsg); return 0; }
                    j.i++;
                    char *role = NULL, *content = NULL;
                    jws(&j);
                    if (j.i < j.n && j.p[j.i] == '}') { j.i++; goto push_msg; }
                    for (;;) {
                        char *mk = NULL;
                        if (!jstring(&j, &mk)) { free(role); free(content); msgs_free(msgs, nmsg); return 0; }
                        jws(&j);
                        if (j.i >= j.n || j.p[j.i] != ':') { free(mk); free(role); free(content); msgs_free(msgs, nmsg); return 0; }
                        j.i++;
                        if (strcmp(mk, "role") == 0) { free(mk); free(role); if (!jstring(&j, &role)) role = NULL; }
                        else if (strcmp(mk, "content") == 0) { free(mk); free(content); if (!jstring(&j, &content)) content = NULL; }
                        else { free(mk); if (!jskip(&j)) { free(role); free(content); msgs_free(msgs, nmsg); return 0; } }
                        jws(&j);
                        if (j.i < j.n && j.p[j.i] == ',') { j.i++; continue; }
                        if (j.i < j.n && j.p[j.i] == '}') { j.i++; break; }
                        free(role); free(content); msgs_free(msgs, nmsg); return 0;
                    }
                push_msg: {
                        Msg *nm = (Msg *)realloc(msgs, (size_t)(nmsg + 1) * sizeof(Msg));
                        if (!nm) { free(role); free(content); msgs_free(msgs, nmsg); return 0; }
                        msgs = nm;
                        msgs[nmsg].role = role ? role : strdup("user");
                        msgs[nmsg].content = content ? content : strdup("");
                        nmsg++;
                    }
                    jws(&j);
                    if (j.i < j.n && j.p[j.i] == ',') { j.i++; continue; }
                    if (j.i < j.n && j.p[j.i] == ']') { j.i++; break; }
                    msgs_free(msgs, nmsg); return 0;
                }
            }
        } else {
            free(key);
            if (!jskip(&j)) { msgs_free(msgs, nmsg); return 0; }
        }
        jws(&j);
        if (j.i < j.n && j.p[j.i] == ',') { j.i++; continue; }
        if (j.i < j.n && j.p[j.i] == '}') { j.i++; break; }
        msgs_free(msgs, nmsg); return 0;
    }
    *out = msgs;
    *out_n = nmsg;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* SFT load                                                                  */
/* ------------------------------------------------------------------------- */

static UenrTokRole role_from_name(const char *r) {
    if (!r) return UENR_ROLE_USER;
    if (!strcmp(r, "system")) return UENR_ROLE_SYSTEM;
    if (!strcmp(r, "user")) return UENR_ROLE_USER;
    if (!strcmp(r, "assistant")) return UENR_ROLE_ASSISTANT;
    if (!strcmp(r, "tool")) return UENR_ROLE_TOOL_RESULT;
    if (!strcmp(r, "tool_call")) return UENR_ROLE_TOOL_CALL;
    if (!strcmp(r, "tool_result")) return UENR_ROLE_TOOL_RESULT;
    return UENR_ROLE_USER;
}

typedef struct { int *ids, *labels; int n, cap; } IVec;

static int iv_push(IVec *v, int id, int label) {
    if (v->n == v->cap) {
        int cap = v->cap ? v->cap * 2 : 128;
        int *ni = (int *)realloc(v->ids, (size_t)cap * sizeof(int));
        if (!ni) return 0;
        v->ids = ni;
        int *nl = (int *)realloc(v->labels, (size_t)cap * sizeof(int));
        if (!nl) return 0;
        v->labels = nl;
        v->cap = cap;
    }
    v->ids[v->n] = id;
    v->labels[v->n] = label;
    v->n++;
    return 1;
}

static int build_sft_sample(const UenrTokenizer *tok, Msg *msgs, int nmsg,
                            UenrSample *out) {
    int ignore = -100;
    IVec v = {0};
    if (!iv_push(&v, uenr_tok_id(tok, UENR_ROLE_BOS), ignore)) return 0;
    for (int i = 0; i < nmsg; i++) {
        UenrTokRole role = role_from_name(msgs[i].role);
        int rid = uenr_tok_id(tok, role);
        int supervised = (role == UENR_ROLE_ASSISTANT);
        if (!iv_push(&v, rid, ignore)) return 0;
        int *ids = NULL, n = 0;
        if (uenr_tok_encode(tok, msgs[i].content, &ids, &n) != UENR_OK) {
            free(v.ids); free(v.labels); return 0;
        }
        for (int k = 0; k < n; k++)
            if (!iv_push(&v, ids[k], supervised ? ids[k] : ignore)) {
                free(ids); free(v.ids); free(v.labels); return 0;
            }
        free(ids);
        int end = uenr_tok_id(tok, UENR_ROLE_END);
        if (!iv_push(&v, end, supervised ? end : ignore)) {
            free(v.ids); free(v.labels); return 0;
        }
    }
    out->ids = v.ids;
    out->labels = v.labels;
    out->n = v.n;
    return 1;
}

UenrResult uenr_sft_load(const UenrConfig *cfg, const UenrTokenizer *tok,
                         const char *path, UenrSFTData *out) {
    (void)cfg;
    if (!tok || !path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    char *p = (char *)data;
    int cap = 0;
    out->samples = NULL;
    out->count = 0;

    while (*p) {
        char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char saved = p[len];
        p[len] = '\0';
        if (len > 1) {
            Msg *msgs = NULL;
            int nmsg = 0;
            if (parse_messages(p, &msgs, &nmsg) && nmsg > 0) {
                UenrSample s;
                if (build_sft_sample(tok, msgs, nmsg, &s)) {
                    if (out->count == cap) {
                        cap = cap ? cap * 2 : 64;
                        UenrSample *ns = (UenrSample *)realloc(out->samples, (size_t)cap * sizeof(UenrSample));
                        if (!ns) { msgs_free(msgs, nmsg); free(data); uenr_sft_free(out); return UENR_ERR_ALLOC; }
                        out->samples = ns;
                    }
                    out->samples[out->count++] = s;
                }
                msgs_free(msgs, nmsg);
            }
        }
        p[len] = saved;
        if (!nl) break;
        p = nl + 1;
    }
    free(data);
    return out->count > 0 ? UENR_OK : UENR_ERR_INVALID;
}

void uenr_sft_free(UenrSFTData *d) {
    if (!d) return;
    for (int i = 0; i < d->count; i++) {
        free(d->samples[i].ids);
        free(d->samples[i].labels);
    }
    free(d->samples);
    d->samples = NULL;
    d->count = 0;
}

int uenr_sft_batch(const UenrSFTData *d, UenrRng *rng, int seq,
                   int nseq, int *inputs, int *labels,
                   int pad_id, int ignore_index) {
    if (!d || d->count <= 0 || !rng || !inputs || !labels) return -1;
    for (int s = 0; s < nseq; s++) {
        int which = uenr_rng_range(rng, d->count);
        const UenrSample *sm = &d->samples[which];
        int copy = sm->n < seq ? sm->n : seq;
        int *in = inputs + (size_t)s * seq;
        int *lb = labels + (size_t)s * seq;
        for (int i = 0; i < copy; i++) { in[i] = sm->ids[i]; lb[i] = sm->labels[i]; }
        for (int i = copy; i < seq; i++) { in[i] = pad_id; lb[i] = ignore_index; }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* preference pairs                                                          */
/* ------------------------------------------------------------------------- */

static int parse_top_strings(const char *line, char **keys, int nkeys, char **vals) {
    JParser j = { line, strlen(line), 0 };
    jws(&j);
    if (j.i >= j.n || j.p[j.i] != '{') return 0;
    j.i++;
    for (int k = 0; k < nkeys; k++) vals[k] = NULL;
    jws(&j);
    if (j.i < j.n && j.p[j.i] == '}') { j.i++; return 1; }
    for (;;) {
        char *key = NULL;
        if (!jstring(&j, &key)) return 0;
        jws(&j);
        if (j.i >= j.n || j.p[j.i] != ':') { free(key); return 0; }
        j.i++;
        int matched = 0;
        for (int k = 0; k < nkeys; k++) {
            if (!strcmp(key, keys[k])) {
                char *s = NULL;
                if (!jstring(&j, &s)) { free(key); return 0; }
                vals[k] = s;
                matched = 1;
                break;
            }
        }
        free(key);
        if (!matched && !jskip(&j)) return 0;
        jws(&j);
        if (j.i < j.n && j.p[j.i] == ',') { j.i++; continue; }
        if (j.i < j.n && j.p[j.i] == '}') { j.i++; return 1; }
        return 0;
    }
}

UenrResult uenr_pairs_load(const char *path, UenrPairData *out) {
    if (!path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    char *p = (char *)data;
    int cap = 0;
    out->pairs = NULL;
    out->count = 0;
    char *keys[3] = { "prompt", "chosen", "rejected" };

    while (*p) {
        char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char saved = p[len];
        p[len] = '\0';
        if (len > 2) {
            char *vals[3];
            if (parse_top_strings(p, keys, 3, vals) && vals[0] && vals[1] && vals[2]) {
                if (out->count == cap) {
                    cap = cap ? cap * 2 : 64;
                    UenrPair *np = (UenrPair *)realloc(out->pairs, (size_t)cap * sizeof(UenrPair));
                    if (!np) { free(vals[0]); free(vals[1]); free(vals[2]); free(data); uenr_pairs_free(out); return UENR_ERR_ALLOC; }
                    out->pairs = np;
                }
                out->pairs[out->count].prompt = vals[0];
                out->pairs[out->count].chosen = vals[1];
                out->pairs[out->count].rejected = vals[2];
                out->count++;
            } else {
                for (int k = 0; k < 3; k++) free(vals[k]);
            }
        }
        p[len] = saved;
        if (!nl) break;
        p = nl + 1;
    }
    free(data);
    return out->count > 0 ? UENR_OK : UENR_ERR_INVALID;
}

void uenr_pairs_free(UenrPairData *d) {
    if (!d) return;
    for (int i = 0; i < d->count; i++) {
        free(d->pairs[i].prompt);
        free(d->pairs[i].chosen);
        free(d->pairs[i].rejected);
    }
    free(d->pairs);
    d->pairs = NULL;
    d->count = 0;
}
