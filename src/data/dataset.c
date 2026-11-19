#include "uenr/dataset.h"
#include "uenr/io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define UDAT_MAGIC 0x54414455u /* "UDAT" */

/* ------------------------------------------------------------------------- */
/* token buffer                                                              */
/* ------------------------------------------------------------------------- */

UenrResult uenr_data_load(const char *path, UenrTokenData *out) {
    if (!path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_mmap_file(path, &data, &size));
    if (size < 7 * sizeof(uint32_t)) { uenr_munmap(data, size); return UENR_ERR_INVALID; }
    uint32_t *h = (uint32_t *)data;
    if (h[0] != UDAT_MAGIC || h[1] != 1 || h[5] != 1) {
        uenr_munmap(data, size);
        return UENR_ERR_INVALID;
    }
    size_t count = (size_t)h[2] | ((size_t)h[3] << 32);
    if (size != 7 * sizeof(uint32_t) + count * sizeof(uint16_t)) {
        uenr_munmap(data, size);
        return UENR_ERR_INVALID;
    }
#if defined(MADV_WILLNEED)
    madvise(data, size, MADV_WILLNEED);
#endif
    uenr_madvise_huge(data, size);
    out->tokens = (uint16_t *)((unsigned char *)data + 7 * sizeof(uint32_t));
    out->count = count;
    out->vocab = (int)h[4];
    out->map = data;
    out->map_size = size;
    return UENR_OK;
}

void uenr_data_free(UenrTokenData *d) {
    if (!d) return;
    if (d->map) uenr_munmap(d->map, d->map_size);
    d->tokens = NULL;
    d->count = 0;
    d->map = NULL;
    d->map_size = 0;
}

int uenr_data_batch(const UenrTokenData *d, UenrRng *rng, int seq,
                    int nseq, int *inputs, int *targets) {
    if (!d || !rng || !inputs || !targets || seq <= 0 || nseq <= 0) return -1;
    if (d->count < (size_t)seq + 1) return -1;
    size_t max_start = d->count - (size_t)seq - 1;
    for (int s = 0; s < nseq; s++) {
        size_t start = (size_t)(uenr_rng_u64(rng) % (max_start + 1));
        const uint16_t *src = d->tokens + start;
        __builtin_prefetch(src + seq, 0, 1);
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

typedef struct {
    FILE *f;
    uint16_t buf[8192];
    int n;
    uint64_t count;
} TokWriter;

static int tw_put(TokWriter *w, uint32_t id) {
    w->buf[w->n++] = (uint16_t)id;
    w->count++;
    if (w->n == 8192) {
        if (fwrite(w->buf, sizeof(uint16_t), 8192, w->f) != 8192) return 0;
        w->n = 0;
    }
    return 1;
}
static int tw_flush(TokWriter *w) {
    if (w->n > 0) {
        if (fwrite(w->buf, sizeof(uint16_t), (size_t)w->n, w->f) != (size_t)w->n) return 0;
        w->n = 0;
    }
    return 1;
}
static int write_udat_header(FILE *f, int vocab) {
    uint32_t h[7] = { UDAT_MAGIC, 1, 0, 0, (uint32_t)vocab, 1, 0 };
    fseek(f, 0, SEEK_SET);
    return fwrite(h, sizeof(uint32_t), 7, f) == 7;
}
static UenrResult finalize_udat(TokWriter *w, const char *tmp, const char *final,
                                int vocab) {
    uint32_t h[7] = { UDAT_MAGIC, 1, (uint32_t)(w->count & 0xffffffffu),
                      (uint32_t)(w->count >> 32), (uint32_t)vocab, 1, 0 };
    if (!tw_flush(w)) return UENR_ERR_IO;
    fseek(w->f, 0, SEEK_SET);
    if (fwrite(h, sizeof(uint32_t), 7, w->f) != 7) return UENR_ERR_IO;
    if (fflush(w->f) != 0) return UENR_ERR_IO;
    if (fsync(fileno(w->f)) != 0) return UENR_ERR_IO;
    if (fclose(w->f) != 0) return UENR_ERR_IO;
    w->f = NULL;
    if (rename(tmp, final) != 0) return UENR_ERR_IO;
    return UENR_OK;
}

static char *reverse_words(const char *src, size_t cap) {
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    size_t w = n, o = 0;
    while (w > 0) {
        while (w > 0 && src[w - 1] == ' ') w--;
        size_t e = w;
        while (w > 0 && src[w - 1] != ' ') w--;
        if (w < e) {
            size_t len = e - w;
            if (o + len + 2 >= cap) break;
            memcpy(out + o, src + w, len);
            o += len;
            if (w > 0 && o + 1 < cap) out[o++] = ' ';
        }
    }
    out[o] = '\0';
    return out;
}

/* Generate the SFT and preference JSONL files from the first N stories. */
static void prepare_sft_files(const UenrConfig *cfg, const char *raw_path,
                              size_t limit, int pretok_dummy) {
    (void)pretok_dummy;
    if (cfg->prep.sft_stories <= 0) return;
    FILE *in = fopen(raw_path, "r");
    if (!in) return;
    if (uenr_make_parent(cfg->paths.sft_jsonl) != UENR_OK ||
        uenr_make_parent(cfg->paths.pairs_jsonl) != UENR_OK) { fclose(in); return; }
    FILE *sf = fopen(cfg->paths.sft_jsonl, "w");
    FILE *pf = fopen(cfg->paths.pairs_jsonl, "w");
    if (!sf || !pf) { if (sf) fclose(sf); if (pf) fclose(pf); fclose(in); return; }

    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    size_t offset = 0;
    int story = 0;
    while ((n = getline(&line, &cap, in)) > 0 && story < cfg->prep.sft_stories) {
        offset += (size_t)n;
        if (limit && offset > limit + n) break;
        size_t tlen = 0;
        const char *text = trim_line(line, &tlen);
        if (tlen <= (size_t)cfg->prep.sft_prompt_chars + 16) continue;
        size_t cut = (size_t)cfg->prep.sft_prompt_chars;
        if (cut > tlen / 2) cut = tlen / 2;
        while (cut > 0 && text[cut] != ' ') cut--;
        if (cut == 0) cut = (size_t)cfg->prep.sft_prompt_chars;
        char prompt[1024];
        if (cut >= sizeof(prompt)) cut = sizeof(prompt) - 1;
        memcpy(prompt, text, cut);
        prompt[cut] = '\0';
        const char *assistant = text + cut;
        while (*assistant == ' ') assistant++;
        size_t alen = strlen(assistant);
        if (alen > 600) alen = 600;
        char *asst = (char *)malloc(alen + 1);
        if (!asst) break;
        memcpy(asst, assistant, alen);
        asst[alen] = '\0';

        fprintf(sf, "{\"messages\":[{\"role\":\"user\",\"content\":\"Continue the story: ");
        json_escape(sf, prompt);
        fprintf(sf, "\"},{\"role\":\"assistant\",\"content\":\"");
        json_escape(sf, asst);
        fprintf(sf, "\"}]}\n");

        {
            char *rej = reverse_words(asst, 720);
            if (rej) {
                fprintf(pf, "{\"prompt\":\"Continue the story: ");
                json_escape(pf, prompt);
                fprintf(pf, "\",\"chosen\":\"");
                json_escape(pf, asst);
                fprintf(pf, "\",\"rejected\":\"");
                json_escape(pf, rej);
                fprintf(pf, "\"}\n");
                free(rej);
            }
        }
        free(asst); /* ownership ends after both JSONL records are written */
        story++;
    }
    free(line);

    /* Synthetic tool-call conversations: assistant spans are supervised and
     * the <|tool_call|>/<|end|> markup stays atomic through the tokenizer. */
    static const char *cities[] = { "Accra", "Kumasi", "Lagos", "Nairobi", "London" };
    static const char *conds[] = { "sunny", "rainy", "cloudy", "windy" };
    char content[512];
    for (int t = 0; t < cfg->prep.tool_examples; t++) {
        const char *city = cities[t % 5];
        const char *cond = conds[t % 4];
        int temp = 18 + (t * 7) % 20;
        fprintf(sf, "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant. Available tool: weather(city).\"},");
        fprintf(sf, "{\"role\":\"user\",\"content\":\"What is the weather in ");
        json_escape(sf, city);
        fprintf(sf, "?\"},");
        snprintf(content, sizeof(content),
                 "<|tool_call|>{\"name\":\"weather\",\"arguments\":{\"city\":\"%s\"}}<|end|>",
                 city);
        fprintf(sf, "{\"role\":\"assistant\",\"content\":\"");
        json_escape(sf, content);
        fprintf(sf, "\"},");
        snprintf(content, sizeof(content), "{\"temp_c\":%d,\"conditions\":\"%s\"}", temp, cond);
        fprintf(sf, "{\"role\":\"tool\",\"content\":\"");
        json_escape(sf, content);
        fprintf(sf, "\"},");
        snprintf(content, sizeof(content), "It is %d C and %s in %s.<|end|>", temp, cond, city);
        fprintf(sf, "{\"role\":\"assistant\",\"content\":\"");
        json_escape(sf, content);
        fprintf(sf, "\"}]}\n");
    }

    fclose(sf);
    fclose(pf);
    fclose(in);
}

static UenrResult prepare_pretrain(const UenrConfig *cfg, const UenrTokenizer *tok) {
    if (!cfg || !tok) return UENR_ERR_INVALID;
    const UenrDatasetCfg *ds = &cfg->datasets.pretrain;
    if (ds->n_paths < 1) return UENR_ERR_INVALID;

    size_t total = 0;
    for (int i = 0; i < ds->n_paths; i++) {
        int64_t sz = uenr_file_size(ds->paths[i]);
        if (sz <= 0) {
            fprintf(stderr, "prepare: cannot stat pretrain path %s\n", ds->paths[i]);
            return UENR_ERR_IO;
        }
        total += (size_t)sz;
    }
    size_t limit = total;
    if (ds->max_bytes > 0 && (size_t)ds->max_bytes < limit)
        limit = (size_t)ds->max_bytes;
    size_t val_start = (size_t)((double)limit * (100 - cfg->prep.val_percent) / 100.0);
    if (cfg->prep.val_percent <= 0) val_start = limit;

    char train_tmp[1024], val_tmp[1024];
    if (snprintf(train_tmp, sizeof(train_tmp), "%s.streamtmp", cfg->paths.train_bin) >= (int)sizeof(train_tmp) ||
        snprintf(val_tmp, sizeof(val_tmp), "%s.streamtmp", cfg->paths.val_bin) >= (int)sizeof(val_tmp))
        return UENR_ERR_INVALID;
    if (uenr_make_parent(cfg->paths.train_bin) != UENR_OK ||
        uenr_make_parent(cfg->paths.val_bin) != UENR_OK) return UENR_ERR_IO;
    FILE *ftr = fopen(train_tmp, "wb");
    FILE *fva = fopen(val_tmp, "wb");
    if (!ftr || !fva) { if (ftr) fclose(ftr); if (fva) fclose(fva); return UENR_ERR_IO; }
    if (!write_udat_header(ftr, uenr_tok_vocab(tok)) ||
        !write_udat_header(fva, uenr_tok_vocab(tok))) {
        fclose(ftr); fclose(fva); remove(train_tmp); remove(val_tmp);
        return UENR_ERR_IO;
    }
    TokWriter tw = { ftr, {0}, 0, 0 }, vw = { fva, {0}, 0, 0 };
    uint64_t *dedup_ht = NULL;
    int dedup_mask = 0, dup_count = 0;
    if (cfg->prep.dedup && cfg->prep.dedup_slots > 0) {
        int slots = cfg->prep.dedup_slots;
        while ((slots & (slots - 1)) != 0) slots++;
        dedup_ht = (uint64_t *)calloc((size_t)slots, sizeof(uint64_t));
        if (dedup_ht) dedup_mask = slots - 1;
    }

    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    size_t offset = 0, lines = 0;
    int oom = 0, pi = 0;
    FILE *f = NULL;
    while (offset < limit) {
        if (!f) {
            if (pi >= ds->n_paths) break;
            f = fopen(ds->paths[pi++], "rb");
            if (!f) { oom = 1; break; }
        }
        n = getline(&line, &cap, f);
        if (n <= 0) { fclose(f); f = NULL; continue; }
        size_t line_start = offset;
        offset += (size_t)n;
        size_t tlen = 0;
        const char *text = trim_line(line, &tlen);
        if (tlen == 0) continue;
        if (dedup_ht) {
            uint64_t hv = 1469598103934665603ULL;
            for (size_t k = 0; k < tlen; k++) { hv ^= (unsigned char)text[k]; hv *= 1099511628211ULL; }
            if (hv == 0) hv = 1;
            int idx = (int)(hv & (uint64_t)dedup_mask), found = 0;
            for (int probe = 0; probe < 8; probe++) {
                int slot = (idx + probe) & dedup_mask;
                if (dedup_ht[slot] == hv) { found = 1; break; }
                if (dedup_ht[slot] == 0) { dedup_ht[slot] = hv; break; }
            }
            if (found) { dup_count++; continue; }
        }
        int *ids = NULL, ni = 0;
        if (uenr_tok_encode(tok, text, &ids, &ni) != UENR_OK) { oom = 1; break; }
        TokWriter *w = (line_start >= val_start) ? &vw : &tw;
        if (cfg->tok.add_bos && !tw_put(w, (uint32_t)uenr_tok_id(tok, UENR_ROLE_BOS))) oom = 1;
        for (int i = 0; i < ni && !oom; i++) if (!tw_put(w, (uint32_t)ids[i])) oom = 1;
        if (cfg->tok.add_eos && !oom && !tw_put(w, (uint32_t)uenr_tok_id(tok, UENR_ROLE_EOS))) oom = 1;
        free(ids);
        if (oom) break;
        lines++;
    }
    if (f) fclose(f);
    free(line);
    free(dedup_ht);

    UenrResult r = UENR_OK;
    if (oom) {
        r = UENR_ERR_ALLOC;
        fclose(ftr); fclose(fva); remove(train_tmp); remove(val_tmp);
    } else {
        r = finalize_udat(&tw, train_tmp, cfg->paths.train_bin, uenr_tok_vocab(tok));
        if (UENR_OK == r)
            r = finalize_udat(&vw, val_tmp, cfg->paths.val_bin, uenr_tok_vocab(tok));
    }
    fprintf(stderr, "prepare[pretrain]: %zu lines (%d duplicates skipped), train %llu tokens, "
            "val %llu tokens (streaming)\n", lines, dup_count,
            (unsigned long long)tw.count, (unsigned long long)vw.count);
    return r;
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

/* Build one chat sample with autoregressive next-token targets.
 *
 * `labels[t]` is the token the model must predict at position t, i.e. the
 * *next* token ids[t+1].  It is supervised only when that next token was
 * produced by the assistant (assistant content or the <end> marker closing an
 * assistant turn).  Role markers, user/tool/system text and the final token
 * (which has no successor) use ignore_index.  The label is therefore never the
 * same token as the input at its own position. */
static int build_sft_sample(const UenrTokenizer *tok, Msg *msgs, int nmsg,
                            int ignore, UenrSample *out) {
    IVec v = {0};
    if (!iv_push(&v, uenr_tok_id(tok, UENR_ROLE_BOS), 0)) return 0;
    for (int i = 0; i < nmsg; i++) {
        UenrTokRole role = role_from_name(msgs[i].role);
        int rid = uenr_tok_id(tok, role);
        int supervised = (role == UENR_ROLE_ASSISTANT);
        if (!iv_push(&v, rid, 0)) return 0;
        int *ids = NULL, n = 0;
        if (uenr_tok_encode(tok, msgs[i].content, &ids, &n) != UENR_OK) {
            free(v.ids); free(v.labels); return 0;
        }
        /* 1 marks "this token was generated by the assistant", 0 marks
         * context.  The shift below turns the marks into next-token labels. */
        for (int k = 0; k < n; k++)
            if (!iv_push(&v, ids[k], supervised ? 1 : 0)) {
                free(ids); free(v.ids); free(v.labels); return 0;
            }
        free(ids);
        int end = uenr_tok_id(tok, UENR_ROLE_END);
        if (!iv_push(&v, end, supervised ? 1 : 0)) {
            free(v.ids); free(v.labels); return 0;
        }
    }
    for (int t = 0; t + 1 < v.n; t++)
        v.labels[t] = v.labels[t + 1] ? v.ids[t + 1] : ignore;
    if (v.n > 0) v.labels[v.n - 1] = ignore;
    out->ids = v.ids;
    out->labels = v.labels;
    out->n = v.n;
    return 1;
}

UenrResult uenr_sft_load(const UenrConfig *cfg, const UenrTokenizer *tok,
                         const char *path, UenrSFTData *out) {
    if (!tok || !path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    char *buf = (char *)realloc(data, size + 1);
    if (!buf) { free(data); return UENR_ERR_ALLOC; }
    buf[size] = '\0';
    data = buf;
    char *p = buf;
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
                if (build_sft_sample(tok, msgs, nmsg, cfg->tok.ignore_index, &s)) {
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
    out->pack_seq = (cfg->train.sft_pack && cfg->train.seq_len > 0)
                        ? cfg->train.seq_len : 0;
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
    d->pack_seq = 0;
}

/* Best-fit packing within one batch: draw a bounded pool, order it longest
 * first, then first-fit each conversation into a row.  Documents never share
 * attention because doc_start marks each boundary.  Rows that receive no
 * conversation are filled with one (possibly truncated) sample. */
static void pack_batch(const UenrSFTData *d, UenrRng *rng, int seq, int nseq,
                       int *inputs, int *labels, int *doc_start,
                       int pad_id, int ignore_index) {
    int pooln = nseq * 8;
    if (pooln < 16) pooln = 16;
    int *pool = (int *)malloc((size_t)pooln * sizeof(int));
    int *rowlen = (int *)calloc((size_t)nseq, sizeof(int));
    if (!pool || !rowlen) { free(pool); free(rowlen); pool = NULL; }
    if (!pool) {
        /* Allocation failure: fall back to one sample per row. */
        for (int s = 0; s < nseq; s++) {
            const UenrSample *sm = &d->samples[uenr_rng_range(rng, d->count)];
            int start = sm->n > seq ? sm->n - seq : 0;
            int copy = sm->n - start;
            if (copy > seq) copy = seq;
            int *in = inputs + (size_t)s * seq, *lb = labels + (size_t)s * seq;
            int *ds = doc_start + (size_t)s * seq;
            for (int i = 0; i < copy; i++) {
                in[i] = sm->ids[start + i]; lb[i] = sm->labels[start + i];
                ds[i] = 0;
            }
            for (int i = copy; i < seq; i++) {
                in[i] = pad_id; lb[i] = ignore_index; ds[i] = -1;
            }
        }
        return;
    }
    for (int i = 0; i < pooln; i++) pool[i] = uenr_rng_range(rng, d->count);
    /* Longest-first reduces the number of rows that stay mostly empty. */
    for (int i = 1; i < pooln; i++) {
        int v = pool[i], j = i - 1;
        while (j >= 0 && d->samples[pool[j]].n < d->samples[v].n) {
            pool[j + 1] = pool[j]; j--;
        }
        pool[j + 1] = v;
    }
    for (int r = 0; r < nseq; r++) {
        int *in = inputs + (size_t)r * seq, *lb = labels + (size_t)r * seq;
        int *ds = doc_start + (size_t)r * seq;
        for (int i = 0; i < seq; i++) { in[i] = pad_id; lb[i] = ignore_index; ds[i] = -1; }
        rowlen[r] = 0;
    }
    for (int i = 0; i < pooln; i++) {
        const UenrSample *sm = &d->samples[pool[i]];
        int start = sm->n > seq ? sm->n - seq : 0;
        int len = sm->n - start;
        if (len > seq) len = seq;
        if (len <= 0) continue;
        int r = -1;
        for (int q = 0; q < nseq; q++)
            if (rowlen[q] + len <= seq) { r = q; break; }
        if (r < 0) continue; /* no row has enough room for this document */
        int *in = inputs + (size_t)r * seq, *lb = labels + (size_t)r * seq;
        int *ds = doc_start + (size_t)r * seq;
        int off = rowlen[r];
        for (int k = 0; k < len; k++) {
            in[off + k] = sm->ids[start + k];
            lb[off + k] = sm->labels[start + k];
            ds[off + k] = off;
        }
        rowlen[r] += len;
    }
    /* Guarantee training signal in every row. */
    for (int r = 0; r < nseq; r++) {
        if (rowlen[r] > 0) continue;
        const UenrSample *sm = &d->samples[uenr_rng_range(rng, d->count)];
        int start = sm->n > seq ? sm->n - seq : 0;
        int copy = sm->n - start;
        if (copy > seq) copy = seq;
        int *in = inputs + (size_t)r * seq, *lb = labels + (size_t)r * seq;
        int *ds = doc_start + (size_t)r * seq;
        for (int i = 0; i < copy; i++) {
            in[i] = sm->ids[start + i]; lb[i] = sm->labels[start + i]; ds[i] = 0;
        }
    }
    free(pool);
    free(rowlen);
}

int uenr_sft_batch(const UenrSFTData *d, UenrRng *rng, int seq,
                   int nseq, int *inputs, int *labels, int *doc_start,
                   int pad_id, int ignore_index) {
    if (!d || d->count <= 0 || !rng || !inputs || !labels) return -1;
    int *ds = doc_start;
    int *tmp = NULL;
    if (!ds) {
        tmp = (int *)malloc((size_t)seq * (size_t)nseq * sizeof(int));
        if (!tmp) return -1;
        ds = tmp;
    }
    if (d->pack_seq > 0) {
        pack_batch(d, rng, seq, nseq, inputs, labels, ds, pad_id, ignore_index);
        free(tmp);
        return 0;
    }
    for (int s = 0; s < nseq; s++) {
        int which = uenr_rng_range(rng, d->count);
        const UenrSample *sm = &d->samples[which];
        /* Keep the tail of an over-long conversation: the supervised
         * assistant span lives at the end, so head truncation can otherwise
         * produce batches with no targets at all. */
        int start = sm->n > seq ? sm->n - seq : 0;
        int copy = sm->n - start;
        if (copy > seq) copy = seq;
        int *in = inputs + (size_t)s * seq;
        int *lb = labels + (size_t)s * seq;
        int *dsp = ds + (size_t)s * seq;
        for (int i = 0; i < copy; i++) {
            in[i] = sm->ids[start + i];
            lb[i] = sm->labels[start + i];
            dsp[i] = 0;
        }
        for (int i = copy; i < seq; i++) {
            in[i] = pad_id; lb[i] = ignore_index; dsp[i] = -1;
        }
    }
    free(tmp);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* preference pairs                                                          */
/* ------------------------------------------------------------------------- */

static int parse_top_strings(const char *line, char **keys, int nkeys, char **vals) {
    JParser j = { line, strlen(line), 0 };
    /* Initialise ownership immediately: every early return must leave the
     * caller with a consistent, freeable array. */
    for (int k = 0; k < nkeys; k++) vals[k] = NULL;
    jws(&j);
    if (j.i >= j.n || j.p[j.i] != '{') return 0;
    j.i++;
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
                if (vals[k]) { /* duplicate key: malformed, don't leak either */
                    free(s); free(key); return 0;
                }
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

/* A preference record larger than this is treated as malformed instead of
 * driving unbounded allocations on a single line. */
#define UENR_PAIR_LINE_MAX (1u << 20)

UenrResult uenr_pairs_load(const char *path, UenrPairData *out) {
    if (!path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    /* read_file returns exactly `size` bytes without a terminator; this
     * loader walks the file as a C string. */
    char *zbuf = (char *)realloc(data, size + 1);
    if (!zbuf) { free(data); return UENR_ERR_ALLOC; }
    data = zbuf;
    ((char *)data)[size] = '\0';
    char *p = (char *)data;
    int cap = 0;
    out->pairs = NULL;
    out->count = 0;
    int skipped = 0, oversized = 0;
    char *keys[3] = { "prompt", "chosen", "rejected" };

    while (*p) {
        char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char saved = p[len];
        p[len] = '\0';
        if (len > 2) {
            char *vals[3] = { NULL, NULL, NULL };
            int ok = len <= UENR_PAIR_LINE_MAX &&
                     parse_top_strings(p, keys, 3, vals) &&
                     vals[0] && vals[1] && vals[2] &&
                     vals[1][0] && vals[2][0];
            if (ok) {
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
                /* missing fields, wrong types, duplicate keys, empty
                 * chosen/rejected, non-JSON and oversized records */
                if (len > UENR_PAIR_LINE_MAX) oversized++;
                else skipped++;
                for (int k = 0; k < 3; k++) free(vals[k]);
            }
        }
        p[len] = saved;
        if (!nl) break;
        p = nl + 1;
    }
    free(data);
    if (skipped || oversized)
        fprintf(stderr, "pairs: skipped %d malformed record(s)%s\n",
                skipped + oversized,
                oversized ? " (including oversized lines)" : "");
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

/* ------------------------------------------------------------------------- */
/* per-phase prepared-data builders                                          */
/* ------------------------------------------------------------------------- */

typedef struct { char **v; int n, cap; } LineVec;

static int lv_push(LineVec *l, const char *s) {
    if (l->n == l->cap) {
        int cap = l->cap ? l->cap * 2 : 1024;
        char **nv = (char **)realloc(l->v, (size_t)cap * sizeof(char *));
        if (!nv) return 0;
        l->v = nv; l->cap = cap;
    }
    char *d = strdup(s);
    if (!d) return 0;
    l->v[l->n++] = d;
    return 1;
}
static void lv_free(LineVec *l) {
    for (int i = 0; i < l->n; i++) free(l->v[i]);
    free(l->v);
    memset(l, 0, sizeof(*l));
}
static int lv_read_paths(const UenrDatasetCfg *ds, LineVec *l) {
    for (int p = 0; p < ds->n_paths; p++) {
        FILE *f = fopen(ds->paths[p], "rb");
        if (!f) {
            fprintf(stderr, "prepare: missing dataset file '%s' (see docs/CLOUD_TRAINING.md)\n",
                    ds->paths[p]);
            return 0;
        }
        char *line = NULL; size_t cap = 0; ssize_t n;
        while ((n = getline(&line, &cap, f)) > 0) {
            while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
            if (n == 0) continue;
            if (!lv_push(l, line)) { free(line); fclose(f); return 0; }
        }
        free(line); fclose(f);
    }
    return 1;
}
static void lv_shuffle(LineVec *l, uint64_t seed) {
    UenrRng rng;
    uenr_rng_seed(&rng, seed);
    for (int i = l->n - 1; i > 0; i--) {
        int j = uenr_rng_range(&rng, i + 1);
        char *t = l->v[i]; l->v[i] = l->v[j]; l->v[j] = t;
    }
}
typedef int (*JsonLineEmitter)(FILE *f, const char *line, const char *aux);

static int emit_sft(FILE *f, const char *line, const char *format) {
    char *keys[2] = { "prompt", "completion" };
    char *vals[2] = { NULL, NULL };
    if (format && !strcmp(format, "jsonl_messages")) {
        fprintf(f, "%s\n", line);
        return 1;
    }
    if (!parse_top_strings(line, keys, 2, vals) || !vals[0] || !vals[1]) {
        free(vals[0]); free(vals[1]);
        return 0;
    }
    fprintf(f, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    json_escape(f, vals[0]);
    fprintf(f, "\"},{\"role\":\"assistant\",\"content\":\"");
    json_escape(f, vals[1]);
    fprintf(f, "\"}]}\n");
    free(vals[0]); free(vals[1]);
    return 1;
}
static int emit_dpo(FILE *f, const char *line, const char *aux) {
    (void)aux;
    char *keys[3] = { "prompt", "chosen", "rejected" };
    char *vals[3] = { NULL, NULL, NULL };
    int ok = parse_top_strings(line, keys, 3, vals) &&
             vals[0] && vals[1] && vals[2] && vals[1][0] && vals[2][0];
    if (ok) fprintf(f, "%s\n", line);
    for (int k = 0; k < 3; k++) free(vals[k]);
    return ok;
}
static int emit_prompt(FILE *f, const char *line, const char *aux) {
    (void)aux;
    char *keys[1] = { "prompt" };
    char *vals[1] = { NULL };
    int ok = parse_top_strings(line, keys, 1, vals) && vals[0] && vals[0][0];
    if (ok) {
        fprintf(f, "{\"prompt\":\"");
        json_escape(f, vals[0]);
        fprintf(f, "\"}\n");
    }
    free(vals[0]);
    return ok;
}

/* Shared "read paths -> shuffle -> cap -> write" pipeline for JSONL phases. */
static UenrResult prepare_jsonl(const UenrConfig *cfg, const UenrDatasetCfg *ds,
                                const char *out_path, const char *label,
                                JsonLineEmitter emit, const char *aux) {
    if (ds->n_paths == 0) {
        fprintf(stderr, "prepare[%s]: no paths configured; keeping existing %s\n",
                label, out_path);
        return UENR_OK;
    }
    LineVec l = {0};
    if (!lv_read_paths(ds, &l)) { lv_free(&l); return UENR_ERR_IO; }
    uint64_t seed = (uint64_t)(ds->shuffle_seed > 0 ? ds->shuffle_seed : cfg->run.seed);
    lv_shuffle(&l, seed);
    int limit = ds->max_samples > 0 && ds->max_samples < l.n ? ds->max_samples : l.n;
    if (uenr_make_parent(out_path) != UENR_OK) { lv_free(&l); return UENR_ERR_IO; }
    FILE *f = fopen(out_path, "wb");
    if (!f) { lv_free(&l); return UENR_ERR_IO; }
    int kept = 0;
    for (int i = 0; i < limit; i++) if (emit(f, l.v[i], aux)) kept++;
    fclose(f);
    lv_free(&l);
    fprintf(stderr, "prepare[%s]: wrote %d/%d samples to %s\n",
            label, kept, limit, out_path);
    return kept > 0 ? UENR_OK : UENR_ERR_INVALID;
}
static UenrResult prepare_sft_phase(const UenrConfig *cfg, const UenrDatasetCfg *ds,
                                    const char *out_path) {
    const char *fmt = ds->format;
    if (fmt[0] && !strcmp(fmt, "inherit")) fmt = cfg->datasets.sft.format;
    return prepare_jsonl(cfg, ds, out_path, "sft", emit_sft, fmt);
}
static UenrResult copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return UENR_ERR_IO;
    if (uenr_make_parent(dst) != UENR_OK) { fclose(in); return UENR_ERR_IO; }
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return UENR_ERR_IO; }
    char buf[65536];
    size_t n;
    int ok = 1;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { ok = 0; break; }
    fclose(in); fclose(out);
    return ok ? UENR_OK : UENR_ERR_IO;
}
UenrResult uenr_data_prepare_phase(const UenrConfig *cfg, const UenrTokenizer *tok,
                                   const char *phase) {
    if (!cfg || !phase) return UENR_ERR_INVALID;
    static const char *const valid[] = {
        "all", "pretrain", "sft", "lora", "dpo", "rl", "grpo", "ppo", "reward", "eval"
    };
    int known = 0;
    for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); i++)
        if (!strcmp(phase, valid[i])) known = 1;
    if (!known) {
        fprintf(stderr, "prepare: unknown phase '%s'\n", phase);
        return UENR_ERR_INVALID;
    }
    int all = !strcmp(phase, "all");
    if ((all || !strcmp(phase, "pretrain")) && !tok) return UENR_ERR_INVALID;
    UenrResult r = UENR_OK;
    if (all || !strcmp(phase, "pretrain"))
        r = prepare_pretrain(cfg, tok);
    if (r == UENR_OK && (all || !strcmp(phase, "sft")))
        r = prepare_sft_phase(cfg, &cfg->datasets.sft, cfg->paths.sft_jsonl);
    if (r == UENR_OK && (all || !strcmp(phase, "lora"))) {
        const char *lout = cfg->paths.lora_jsonl[0] ? cfg->paths.lora_jsonl : cfg->paths.sft_jsonl;
        if (cfg->datasets.lora.n_paths > 0)
            r = prepare_sft_phase(cfg, &cfg->datasets.lora, lout);
        else if (strcmp(lout, cfg->paths.sft_jsonl) != 0 && cfg->paths.sft_jsonl[0])
            r = copy_file(cfg->paths.sft_jsonl, lout);
        /* If the paths alias, the SFT output is already the LoRA source. */
    }
    if (r == UENR_OK && (all || !strcmp(phase, "dpo"))) {
        const char *pout = cfg->paths.pairs_jsonl;
        r = prepare_jsonl(cfg, &cfg->datasets.dpo, pout, "dpo", emit_dpo, NULL);
    }
    if (r == UENR_OK && (all || !strcmp(phase, "rl") || !strcmp(phase, "grpo") || !strcmp(phase, "ppo"))) {
        const char *rout = cfg->paths.rl_jsonl[0] ? cfg->paths.rl_jsonl : cfg->paths.pairs_jsonl;
        r = prepare_jsonl(cfg, &cfg->datasets.rl, rout, "rl", emit_prompt, NULL);
    }
    /* eval uses val.bin from pretrain; the heuristic reward needs no data. */
    return r;
}

UenrResult uenr_data_prepare(const UenrConfig *cfg, const UenrTokenizer *tok) {
    UenrResult r = uenr_data_prepare_phase(cfg, tok, "all");
    /* Legacy laptop profiles have no [datasets.sft]/[datasets.dpo] sources;
     * keep generating the synthetic SFT/preference files from the raw text. */
    if (r == UENR_OK && cfg->datasets.sft.n_paths == 0 &&
        cfg->datasets.dpo.n_paths == 0 && cfg->datasets.pretrain.n_paths > 0) {
        size_t limit = cfg->datasets.pretrain.max_bytes > 0
                           ? (size_t)cfg->datasets.pretrain.max_bytes : 0;
        prepare_sft_files(cfg, cfg->datasets.pretrain.paths[0], limit, 0);
    }
    return r;
}

UenrResult uenr_rl_load(const char *path, UenrPairData *out) {
    if (!path || !out) return UENR_ERR_INVALID;
    void *data = NULL; size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    char *zbuf = (char *)realloc(data, size + 1);
    if (!zbuf) { free(data); return UENR_ERR_ALLOC; }
    data = zbuf; ((char *)data)[size] = '\0';
    char *p = (char *)data;
    out->pairs = NULL; out->count = 0;
    int cap = 0;
    char *keys[1] = { "prompt" };
    while (*p) {
        char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char saved = p[len]; p[len] = '\0';
        if (len > 2) {
            char *vals[1] = { NULL };
            if (parse_top_strings(p, keys, 1, vals) && vals[0] && vals[0][0]) {
                if (out->count == cap) {
                    cap = cap ? cap * 2 : 256;
                    UenrPair *np = (UenrPair *)realloc(out->pairs, (size_t)cap * sizeof(UenrPair));
                    if (!np) { free(vals[0]); free(data); uenr_pairs_free(out); return UENR_ERR_ALLOC; }
                    out->pairs = np;
                }
                out->pairs[out->count].prompt = vals[0];
                out->pairs[out->count].chosen = NULL;
                out->pairs[out->count].rejected = NULL;
                out->count++;
            } else free(vals[0]);
        }
        p[len] = saved;
        if (!nl) break;
        p = nl + 1;
    }
    free(data);
    return out->count > 0 ? UENR_OK : UENR_ERR_INVALID;
}

/* ------------------------------------------------------------------------- */
/* dataset conversion helper (pure C, for bootstrap workflows)               */
/* ------------------------------------------------------------------------- */

/* TinyStoriesInstruct is a text file of blocks:
 *   Features: ...
 *   Words: ...
 *   Summary: ...
 *   Story:
 *   <story...>
 *   <|endoftext|>
 * Convert each block to {"prompt": fields, "completion": story}. */
static int convert_instruct(const char *in_path, const char *out_path, int limit) {
    FILE *in = fopen(in_path, "rb");
    if (!in) { fprintf(stderr, "convert: cannot open %s\n", in_path); return 0; }
    if (uenr_make_parent(out_path) != UENR_OK) { fclose(in); return 0; }
    FILE *out = fopen(out_path, "wb");
    if (!out) { fclose(in); return 0; }
    char *line = NULL; size_t cap = 0; ssize_t n;
    char *block = NULL; size_t blen = 0, bcap = 0;
    int kept = 0;
    for (;;) {
        if (kept >= limit) break;
        n = getline(&line, &cap, in);
        int eof = n <= 0;
        int endmark = !eof && strstr(line, "<|endoftext|>") != NULL;
        if (!eof && !endmark) {
            if (blen + (size_t)n + 1 > bcap) {
                size_t nc = bcap ? bcap * 2 : 4096;
                while (nc < blen + (size_t)n + 1) nc *= 2;
                char *nb = (char *)realloc(block, nc);
                if (!nb) break;
                block = nb; bcap = nc;
            }
            memcpy(block + blen, line, (size_t)n); blen += (size_t)n;
            block[blen] = '\0';
            continue;
        }
        if (block && blen) {
            char *story = strstr(block, "Story:");
            if (story && kept < limit) {
                size_t plen = (size_t)(story - block);
                char *prompt = (char *)malloc(plen + 1);
                char *completion = strdup(story + 6);
                if (prompt && completion) {
                    memcpy(prompt, block, plen); prompt[plen] = '\0';
                    /* trim leading newlines/spaces from the story */
                    char *c = completion;
                    while (*c == '\n' || *c == '\r' || *c == ' ') c++;
                    /* trim trailing whitespace */
                    size_t cl = strlen(c);
                    while (cl > 0 && (c[cl - 1] == '\n' || c[cl - 1] == '\r' || c[cl - 1] == ' '))
                        c[--cl] = '\0';
                    if (cl > 0) {
                        fprintf(out, "{\"prompt\":\"");
                        json_escape(out, prompt);
                        fprintf(out, "\",\"completion\":\"");
                        json_escape(out, c);
                        fprintf(out, "\"}\n");
                        kept++;
                    }
                }
                free(prompt); free(completion);
            }
            blen = 0;
            if (block) block[0] = '\0';
        }
        if (eof) break;
    }
    free(block); free(line);
    fclose(in); fclose(out);
    fprintf(stderr, "convert: wrote %d prompt/completion samples to %s\n", kept, out_path);
    return kept > 0;
}

static int convert_sft_derive(const char *in_path, const char *out_path,
                              int limit, int to_prefs) {
    FILE *in = fopen(in_path, "rb");
    if (!in) { fprintf(stderr, "convert: cannot open %s\n", in_path); return 0; }
    if (uenr_make_parent(out_path) != UENR_OK) { fclose(in); return 0; }
    FILE *out = fopen(out_path, "wb");
    if (!out) { fclose(in); return 0; }
    char *line = NULL; size_t cap = 0; ssize_t n;
    int kept = 0;
    char *keys[2] = { "prompt", "completion" };
    while ((n = getline(&line, &cap, in)) > 0 && kept < limit) {
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        char *vals[2] = { NULL, NULL };
        if (!parse_top_strings(line, keys, 2, vals) || !vals[0] || !vals[1] ||
            !vals[0][0] || !vals[1][0]) {
            free(vals[0]); free(vals[1]); continue;
        }
        if (to_prefs) {
            char *rej = reverse_words(vals[1], 4096);
            if (rej && rej[0]) {
                fprintf(out, "{\"prompt\":\"");
                json_escape(out, vals[0]);
                fprintf(out, "\",\"chosen\":\"");
                json_escape(out, vals[1]);
                fprintf(out, "\",\"rejected\":\"");
                json_escape(out, rej);
                fprintf(out, "\"}\n");
                kept++;
            }
            free(rej);
        } else {
            fprintf(out, "{\"prompt\":\"");
            json_escape(out, vals[0]);
            fprintf(out, "\"}\n");
            kept++;
        }
        free(vals[0]); free(vals[1]);
    }
    free(line);
    fclose(in); fclose(out);
    fprintf(stderr, "convert: wrote %d samples to %s\n", kept, out_path);
    return kept > 0;
}

int uenr_convert_file(const char *from, const char *to,
                      const char *in_path, const char *out_path, int limit) {
    if (!from || !to || !in_path || !out_path) return 0;
    if (limit <= 0) limit = 1000000000;
    if (!strcmp(from, "tinystories_instruct") && !strcmp(to, "prompt_completion"))
        return convert_instruct(in_path, out_path, limit);
    if (!strcmp(from, "prompt_completion") && !strcmp(to, "prefs"))
        return convert_sft_derive(in_path, out_path, limit, 1);
    if (!strcmp(from, "prompt_completion") && !strcmp(to, "prompts"))
        return convert_sft_derive(in_path, out_path, limit, 0);
    fprintf(stderr, "convert: unsupported %s -> %s\n", from, to);
    return 0;
}
