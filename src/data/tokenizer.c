#include "uenr/tokenizer.h"
#include "uenr/io.h"
#include "uenr/rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UTOK_MAGIC 0x4b4f5455u /* "UTOK" */

struct UenrTokenizer {
    int vocab_size;
    int n_merges;
    uint32_t *merge_a;
    uint32_t *merge_b;

    int roles[UENR_ROLE_COUNT];
    const char *special_str[UENR_SPECIAL_SLOTS];
    int add_bos, add_eos;
    int normalize_unicode, strip_control, lowercase;

    /* rank lookup for the encoder */
    uint64_t *rank_key;
    int      *rank_val;
    uint32_t  rank_mask;
};

/* ------------------------------------------------------------------------- */
/* misc                                                                      */
/* ------------------------------------------------------------------------- */

static uint32_t next_pow2(uint32_t x) {
    x = x ? x : 1;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

static uint64_t pair_key(uint32_t a, uint32_t b) {
    return ((uint64_t)a << 32) | (uint64_t)b;
}

static uint32_t hash64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

static void init_roles(UenrTokenizer *t, const UenrConfig *cfg) {
    t->roles[UENR_ROLE_PAD] = cfg->tok.pad_id;
    t->roles[UENR_ROLE_BOS] = cfg->tok.bos_id;
    t->roles[UENR_ROLE_EOS] = cfg->tok.eos_id;
    t->roles[UENR_ROLE_UNK] = cfg->tok.unk_id;
    t->roles[UENR_ROLE_SYSTEM] = cfg->tok.special_system;
    t->roles[UENR_ROLE_USER] = cfg->tok.special_user;
    t->roles[UENR_ROLE_ASSISTANT] = cfg->tok.special_assistant;
    t->roles[UENR_ROLE_TOOL_CALL] = cfg->tok.special_tool_call;
    t->roles[UENR_ROLE_TOOL_RESULT] = cfg->tok.special_tool_result;
    t->roles[UENR_ROLE_END] = cfg->tok.special_end;

    for (int i = 0; i < UENR_SPECIAL_SLOTS; i++) t->special_str[i] = "";
    for (int r = 0; r < UENR_ROLE_COUNT; r++) {
        static const char *names[UENR_ROLE_COUNT] = {
            "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|system|>",
            "<|user|>", "<|assistant|>", "<|tool_call|>", "<|tool_result|>",
            "<|end|>"
        };
        t->special_str[t->roles[r]] = names[r];
    }
    t->add_bos = cfg->tok.add_bos;
    t->add_eos = cfg->tok.add_eos;
    t->normalize_unicode = cfg->tok.normalize_unicode;
    t->strip_control = cfg->tok.strip_control;
    t->lowercase = cfg->tok.lowercase;
}

/* ------------------------------------------------------------------------- */
/* sanitizer                                                                 */
/* ------------------------------------------------------------------------- */

static int utf8_len(const unsigned char *p, size_t rem) {
    if (p[0] < 0x80) return 1;
    if (p[0] >= 0xC2 && p[0] <= 0xDF) {
        if (rem < 2 || (p[1] & 0xC0) != 0x80) return 0;
        return 2;
    }
    if (p[0] >= 0xE0 && p[0] <= 0xEF) {
        if (rem < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        if (p[0] == 0xE0 && p[1] < 0xA0) return 0;
        if (p[0] == 0xED && p[1] >= 0xA0) return 0; /* surrogate */
        return 3;
    }
    if (p[0] >= 0xF0 && p[0] <= 0xF4) {
        if (rem < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 ||
            (p[3] & 0xC0) != 0x80) return 0;
        if (p[0] == 0xF0 && p[1] < 0x90) return 0;
        if (p[0] == 0xF4 && p[1] >= 0x90) return 0;
        return 4;
    }
    return 0;
}

UenrResult uenr_tok_sanitize(const UenrTokenizer *t, const char *in, size_t n,
                             char **out, size_t *out_n) {
    if (!t || !in || !out || !out_n) return UENR_ERR_INVALID;
    unsigned char *buf = (unsigned char *)malloc(n + 1);
    if (!buf) return UENR_ERR_ALLOC;
    const unsigned char *p = (const unsigned char *)in;
    size_t i = 0, o = 0;
    int newlines = 0;

    while (i < n) {
        unsigned char c = p[i];
        if (c == '\r') {
            if (i + 1 < n && p[i + 1] == '\n') i++;
            c = '\n';
            i++;
        } else if (c == '\n') {
            i++;
        } else if (c < 0x80) {
            i++;
            if (t->strip_control && (c < 0x20 || c == 0x7f) && c != '\t') continue;
            if (t->lowercase && c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        } else {
            int len = utf8_len(p + i, n - i);
            if (len == 0) {
                buf[o++] = '?';
                i++;
                newlines = 0;
                continue;
            }
            for (int k = 0; k < len; k++) buf[o++] = p[i + k];
            i += (size_t)len;
            newlines = 0;
            continue;
        }

        if (c == '\n') {
            newlines++;
            if (newlines > 2) continue;
        } else {
            newlines = 0;
        }
        buf[o++] = c;
    }
    buf[o] = '\0';
    *out = (char *)buf;
    *out_n = o;
    return UENR_OK;
}

/* ------------------------------------------------------------------------- */
/* rank map + encoder                                                        */
/* ------------------------------------------------------------------------- */

static void build_rank_map(UenrTokenizer *t) {
    uint32_t size = next_pow2((uint32_t)(t->n_merges * 2 + 8));
    t->rank_key = (uint64_t *)calloc(size, sizeof(uint64_t));
    t->rank_val = (int *)malloc((size_t)size * sizeof(int));
    t->rank_mask = size - 1;
    for (uint32_t i = 0; i < size; i++) t->rank_val[i] = -1;
    for (int r = 0; r < t->n_merges; r++) {
        uint64_t key = pair_key(t->merge_a[r], t->merge_b[r]);
        uint32_t idx = hash64(key) & t->rank_mask;
        while (t->rank_key[idx] != 0 && t->rank_key[idx] != key) idx = (idx + 1) & t->rank_mask;
        t->rank_key[idx] = key;
        t->rank_val[idx] = r;
    }
}

static int rank_lookup(const UenrTokenizer *t, uint64_t key) {
    uint32_t idx = hash64(key) & t->rank_mask;
    for (;;) {
        if (t->rank_key[idx] == 0) return -1;
        if (t->rank_key[idx] == key) return t->rank_val[idx];
        idx = (idx + 1) & t->rank_mask;
    }
}

typedef struct { int rank, pos; } HNode;

static inline int hless(const HNode *a, const HNode *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->pos < b->pos;
}

static void heap_push(HNode *h, int *n, HNode v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!hless(&h[i], &h[p])) break;
        HNode tmp = h[i]; h[i] = h[p]; h[p] = tmp;
        i = p;
    }
}

static HNode heap_pop(HNode *h, int *n) {
    HNode top = h[0];
    h[0] = h[--(*n)];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && hless(&h[l], &h[m])) m = l;
        if (r < *n && hless(&h[r], &h[m])) m = r;
        if (m == i) break;
        HNode tmp = h[i]; h[i] = h[m]; h[m] = tmp;
        i = m;
    }
    return top;
}

static UenrResult bpe_encode_bytes(const UenrTokenizer *t, const unsigned char *bytes,
                                   size_t n, int **out_ids, int *out_n) {
    if (n == 0) {
        *out_ids = NULL; *out_n = 0; return UENR_OK;
    }
    if (n > (size_t)0x7fffffff - 8) return UENR_ERR_BOUNDS;

    int *ids = (int *)malloc(n * sizeof(int));
    int *next = (int *)malloc(n * sizeof(int));
    int *prev = (int *)malloc(n * sizeof(int));
    HNode *heap = (HNode *)malloc((4 * n + 16) * sizeof(HNode));
    if (!ids || !next || !prev || !heap) {
        free(ids); free(next); free(prev); free(heap);
        return UENR_ERR_ALLOC;
    }
    for (size_t i = 0; i < n; i++) {
        ids[i] = UENR_BYTE_BASE + (int)bytes[i];
        prev[i] = (int)i - 1;
        next[i] = (i + 1 < n) ? (int)i + 1 : -1;
    }

    int hn = 0;
    for (int i = 0; i + 1 < (int)n; i++) {
        int r = rank_lookup(t, pair_key((uint32_t)ids[i], (uint32_t)ids[i + 1]));
        if (r >= 0) heap_push(heap, &hn, (HNode){ r, i });
    }

    while (hn > 0) {
        HNode e = heap_pop(heap, &hn);
        int l = e.pos;
        int r = next[l];
        if (r < 0) continue;
        int cur = rank_lookup(t, pair_key((uint32_t)ids[l], (uint32_t)ids[r]));
        if (cur != e.rank) continue;

        int nxt = next[r];
        ids[l] = UENR_MERGE_BASE + e.rank;
        next[r] = -2; /* removed: invalidate any heap entry at r */
        next[l] = nxt;
        if (nxt >= 0) prev[nxt] = l;

        if (prev[l] >= 0) {
            int rr = rank_lookup(t, pair_key((uint32_t)ids[prev[l]], (uint32_t)ids[l]));
            if (rr >= 0) heap_push(heap, &hn, (HNode){ rr, prev[l] });
        }
        if (next[l] >= 0) {
            int rr = rank_lookup(t, pair_key((uint32_t)ids[l], (uint32_t)ids[next[l]]));
            if (rr >= 0) heap_push(heap, &hn, (HNode){ rr, l });
        }
    }

    int cnt = 0;
    for (int i = 0; i >= 0; i = next[i]) cnt++;
    int *res = (int *)malloc((size_t)cnt * sizeof(int));
    if (!res) { free(ids); free(next); free(prev); free(heap); return UENR_ERR_ALLOC; }
    int k = 0;
    for (int i = 0; i >= 0; i = next[i]) res[k++] = ids[i];

    free(ids); free(next); free(prev); free(heap);
    *out_ids = res;
    *out_n = cnt;
    return UENR_OK;
}

UenrResult uenr_tok_encode(const UenrTokenizer *t, const char *text,
                           int **ids, int *n) {
    if (!t || !text || !ids || !n) return UENR_ERR_INVALID;
    char *clean = NULL;
    size_t cn = 0;
    UENR_TRY(uenr_tok_sanitize(t, text, strlen(text), &clean, &cn));
    UenrResult r = bpe_encode_bytes(t, (const unsigned char *)clean, cn, ids, n);
    free(clean);
    return r;
}

/* ------------------------------------------------------------------------- */
/* decoder                                                                   */
/* ------------------------------------------------------------------------- */

typedef struct {
    char *buf;
    size_t len, cap;
    int depth;
} DecBuf;

static int dbuf_put(DecBuf *d, const char *s, size_t n) {
    if (d->len + n + 1 > d->cap) {
        size_t cap = d->cap ? d->cap * 2 : 256;
        while (cap < d->len + n + 1) cap *= 2;
        char *p = (char *)realloc(d->buf, cap);
        if (!p) return 0;
        d->buf = p; d->cap = cap;
    }
    memcpy(d->buf + d->len, s, n);
    d->len += n;
    d->buf[d->len] = '\0';
    return 1;
}

static int decode_id(const UenrTokenizer *t, int id, DecBuf *d) {
    if (d->depth++ > 4096) return 0;
    int ok = 1;
    if (id >= 0 && id < UENR_SPECIAL_SLOTS) {
        const char *s = t->special_str[id];
        ok = dbuf_put(d, s, strlen(s));
    } else if (id >= UENR_BYTE_BASE && id < UENR_MERGE_BASE) {
        unsigned char c = (unsigned char)(id - UENR_BYTE_BASE);
        ok = dbuf_put(d, (const char *)&c, 1);
    } else if (id >= UENR_MERGE_BASE && id < UENR_MERGE_BASE + t->n_merges) {
        int r = id - UENR_MERGE_BASE;
        ok = decode_id(t, (int)t->merge_a[r], d) &&
             decode_id(t, (int)t->merge_b[r], d);
    } else {
        ok = dbuf_put(d, "?", 1);
    }
    d->depth--;
    return ok;
}

UenrResult uenr_tok_decode(const UenrTokenizer *t, const int *ids, int n,
                           char **text, size_t *len) {
    if (!t || !ids || !text) return UENR_ERR_INVALID;
    DecBuf d = {0};
    for (int i = 0; i < n; i++) {
        if (!decode_id(t, ids[i], &d)) { free(d.buf); return UENR_ERR_BOUNDS; }
    }
    if (!d.buf) {
        d.buf = (char *)malloc(1);
        if (!d.buf) return UENR_ERR_ALLOC;
        d.buf[0] = '\0';
    }
    *text = d.buf;
    if (len) *len = d.len;
    return UENR_OK;
}

/* ------------------------------------------------------------------------- */
/* train                                                                     */
/* ------------------------------------------------------------------------- */

typedef struct {
    uint64_t key;
    uint32_t count;
    uint32_t gen;
} PairSlot;

typedef struct {
    PairSlot *slots;
    uint32_t mask;
    uint32_t gen;
    uint32_t *used;
    size_t nused, cap;
} PairCounter;

static int pc_init(PairCounter *pc, size_t approx) {
    uint32_t size = next_pow2((uint32_t)(approx * 2 + 16));
    while (size < (1u << 16)) size <<= 1;
    pc->slots = (PairSlot *)calloc(size, sizeof(PairSlot));
    pc->used = (uint32_t *)malloc((size_t)size * sizeof(uint32_t));
    pc->mask = size - 1;
    pc->gen = 1;
    pc->nused = 0;
    pc->cap = size;
    return pc->slots && pc->used;
}

static void pc_begin(PairCounter *pc) {
    pc->gen++;
    if (pc->gen == 0) {
        memset(pc->slots, 0, ((size_t)pc->mask + 1) * sizeof(PairSlot));
        pc->gen = 1;
    }
    pc->nused = 0;
}

static void pc_add(PairCounter *pc, uint64_t key) {
    uint32_t idx = hash64(key) & pc->mask;
    for (;;) {
        PairSlot *s = &pc->slots[idx];
        if (s->gen != pc->gen) {
            s->gen = pc->gen;
            s->key = key;
            s->count = 1;
            if (pc->nused < pc->cap) pc->used[pc->nused++] = idx;
            return;
        }
        if (s->key == key) { s->count++; return; }
        idx = (idx + 1) & pc->mask;
    }
}

static void pc_destroy(PairCounter *pc) {
    free(pc->slots);
    free(pc->used);
}

UenrResult uenr_tok_train(const UenrConfig *cfg, const char *data_path,
                          const char *out_path, const char *vocab_path,
                          const char *merges_path) {
    if (!cfg || !data_path || !out_path) return UENR_ERR_INVALID;
    int n_merges = cfg->model.vocab_size - UENR_MERGE_BASE;
    if (n_merges <= 0) return UENR_ERR_INVALID;

    /* Read a bounded training sample. */
    FILE *f = fopen(data_path, "rb");
    if (!f) return UENR_ERR_IO;
    size_t want = (size_t)cfg->tok.train_bytes;
    char *raw = (char *)malloc(want + 1);
    if (!raw) { fclose(f); return UENR_ERR_ALLOC; }
    size_t got = fread(raw, 1, want, f);
    fclose(f);
    raw[got] = '\0';
    fprintf(stderr, "tok: read %zu bytes from %s\n", got, data_path);

    UenrTokenizer tmp;
    memset(&tmp, 0, sizeof(tmp));
    init_roles(&tmp, cfg);
    char *clean = NULL;
    size_t cn = 0;
    UenrResult rc = uenr_tok_sanitize(&tmp, raw, got, &clean, &cn);
    free(raw);
    fprintf(stderr, "tok: sanitized %zu bytes\n", cn);
    if (rc != UENR_OK) return rc;

    int *ids = (int *)malloc(cn * sizeof(int));
    if (!ids) { free(clean); return UENR_ERR_ALLOC; }
    int n = (int)cn;
    for (int i = 0; i < n; i++) ids[i] = UENR_BYTE_BASE + (unsigned char)clean[i];
    free(clean);

    uint32_t *ma = (uint32_t *)malloc((size_t)n_merges * sizeof(uint32_t));
    uint32_t *mb = (uint32_t *)malloc((size_t)n_merges * sizeof(uint32_t));
    int *tmpids = (int *)malloc((size_t)n * sizeof(int));
    PairCounter pc;
    if (!ma || !mb || !tmpids || !pc_init(&pc, (size_t)n)) {
        free(ids); free(ma); free(mb); free(tmpids);
        return UENR_ERR_ALLOC;
    }

    for (int m = 0; m < n_merges; m++) {
        if (n < 2) { /* freeze early: pad with impossible merges */
            ma[m] = 0xFFFFFFFEu; mb[m] = 0xFFFFFFFEu;
            continue;
        }
        pc_begin(&pc);
        for (int i = 0; i + 1 < n; i++)
            pc_add(&pc, pair_key((uint32_t)ids[i], (uint32_t)ids[i + 1]));

        uint64_t best_key = 0;
        uint32_t best_cnt = 0;
        for (size_t u = 0; u < pc.nused; u++) {
            PairSlot *s = &pc.slots[pc.used[u]];
            if (s->count > best_cnt ||
                (s->count == best_cnt && best_key != 0 && s->key < best_key)) {
                best_cnt = s->count;
                best_key = s->key;
            }
        }
        if (best_cnt == 0) {
            ma[m] = 0xFFFFFFFEu; mb[m] = 0xFFFFFFFEu;
            continue;
        }
        uint32_t a = (uint32_t)(best_key >> 32);
        uint32_t b = (uint32_t)(best_key & 0xffffffffu);
        ma[m] = a; mb[m] = b;
        int new_id = UENR_MERGE_BASE + m;

        int o = 0;
        for (int i = 0; i < n;) {
            if (i + 1 < n && (uint32_t)ids[i] == a && (uint32_t)ids[i + 1] == b) {
                tmpids[o++] = new_id;
                i += 2;
            } else {
                tmpids[o++] = ids[i++];
            }
        }
        int *sw = ids; ids = tmpids; tmpids = sw;
        n = o;
        if ((m & 63) == 0)
            fprintf(stderr, "  bpe merge %d/%d  tokens=%d  best=%u\n",
                    m + 1, n_merges, n, best_cnt);
    }
    pc_destroy(&pc);
    free(ids); free(tmpids);

    /* Serialize artifact. */
    size_t hdr = 5 * sizeof(uint32_t);
    size_t bytes = hdr + (size_t)n_merges * 2 * sizeof(uint32_t);
    unsigned char *blob = (unsigned char *)malloc(bytes);
    if (!blob) { free(ma); free(mb); return UENR_ERR_ALLOC; }
    uint32_t *u = (uint32_t *)blob;
    u[0] = UTOK_MAGIC;
    u[1] = 1;
    u[2] = (uint32_t)cfg->model.vocab_size;
    u[3] = (uint32_t)n_merges;
    u[4] = UENR_MERGE_BASE;
    for (int i = 0; i < n_merges; i++) {
        u[5 + 2 * i] = ma[i];
        u[6 + 2 * i] = mb[i];
    }
    UenrResult r = uenr_write_file_atomic(out_path, blob, bytes);
    if (UENR_OK == r && vocab_path) {
        FILE *vf = fopen(vocab_path, "w");
        if (vf) {
            for (int id = 0; id < UENR_SPECIAL_SLOTS; id++)
                fprintf(vf, "%d\t%s\n", id, tmp.special_str[id]);
            for (int b = 0; b < 256; b++)
                fprintf(vf, "%d\t<byte %02x>\n", UENR_BYTE_BASE + b, b);
            for (int i = 0; i < n_merges; i++)
                fprintf(vf, "%d\t%d %d\n", UENR_MERGE_BASE + i, (int)ma[i], (int)mb[i]);
            fclose(vf);
        }
    }
    if (UENR_OK == r && merges_path) {
        FILE *mf = fopen(merges_path, "w");
        if (mf) {
            for (int i = 0; i < n_merges; i++)
                fprintf(mf, "%u %u\n", ma[i], mb[i]);
            fclose(mf);
        }
    }
    free(blob); free(ma); free(mb);
    return r;
}

/* ------------------------------------------------------------------------- */
/* load / free                                                               */
/* ------------------------------------------------------------------------- */

UenrResult uenr_tok_load(const UenrConfig *cfg, const char *path, UenrTokenizer **out) {
    if (!cfg || !path || !out) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    if (size < 5 * sizeof(uint32_t)) { free(data); return UENR_ERR_INVALID; }

    uint32_t *u = (uint32_t *)data;
    if (u[0] != UTOK_MAGIC || u[1] != 1) { free(data); return UENR_ERR_INVALID; }
    int vocab = (int)u[2];
    int n_merges = (int)u[3];
    if (vocab != cfg->model.vocab_size || n_merges != vocab - UENR_MERGE_BASE ||
        size != 5 * sizeof(uint32_t) + (size_t)n_merges * 2 * sizeof(uint32_t)) {
        free(data);
        return UENR_ERR_INVALID;
    }

    UenrTokenizer *t = (UenrTokenizer *)calloc(1, sizeof(*t));
    if (!t) { free(data); return UENR_ERR_ALLOC; }
    t->vocab_size = vocab;
    t->n_merges = n_merges;
    t->merge_a = (uint32_t *)malloc((size_t)n_merges * sizeof(uint32_t));
    t->merge_b = (uint32_t *)malloc((size_t)n_merges * sizeof(uint32_t));
    if (!t->merge_a || !t->merge_b) {
        free(data); uenr_tok_free(t); return UENR_ERR_ALLOC;
    }
    for (int i = 0; i < n_merges; i++) {
        t->merge_a[i] = u[5 + 2 * i];
        t->merge_b[i] = u[6 + 2 * i];
    }
    free(data);
    init_roles(t, cfg);
    build_rank_map(t);
    if (!t->rank_key || !t->rank_val) { uenr_tok_free(t); return UENR_ERR_ALLOC; }
    *out = t;
    return UENR_OK;
}

void uenr_tok_free(UenrTokenizer *t) {
    if (!t) return;
    free(t->merge_a);
    free(t->merge_b);
    free(t->rank_key);
    free(t->rank_val);
    free(t);
}

int uenr_tok_vocab(const UenrTokenizer *t) { return t ? t->vocab_size : 0; }
int uenr_tok_id(const UenrTokenizer *t, UenrTokRole role) {
    return (t && role >= 0 && role < UENR_ROLE_COUNT) ? t->roles[role] : -1;
}
int uenr_tok_role(const UenrTokenizer *t, int id) {
    if (!t) return -1;
    for (int r = 0; r < UENR_ROLE_COUNT; r++)
        if (t->roles[r] == id) return r;
    return -1;
}

/* ------------------------------------------------------------------------- */
/* chat templates                                                            */
/* ------------------------------------------------------------------------- */

typedef struct {
    int *ids;
    int *labels;
    int n, cap;
    int has_labels;
} TokVec;

static int tv_push(TokVec *v, int id, int label) {
    if (v->n + 1 > v->cap) {
        int cap = v->cap ? v->cap * 2 : 256;
        int *ni = (int *)realloc(v->ids, (size_t)cap * sizeof(int));
        if (!ni) return 0;
        v->ids = ni;
        if (v->has_labels) {
            int *nl = (int *)realloc(v->labels, (size_t)cap * sizeof(int));
            if (!nl) return 0;
            v->labels = nl;
        }
        v->cap = cap;
    }
    v->ids[v->n] = id;
    if (v->has_labels) v->labels[v->n] = label;
    v->n++;
    return 1;
}

/* Encode content, but recognise literal special-token markup (tool calls,
 * message delimiters) and emit the atomic special ids instead of bytes.
 * mode==0: never supervise. mode==1: supervise every emitted token. */
static int tv_text_mode(TokVec *v, const UenrTokenizer *t, const char *s, int mode) {
    const char *p = s;
    while (*p) {
        const char *best = NULL;
        int best_id = -1;
        size_t best_len = 0;
        for (int id = 0; id < UENR_SPECIAL_SLOTS; id++) {
            const char *ss = t->special_str[id];
            if (!ss || !ss[0]) continue;
            const char *q = strstr(p, ss);
            if (q && (!best || q < best)) {
                best = q;
                best_id = id;
                best_len = strlen(ss);
            }
        }
        const char *end = best ? best : p + strlen(p);
        if (end > p) {
            char *chunk = (char *)malloc((size_t)(end - p) + 1);
            if (!chunk) return 0;
            memcpy(chunk, p, (size_t)(end - p));
            chunk[end - p] = '\0';
            int *ids = NULL, n = 0;
            if (uenr_tok_encode(t, chunk, &ids, &n) != UENR_OK) {
                free(chunk);
                return 0;
            }
            free(chunk);
            for (int i = 0; i < n; i++) {
                int lab = mode ? ids[i] : 0;
                if (!tv_push(v, ids[i], lab)) { free(ids); return 0; }
            }
            free(ids);
        }
        if (!best) break;
        int lab = mode ? best_id : 0;
        if (!tv_push(v, best_id, lab)) return 0;
        p = best + best_len;
    }
    return 1;
}

static int tv_text(TokVec *v, const UenrTokenizer *t, const char *s, int label) {
    (void)label;
    return tv_text_mode(v, t, s, 0);
}

UenrResult uenr_chat_infer(const UenrTokenizer *t, const char *system,
                           const char *user, int **ids, int *n) {
    if (!t || !user || !ids || !n) return UENR_ERR_INVALID;
    TokVec v = {0};
    if (!tv_push(&v, t->roles[UENR_ROLE_BOS], 0)) return UENR_ERR_ALLOC;
    if (system && system[0]) {
        if (!tv_push(&v, t->roles[UENR_ROLE_SYSTEM], 0) ||
            !tv_text(&v, t, system, 0) ||
            !tv_push(&v, t->roles[UENR_ROLE_END], 0)) { free(v.ids); return UENR_ERR_ALLOC; }
    }
    if (!tv_push(&v, t->roles[UENR_ROLE_USER], 0) ||
        !tv_text(&v, t, user, 0) ||
        !tv_push(&v, t->roles[UENR_ROLE_END], 0) ||
        !tv_push(&v, t->roles[UENR_ROLE_ASSISTANT], 0)) {
        free(v.ids);
        return UENR_ERR_ALLOC;
    }
    *ids = v.ids;
    *n = v.n;
    return UENR_OK;
}

UenrResult uenr_chat_sft(const UenrTokenizer *t, const char *system,
                         const char *user, const char *assistant,
                         int **ids, int **labels, int *n) {
    if (!t || !user || !assistant || !ids || !labels || !n) return UENR_ERR_INVALID;
    int ignore = -100;
    TokVec v = {0};
    v.has_labels = 1;
    if (!tv_push(&v, t->roles[UENR_ROLE_BOS], ignore)) return UENR_ERR_ALLOC;
    if (system && system[0]) {
        if (!tv_push(&v, t->roles[UENR_ROLE_SYSTEM], ignore) ||
            !tv_text(&v, t, system, ignore) ||
            !tv_push(&v, t->roles[UENR_ROLE_END], ignore)) { free(v.ids); free(v.labels); return UENR_ERR_ALLOC; }
    }
    if (!tv_push(&v, t->roles[UENR_ROLE_USER], ignore) ||
        !tv_text(&v, t, user, ignore) ||
        !tv_push(&v, t->roles[UENR_ROLE_END], ignore) ||
        !tv_push(&v, t->roles[UENR_ROLE_ASSISTANT], ignore)) {
        free(v.ids); free(v.labels);
        return UENR_ERR_ALLOC;
    }
    /* assistant tokens are the only supervised span */
    if (!tv_text_mode(&v, t, assistant, 1)) {
        free(v.ids); free(v.labels);
        return UENR_ERR_ALLOC;
    }
    if (!tv_push(&v, t->roles[UENR_ROLE_END], t->roles[UENR_ROLE_END])) {
        free(v.ids); free(v.labels);
        return UENR_ERR_ALLOC;
    }
    *ids = v.ids;
    *labels = v.labels;
    *n = v.n;
    return UENR_OK;
}
