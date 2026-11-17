#ifndef UENR_TOKENIZER_H
#define UENR_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include "uenr/result.h"
#include "uenr/config.h"

/*
 * Byte-level BPE.
 *   ids 0..15   reserved special tokens (ids come from config)
 *   ids 16..271 the 256 raw bytes (id = 16 + byte)
 *   ids 272..   merges, in rank order (id = 272 + rank)
 */

typedef enum {
    UENR_ROLE_PAD = 0,
    UENR_ROLE_BOS,
    UENR_ROLE_EOS,
    UENR_ROLE_UNK,
    UENR_ROLE_SYSTEM,
    UENR_ROLE_USER,
    UENR_ROLE_ASSISTANT,
    UENR_ROLE_TOOL_CALL,
    UENR_ROLE_TOOL_RESULT,
    UENR_ROLE_END,
    UENR_ROLE_COUNT
} UenrTokRole;

typedef struct UenrTokenizer UenrTokenizer;

UenrResult uenr_tok_train(const UenrConfig *cfg, const char *data_path,
                          const char *out_path, const char *vocab_path,
                          const char *merges_path);
UenrResult uenr_tok_load(const UenrConfig *cfg, const char *path, UenrTokenizer **out);
void       uenr_tok_free(UenrTokenizer *t);

int  uenr_tok_vocab(const UenrTokenizer *t);
int  uenr_tok_id(const UenrTokenizer *t, UenrTokRole role);
int  uenr_tok_role(const UenrTokenizer *t, int id); /* -1 if not special */

/* Sanitize + BPE-encode a UTF-8 string. */
UenrResult uenr_tok_encode(const UenrTokenizer *t, const char *text,
                           int **ids, int *n);
/* Decode to a NUL-terminated UTF-8 buffer (caller frees). */
UenrResult uenr_tok_decode(const UenrTokenizer *t, const int *ids, int n,
                           char **text, size_t *len);

/* Sanitizer (also exposed for the data pipeline). */
UenrResult uenr_tok_sanitize(const UenrTokenizer *t, const char *in, size_t n,
                             char **out, size_t *out_n);

/* Inference prompt: <bos> [<system> text <end>] <user> text <end> <assistant> */
UenrResult uenr_chat_infer(const UenrTokenizer *t, const char *system,
                           const char *user, int **ids, int *n);
/* SFT pair: labels are next-token targets.  Entries predicting assistant
 * content or the assistant <end> marker carry that token id; all other
 * positions (role markers, user/system/tool text, the final token) carry
 * ignore_index.  `ignore` is the configured ignore index. */
UenrResult uenr_chat_sft(const UenrTokenizer *t, const char *system,
                         const char *user, const char *assistant,
                         int ignore, int **ids, int **labels, int *n);

#endif /* UENR_TOKENIZER_H */
