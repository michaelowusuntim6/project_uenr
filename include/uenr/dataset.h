#ifndef UENR_DATASET_H
#define UENR_DATASET_H

#include <stddef.h>
#include <stdint.h>
#include "uenr/config.h"
#include "uenr/rng.h"
#include "uenr/result.h"
#include "uenr/tokenizer.h"

/* Pretraining token stream (uint16). */
typedef struct {
    uint16_t *tokens;
    size_t    count;
    int       vocab;
    void     *map;
    size_t    map_size;
} UenrTokenData;

UenrResult uenr_data_prepare(const UenrConfig *cfg, const UenrTokenizer *tok);
/* Prepare one phase (pretrain, sft, lora, dpo, rl/grpo/ppo, reward, eval) or
 * "all".  Phase-specific outputs are written to the configured paths.* files. */
UenrResult uenr_data_prepare_phase(const UenrConfig *cfg, const UenrTokenizer *tok,
                                   const char *phase);
UenrResult uenr_data_load(const char *path, UenrTokenData *out);
void       uenr_data_free(UenrTokenData *d);
int        uenr_data_batch(const UenrTokenData *d, UenrRng *rng, int seq,
                           int nseq, int *inputs, int *targets);

/* SFT samples: chat-templated ids + assistant-only labels. */
typedef struct {
    int *ids;
    int *labels;
    int  n;
} UenrSample;

typedef struct {
    UenrSample *samples;
    int         count;
} UenrSFTData;

UenrResult uenr_sft_load(const UenrConfig *cfg, const UenrTokenizer *tok,
                         const char *path, UenrSFTData *out);
void       uenr_sft_free(UenrSFTData *d);
int        uenr_sft_batch(const UenrSFTData *d, UenrRng *rng, int seq,
                          int nseq, int *inputs, int *labels,
                          int pad_id, int ignore_index);

/* DPO preference pairs. */
typedef struct {
    char *prompt;
    char *chosen;
    char *rejected;
} UenrPair;

typedef struct {
    UenrPair *pairs;
    int       count;
} UenrPairData;

UenrResult uenr_pairs_load(const char *path, UenrPairData *out);
/* RL prompt-only JSONL: {"prompt":"..."} per line.  chosen/rejected are
 * empty strings so the UenrPairData container can be reused. */
UenrResult uenr_rl_load(const char *path, UenrPairData *out);
/* Offline conversion helper for workflow dataset bootstrap (pure C).
 * Supported: tinystories_instruct -> prompt_completion; prompt_completion ->
 * prefs | prompts.  Returns 1 on success. */
int uenr_convert_file(const char *from, const char *to,
                      const char *in_path, const char *out_path, int limit);
void       uenr_pairs_free(UenrPairData *d);

#endif /* UENR_DATASET_H */
