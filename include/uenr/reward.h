#ifndef UENR_REWARD_H
#define UENR_REWARD_H

#include "uenr/config.h"
#include "uenr/model.h"
#include "uenr/result.h"

typedef struct {
    int    C;
    float  bias;
    float *w;
} UenrRewardModel;

UenrResult uenr_reward_train(const UenrConfig *cfg);
UenrResult uenr_reward_load(UenrRewardModel *r, const char *path);
void       uenr_reward_free(UenrRewardModel *r);

/* Score ids[pn..n) from a frozen base model's final hidden state. */
float uenr_reward_score_seq(UenrModel *base, UenrGemm *g, UenrActs *a,
                            const UenrRewardModel *r,
                            const int *ids, int n, int pn);

#endif /* UENR_REWARD_H */
