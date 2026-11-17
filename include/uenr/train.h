#ifndef UENR_TRAIN_H
#define UENR_TRAIN_H

#include "uenr/config.h"
#include "uenr/result.h"

UenrResult uenr_train_pretrain(const UenrConfig *cfg);
UenrResult uenr_train_sft(const UenrConfig *cfg);
UenrResult uenr_train_lora(const UenrConfig *cfg);
UenrResult uenr_train_dpo(const UenrConfig *cfg);
UenrResult uenr_train_ppo(const UenrConfig *cfg);
UenrResult uenr_train_grpo(const UenrConfig *cfg);

/* DPO objective helpers (exposed for mathematical regression tests).
 * delta = (logp_pi(chosen) - logp_ref(chosen)) -
 *         (logp_pi(rejected) - logp_ref(rejected)).
 * loss = -log sigmoid(beta * delta).  The gradient with respect to the
 * policy log-probabilities is returned in *g_chosen and *g_rejected. */
float uenr_dpo_loss(float delta, float beta);
void  uenr_dpo_logp_grad(float delta, float beta,
                         float *g_chosen, float *g_rejected);

#endif /* UENR_TRAIN_H */
