#ifndef UENR_TRAIN_H
#define UENR_TRAIN_H

#include "uenr/config.h"
#include "uenr/model.h"
#include "uenr/result.h"

UenrResult uenr_train_pretrain(const UenrConfig *cfg);
UenrResult uenr_train_sft(const UenrConfig *cfg);
UenrResult uenr_train_lora(const UenrConfig *cfg);
UenrResult uenr_train_dpo(const UenrConfig *cfg);
UenrResult uenr_train_ppo(const UenrConfig *cfg);
UenrResult uenr_train_grpo(const UenrConfig *cfg);

/* Newest header-valid checkpoints/step_XXXXXX.bin in `dir`, selected by the
 * step stored in the checkpoint header (not the filename).  Corrupt files are
 * skipped.  Returns the header step, or -1 if none is usable. */
int uenr_latest_checkpoint(const char *dir, char *out, size_t cap, int *step_out);

/* DPO objective helpers (exposed for mathematical regression tests).
 * delta = (logp_pi(chosen) - logp_ref(chosen)) -
 *         (logp_pi(rejected) - logp_ref(rejected)).
 * loss = -log sigmoid(beta * delta).  The gradient with respect to the
 * policy log-probabilities is returned in *g_chosen and *g_rejected. */
float uenr_dpo_loss(float delta, float beta);
void  uenr_dpo_logp_grad(float delta, float beta,
                         float *g_chosen, float *g_rejected);

/* PPO clipped-surrogate derivative with respect to the policy log-probability
 * of the sampled completion.  `active` reports whether the unclipped term is
 * the minimum (non-zero gradient).  See src/train/align.c for the derivation. */
float uenr_ppo_surrogate_coef(float advantage, float ratio, float clip,
                              int *active);

/* Alignment checkpoint loaders (also used by tests).  `load_required` fails
 * with an actionable error when the path is missing or invalid instead of
 * silently starting from random weights.  `load_reference` resolves
 * dpo.ref_checkpoint, explicitly falls back to paths.model_out when the
 * configured reference is absent, validates finiteness, and fails closed when
 * no usable reference exists. */
UenrResult uenr_align_load_required(UenrModel *m, const UenrConfig *cfg,
                                    const char *path, int with_opt,
                                    const char *what);
UenrResult uenr_align_load_reference(UenrModel *ref, const UenrConfig *cfg);

#endif /* UENR_TRAIN_H */
