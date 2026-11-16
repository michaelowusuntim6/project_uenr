#ifndef UENR_SCHEDULE_H
#define UENR_SCHEDULE_H

#include "uenr/config.h"

/*
 * UENR adaptive schedule.
 *
 * Step budget (our epoch formula):
 *   target_tokens = tokens_per_param * active_params
 *   epochs        = ceil(target_tokens / corpus_tokens), clamped [min,max]
 *   total_steps   = epochs * corpus_tokens / tokens_per_step, clamped
 *                   [min_steps_before_stop, max_steps]
 *
 * LR curve:
 *   warmup: linear 0 -> lr
 *   then:   min_lr + (lr-min_lr) * 0.5*(1+cos(pi*p))
 *   adaptive factor (kind=="adaptive"): 1 + 0.2*(ema_loss/target_loss - 1),
 *   clamped to [0.75, 1.25]. Higher loss -> slightly higher LR.
 */
int uenr_schedule_plan_steps(const UenrConfig *cfg, size_t corpus_tokens);
float uenr_schedule_lr(const UenrConfig *cfg, int step, int total_steps,
                       float ema_loss);

typedef struct {
    float ema;
    float best;
    int   since_improve;
} UenrSchedState;

void uenr_sched_state_init(UenrSchedState *s);
void uenr_sched_state_update(UenrSchedState *s, const UenrConfig *cfg, float loss);
int  uenr_sched_should_stop(const UenrSchedState *s, const UenrConfig *cfg,
                            int step, int total_steps);

#endif /* UENR_SCHEDULE_H */
