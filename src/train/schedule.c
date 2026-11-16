#include "uenr/schedule.h"
#include "uenr/math.h"

#include <string.h>

#include <stddef.h>

int uenr_schedule_plan_steps(const UenrConfig *cfg, size_t corpus_tokens) {
    if (!cfg || corpus_tokens == 0) return cfg ? cfg->sched.min_steps_before_stop : 1;
    double target = (double)cfg->sched.tokens_per_param * (double)uenr_active_params(cfg);
    double epochs = target / (double)corpus_tokens;
    int ep = (int)(epochs + 0.999999);
    if (ep < cfg->sched.min_epochs) ep = cfg->sched.min_epochs;
    if (ep > cfg->sched.max_epochs) ep = cfg->sched.max_epochs;

    double steps = (double)ep * (double)corpus_tokens /
                   (double)(cfg->train.batch_size * cfg->train.seq_len);
    int st = (int)(steps + 0.5);
    if (st < cfg->sched.min_steps_before_stop) st = cfg->sched.min_steps_before_stop;
    if (st > cfg->train.max_steps) st = cfg->train.max_steps;
    if (st < 1) st = 1;
    return st;
}

float uenr_schedule_lr(const UenrConfig *cfg, int step, int total_steps,
                       float ema_loss) {
    if (!cfg || total_steps <= 0) return 0.0f;
    if (step <= 0) return 0.0f;
    if (step <= cfg->train.warmup_steps && cfg->train.warmup_steps > 0)
        return cfg->train.lr * (float)step / (float)cfg->train.warmup_steps;

    int denom = total_steps - cfg->train.warmup_steps;
    if (denom <= 0) return cfg->train.lr;
    float p = (float)(step - cfg->train.warmup_steps) / (float)denom;
    if (p > 1.0f) p = 1.0f;
    const float pi = 3.141592653589793f;
    float cosine = 0.5f * (1.0f + uenr_cosf(pi * p));
    float base = cfg->train.min_lr + (cfg->train.lr - cfg->train.min_lr) * cosine;

    if (!strcmp(cfg->sched.kind, "adaptive") && ema_loss > 0.0f &&
        cfg->sched.target_loss > 0.0f) {
        float ratio = ema_loss / cfg->sched.target_loss;
        float scale = 1.0f + 0.2f * (ratio - 1.0f);
        if (scale < 0.75f) scale = 0.75f;
        if (scale > 1.25f) scale = 1.25f;
        base *= scale;
    }
    if (base < 0.0f) base = 0.0f;
    return base;
}

void uenr_sched_state_init(UenrSchedState *s) {
    if (!s) return;
    s->ema = 0.0f;
    s->best = 3.402823466e+38f;
    s->since_improve = 0;
}

void uenr_sched_state_update(UenrSchedState *s, const UenrConfig *cfg, float loss) {
    if (!s || !cfg) return;
    if (s->ema <= 0.0f) s->ema = loss;
    else s->ema = cfg->sched.ema_beta * s->ema + (1.0f - cfg->sched.ema_beta) * loss;

    if (s->ema < s->best - cfg->sched.plateau_delta) {
        s->best = s->ema;
        s->since_improve = 0;
    } else {
        s->since_improve++;
    }
}

int uenr_sched_should_stop(const UenrSchedState *s, const UenrConfig *cfg,
                           int step, int total_steps) {
    if (!s || !cfg) return 1;
    if (step >= total_steps) return 1;
    if (step < cfg->sched.min_steps_before_stop) return 0;
    if (s->ema > 0.0f && s->ema <= cfg->sched.target_loss) return 1;
    if (s->since_improve >= cfg->sched.plateau_window) return 1;
    return 0;
}
