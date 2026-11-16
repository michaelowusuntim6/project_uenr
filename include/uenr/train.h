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

#endif /* UENR_TRAIN_H */
