#ifndef UENR_EVAL_H
#define UENR_EVAL_H

#include "uenr/config.h"
#include "uenr/result.h"

UenrResult uenr_eval(const UenrConfig *cfg, const char *model_path,
                     const char *split, int batches, int seq, int nseq);

#endif /* UENR_EVAL_H */
