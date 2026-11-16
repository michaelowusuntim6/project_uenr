#ifndef UENR_INFERENCE_H
#define UENR_INFERENCE_H

#include "uenr/config.h"
#include "uenr/result.h"

/* Generate a reply. model_path may be NULL for cfg->paths.model_out. */
UenrResult uenr_infer(const UenrConfig *cfg, const char *model_path,
                      const char *system, const char *prompt,
                      int max_new_tokens, int greedy);

/* Interactive line-by-line chat. */
UenrResult uenr_infer_chat(const UenrConfig *cfg, const char *model_path,
                           int max_new_tokens, int greedy);

#endif /* UENR_INFERENCE_H */
