#include "uenr/config.h"

#include <stdio.h>

int main(void) {
    UenrConfig c;
    UenrResult r = uenr_config_load("config/uenr.toml", &c);
    if (!uenr_ok(r)) {
        printf("FAIL load err=%d\n", (int)r);
        return 1;
    }
    int fails = 0;
#define EXPECT(cond) do { if (!(cond)) { printf("FAIL %s\n", #cond); fails++; } } while (0)
    EXPECT(c.model.hidden == 192);
    EXPECT(c.model.layers == 4);
    EXPECT(c.model.heads == 6);
    EXPECT(c.model.heads * c.model.head_dim == c.model.hidden);
    EXPECT(c.moe.enabled == 1);
    EXPECT(c.moe.num_experts == 24);
    EXPECT(c.moe.experts_per_tok == 1);
    EXPECT(c.train.batch_size > 0);
    EXPECT(c.lora.n_targets == 4);
    EXPECT(c.sched.kind[0] != '\0');
#undef EXPECT
    if (fails == 0) printf("config ok\n");
    return fails != 0;
}
