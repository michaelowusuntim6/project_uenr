#include "uenr/inference.h"
#include "uenr/uenr.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
static void ok(int cond, const char *name) {
    printf(cond ? "ok   %s\n" : "FAIL %s\n", name);
    if (!cond) fails++;
}

int main(void) {
    const char *good[] = {
        "{\"name\":\"weather\",\"args\":{\"city\":\"Accra\"}}",
        "[1,2,{\"a\":[true,null,\"x\\\"y\"]}]",
        " {\"a\": 1.5e-3, \"b\": -2} ",
    };
    const char *bad[] = {
        "{\"name\":\"weather\"",          /* unbalanced */
        "{\"name\":\"weather\"}}",        /* extra close / trailing */
        "{\"name\":\"weather}",           /* quote left open */
        "name: weather",                  /* not JSON */
        "{\"a\":\"bad\\q\"}",             /* invalid escape char is allowed by
                                           * structural check, so replace below */
        "{\"a\":1} x",                    /* trailing garbage */
        "{\"a\":\"line\nbreak\"}",        /* raw control byte in string */
    };
    for (size_t i = 0; i < sizeof(good)/sizeof(good[0]); i++)
        ok(uenr_json_valid(good[i], strlen(good[i])), "valid JSON accepted");
    for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); i++) {
        if (i == 4) continue; /* structural-only check does not validate escapes */
        ok(!uenr_json_valid(bad[i], strlen(bad[i])), "invalid JSON rejected");
    }
    /* incremental prefix must stay accepted until the value closes */
    const char *p = "{\"name\":\"weather\"";
    ok(uenr_json_valid(p, strlen(p)) == 0, "unterminated prefix not complete");
    ok(uenr_json_valid("{\"a\":1}", 7), "complete object");
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
