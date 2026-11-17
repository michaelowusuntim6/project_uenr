/* Malformed preference-JSONL handling: no invalid frees, no leaks, no UB.
 * Valid records are kept; malformed/oversized ones are skipped safely. */
#include "uenr/dataset.h"
#include "uenr/uenr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static int check(int cond, const char *what) {
    if (!cond) printf("FAIL %s\n", what);
    else printf("ok   %s\n", what);
    if (!cond) fails++;
    return cond;
}

int main(void) {
    const char *path = "/tmp/uenr_pairs_malformed.jsonl";
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    fputs("\n", f);
    fputs("this is not json\n", f);
    fputs("{\"prompt\":\"p\",\"chosen\":\"good\",\"rejected\":\"bad\"}\n", f);
    fputs("{\"prompt\":\"p\",\"chosen\":\"truncated\n", f);
    fputs("{\"prompt\":\"p\",\"chosen\":\"g\"}\n", f);
    fputs("{\"prompt\":1,\"chosen\":\"g\",\"rejected\":\"b\"}\n", f);
    fputs("{\"prompt\":\"p\",\"chosen\":\"\",\"rejected\":\"b\"}\n", f);
    fputs("{\"prompt\":\"p\",\"prompt\":\"q\",\"chosen\":\"g\",\"rejected\":\"b\"}\n", f);
    fputs("{\"prompt\":\"p\",\"chosen\":\"g\",\"rejected\":\"b\",\"extra\":[1,2,{\"x\":3}]}\n", f);
    fputs("{\"chosen\":\"g2\",\"rejected\":\"b2\"}\n", f);
    /* One oversized record (> 1 MiB) must be skipped without unbounded work. */
    size_t big = 1100000;
    char *line = malloc(big + 128);
    if (!line) return 1;
    strcpy(line, "{\"prompt\":\"");
    memset(line + strlen(line), 'x', big);
    strcpy(line + strlen(line), "\",\"chosen\":\"g\",\"rejected\":\"b\"}\n");
    fputs(line, f);
    free(line);
    fputs("{\"prompt\":\"p\",\"chosen\":\"good2\",\"rejected\":\"bad2\"}\n", f);
    fclose(f);

    UenrPairData d;
    UenrResult r = uenr_pairs_load(path, &d);
    check(r == UENR_OK, "malformed file still yields its valid records");
    check(d.count == 3, "keeps exactly the three valid preference pairs");
    int found_good = 0, found_good2 = 0;
    for (int i = 0; i < d.count; i++) {
        if (!strcmp(d.pairs[i].chosen, "good")) found_good = 1;
        if (!strcmp(d.pairs[i].chosen, "good2")) found_good2 = 1;
        if (!d.pairs[i].chosen[0] || !d.pairs[i].rejected[0]) fails++;
    }
    check(found_good && found_good2, "valid strings survive intact");
    uenr_pairs_free(&d);

    /* Empty and all-malformed files fail safely with UENR_ERR_INVALID. */
    f = fopen(path, "wb");
    if (!f) return 1;
    fputs("\n\nnot json\n{}\n", f);
    fclose(f);
    r = uenr_pairs_load(path, &d);
    check(r != UENR_OK && d.count == 0, "all-malformed file is rejected safely");
    uenr_pairs_free(&d);

    remove(path);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
