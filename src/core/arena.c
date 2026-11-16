#include "uenr/arena.h"

#include <stdlib.h>
#include <string.h>

#define UENR_ARENA_DEFAULT_ALIGN 64u

UenrResult uenr_arena_init(UenrArena *a, size_t capacity) {
    if (a == NULL || capacity == 0) return UENR_ERR_INVALID;
    a->base = (unsigned char *)aligned_alloc(UENR_ARENA_DEFAULT_ALIGN,
                                             capacity);
    if (a->base == NULL) return UENR_ERR_ALLOC;
    a->capacity = capacity;
    a->used = 0;
    return UENR_OK;
}

void uenr_arena_reset(UenrArena *a) {
    if (a != NULL) a->used = 0;
}

void uenr_arena_destroy(UenrArena *a) {
    if (a == NULL) return;
    free(a->base);
    a->base = NULL;
    a->capacity = 0;
    a->used = 0;
}

void *uenr_arena_alloc_aligned(UenrArena *a, size_t bytes, size_t align) {
    if (a == NULL || a->base == NULL || bytes == 0) return NULL;
    if (align == 0 || (align & (align - 1)) != 0) return NULL;
    size_t p = (a->used + align - 1) & ~(align - 1);
    if (p > a->capacity || bytes > a->capacity - p) return NULL;
    a->used = p + bytes;
    return a->base + p;
}

void *uenr_arena_alloc(UenrArena *a, size_t bytes) {
    return uenr_arena_alloc_aligned(a, bytes, UENR_ARENA_DEFAULT_ALIGN);
}
