#ifndef UENR_ARENA_H
#define UENR_ARENA_H

#include <stddef.h>
#include "uenr/result.h"

/* Bump allocator with 64-byte alignment. Free is all-or-nothing (reset). */
typedef struct {
    unsigned char *base;
    size_t         capacity;
    size_t         used;
} UenrArena;

UenrResult uenr_arena_init(UenrArena *a, size_t capacity);
void       uenr_arena_reset(UenrArena *a);
void       uenr_arena_destroy(UenrArena *a);
void      *uenr_arena_alloc(UenrArena *a, size_t bytes);
void      *uenr_arena_alloc_aligned(UenrArena *a, size_t bytes, size_t align);

#endif /* UENR_ARENA_H */
