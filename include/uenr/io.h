#ifndef UENR_IO_H
#define UENR_IO_H

#include <stddef.h>
#include <stdint.h>
#include "uenr/result.h"

/* Whole-file read into malloc'd memory (caller frees). */
UenrResult uenr_read_file(const char *path, void **data, size_t *size);

/* Write via "<path>.tmp" then rename(), so a crash never leaves a torn file. */
UenrResult uenr_write_file_atomic(const char *path, const void *data, size_t size);

/* Read-only mmap. */
UenrResult uenr_mmap_file(const char *path, void **data, size_t *size);
void       uenr_munmap(void *data, size_t size);

UenrResult uenr_mkdirs(const char *path);
UenrResult uenr_make_parent(const char *path);
/* Transparent huge-page hint for large allocations (no-op below 2 MiB). */
extern int uenr_huge_pages_enabled;
void uenr_madvise_huge(void *p, size_t bytes);
UenrResult uenr_file_exists(const char *path);
int64_t    uenr_file_size(const char *path); /* -1 on error */

/* Little-endian helpers (x86 is LE; kept explicit for the on-disk format). */
static inline uint16_t uenr_le16(uint16_t x) { return x; }
static inline uint32_t uenr_le32(uint32_t x) { return x; }
static inline uint64_t uenr_le64(uint64_t x) { return x; }

#endif /* UENR_IO_H */
