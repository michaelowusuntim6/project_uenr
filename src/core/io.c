#include "uenr/io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int uenr_huge_pages_enabled = 0;

void uenr_madvise_huge(void *p, size_t bytes) {
#if defined(MADV_HUGEPAGE)
    if (p && uenr_huge_pages_enabled && bytes >= (2u << 20))
        madvise(p, bytes, MADV_HUGEPAGE);
#else
    (void)p; (void)bytes;
#endif
}

static void mkdir_parents(const char *path) {
    char tmp[1024];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return;
    memcpy(tmp, path, n + 1);
    char *slash = strrchr(tmp, '/');
    if (!slash) return;
    *slash = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0777); *p = '/'; }
    }
    mkdir(tmp, 0777);
}

UenrResult uenr_mkdirs(const char *path) {
    if (!path || !*path) return UENR_ERR_INVALID;
    char tmp[1024];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return UENR_ERR_INVALID;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; if (mkdir(tmp, 0777) != 0 && errno != EEXIST) return UENR_ERR_IO; *p = '/'; }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) return UENR_ERR_IO;
    return UENR_OK;
}

UenrResult uenr_read_file(const char *path, void **data, size_t *size) {
    if (!path || !data || !size) return UENR_ERR_INVALID;
    FILE *f = fopen(path, "rb");
    if (!f) return UENR_ERR_IO;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return UENR_ERR_IO; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return UENR_ERR_IO; }
    rewind(f);
    void *buf = malloc((size_t)n ? (size_t)n : 1);
    if (!buf) { fclose(f); return UENR_ERR_ALLOC; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return UENR_ERR_IO; }
    *data = buf;
    *size = (size_t)n;
    return UENR_OK;
}

UenrResult uenr_write_file_atomic(const char *path, const void *data, size_t size) {
    if (!path || (!data && size)) return UENR_ERR_INVALID;
    char tmp[1024];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp))
        return UENR_ERR_INVALID;
    mkdir_parents(path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return UENR_ERR_IO;
    if (size && fwrite(data, 1, size, f) != size) { fclose(f); remove(tmp); return UENR_ERR_IO; }
    if (fflush(f) != 0) { fclose(f); remove(tmp); return UENR_ERR_IO; }
    if (fsync(fileno(f)) != 0) { fclose(f); remove(tmp); return UENR_ERR_IO; }
    if (fclose(f) != 0) { remove(tmp); return UENR_ERR_IO; }
    if (rename(tmp, path) != 0) { remove(tmp); return UENR_ERR_IO; }
    return UENR_OK;
}

UenrResult uenr_mmap_file(const char *path, void **data, size_t *size) {
    if (!path || !data || !size) return UENR_ERR_INVALID;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return UENR_ERR_IO;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return UENR_ERR_IO; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return UENR_ERR_IO;
    *data = p;
    *size = (size_t)st.st_size;
    return UENR_OK;
}

void uenr_munmap(void *data, size_t size) {
    if (data && size) munmap(data, size);
}

UenrResult uenr_file_exists(const char *path) {
    if (!path) return UENR_ERR_INVALID;
    return access(path, F_OK) == 0 ? UENR_OK : UENR_ERR_IO;
}

int64_t uenr_file_size(const char *path) {
    struct stat st;
    if (!path || stat(path, &st) != 0) return -1;
    return (int64_t)st.st_size;
}
