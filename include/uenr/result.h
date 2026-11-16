#ifndef UENR_RESULT_H
#define UENR_RESULT_H

/* Rust-style Result expressed as a C enum. Zero-cost, no allocation. */
typedef enum {
    UENR_OK = 0,
    UENR_ERR_ALLOC,
    UENR_ERR_IO,
    UENR_ERR_INVALID,
    UENR_ERR_BOUNDS,
    UENR_ERR_SHAPE,
    UENR_ERR_THREAD,
} UenrResult;

static inline int uenr_ok(UenrResult r) { return r == UENR_OK; }

#define UENR_TRY(expr) do { UenrResult _r = (expr); if (_r != UENR_OK) return _r; } while (0)

#endif /* UENR_RESULT_H */
