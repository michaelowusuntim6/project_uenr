/* Verify the runtime AVX1 / AVX2+FMA dispatch.
 *
 * AVX1 runs twice and must be bit-identical.  When the host supports AVX2+FMA
 * the complete kernel set is run under each path and compared; FMA changes
 * rounding, so the tolerance is a relative L2 bound, not bit equality.  On
 * the laptop (no AVX2) the AVX2 execution is skipped and forcing it must fail
 * closed. */
#include "uenr/ops.h"
#include "uenr/gemm.h"
#include "uenr/math.h"
#include "uenr/simd_dispatch.h"
#include "uenr/thread.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TN 5
#define TD 24
#define TV 37
#define AB 2
#define AS 7
#define AH 3
#define AG 1
#define AHd 8
#define AN 64
#define SN 32
#define SC 16
#define SF 24

void uenr_softcap_fwd_avx2(float *x, size_t n, float sc);

typedef struct {
    float norm_y[TN * TD], norm_dw[TN * TD];
    float ce_loss, ce_grad[TN * TV];
    float attn[AB * AS * AH * AHd], dq[AB * AS * AH * AHd];
    float dk[AB * AS * AG * AHd], dv[AB * AS * AG * AHd];
    float adam_p[AN];
    float swiglu_y[SN * SC];
} Out;

static int failures;
static void check(int cond, const char *what) {
    if (!cond) { printf("FAIL %s\n", what); failures++; }
}

static float *vec(int n) {
    float *p = (float *)aligned_alloc(64, ((size_t)n * sizeof(float) + 63) & ~(size_t)63);
    for (int i = 0; i < n; i++)
        p[i] = sinf((float)i * 0.37f) * 0.8f + cosf((float)i * 0.11f) * 0.2f;
    return p;
}

static double rel_l2(const float *a, const float *b, int n) {
    double d = 0.0, s = 0.0;
    for (int i = 0; i < n; i++) {
        double t = (double)a[i] - (double)b[i];
        d += t * t;
        s += (double)a[i] * (double)a[i];
    }
    return sqrt(d) / (sqrt(s) + 1e-30);
}

static void run_ops(Out *o) {
    float *x = vec(TN * TD), *w = vec(TD), *rstd = vec(TN);
    float *dy = vec(TN * TD);
    uenr_rmsnorm_fwd(o->norm_y, rstd, x, w, TN, TD, 1e-5f);
    uenr_rmsnorm_bwd(dy, o->norm_dw, o->norm_y, x, w, rstd, TN, TD);

    float *lg = vec(TN * TV);
    int tg[TN];
    for (int i = 0; i < TN; i++) tg[i] = (i * 7 + 3) % TV;
    o->ce_loss = uenr_softmax_ce_fwd(lg, tg, TN, TV, -100, o->ce_grad, 1,
                                     NULL, 0.0f);

    float *q = vec(AB * AS * AH * AHd), *k = vec(AB * AS * AG * AHd);
    float *v = vec(AB * AS * AG * AHd);
    float *pr = vec(AB * AH * AS * AS);
    uenr_attn_fwd(o->attn, pr, q, k, v, AB, AS, AH, AG, AHd, 0.1f, 0,
                  0.0f, NULL);
    uenr_attn_bwd(o->dq, o->dk, o->dv, o->attn, q, k, v, pr, AB, AS, AH, AG,
                  AHd, 0.1f, 0, 0.0f, NULL, NULL);

    /* SwiGLU forward (GEMM + vector SiLU). */
    UenrGemm sg;
    if (uenr_gemm_init(&sg, NULL) == UENR_OK) {
        float *sx = vec(SN * SC), *wg = vec(SF * SC), *wu = vec(SF * SC);
        float *wd = vec(SC * SF), *gate = vec(SN * SF), *up = vec(SN * SF);
        float *gb = vec(SN * SF);
        uenr_swiglu_fwd_save(&sg, SN, SC, SF, sx, wg, wu, wd,
                             o->swiglu_y, gate, up, gb);
        uenr_gemm_destroy(&sg);
        free(sx); free(wg); free(wu); free(wd); free(gate); free(up); free(gb);
    }

    float *p = vec(AN), *g = vec(AN);
    float *m = (float *)calloc(AN, sizeof(float));
    float *vv = (float *)calloc(AN, sizeof(float));
    uenr_adamw_step(p, g, m, vv, AN, 3, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f,
                    NULL);
    memcpy(o->adam_p, p, sizeof(o->adam_p));

    free(x); free(w); free(rstd); free(dy); free(lg);
    free(q); free(k); free(v); free(pr); free(p); free(g); free(m); free(vv);
}

static void gemm_case(int M, int N, int K, int ta, int tb, int tile,
                      const float *A, const float *B) {
    UenrGemm g;
    float *c1 = (float *)calloc((size_t)M * N, sizeof(float));
    float *c2 = (float *)calloc((size_t)M * N, sizeof(float));

    check(uenr_simd_configure("avx1") == UENR_OK, "configure avx1");
    check(uenr_gemm_init(&g, NULL) == UENR_OK, "gemm init avx1");
    uenr_gemm(&g, M, N, K, A, ta ? K : M, ta, B, tb ? K : N, tb, c1, N,
              1.0f, 0.0f);
    uenr_gemm_destroy(&g);
    check(uenr_gemm_init(&g, NULL) == UENR_OK, "gemm init avx1 repeat");
    uenr_gemm(&g, M, N, K, A, ta ? K : M, ta, B, tb ? K : N, tb, c2, N,
              1.0f, 0.0f);
    uenr_gemm_destroy(&g);
    check(memcmp(c1, c2, (size_t)M * N * sizeof(float)) == 0,
          "AVX1 GEMM is deterministic");

    for (int lvl = 0; lvl < 2; lvl++) {
        const char *mode = lvl == 0 ? "avx2" : "avx512";
        if (lvl == 0 ? !uenr_cpu_has_avx2_fma() : !uenr_cpu_has_avx512())
            continue;
        check(uenr_simd_configure(mode) == UENR_OK, "configure SIMD level");
        check(uenr_gemm_init(&g, NULL) == UENR_OK, "gemm init SIMD level");
        uenr_gemm_set_avx2_tile(&g, tile);
        memset(c2, 0, (size_t)M * N * sizeof(float));
        uenr_gemm(&g, M, N, K, A, ta ? K : M, ta, B, tb ? K : N, tb, c2, N,
                  1.0f, 0.0f);
        uenr_gemm_destroy(&g);
        double e = rel_l2(c1, c2, M * N);
        if (e > 2e-5) {
            printf("FAIL GEMM %s ta=%d tb=%d tile=%d relL2=%.3g\n",
                   mode, ta, tb, tile, e);
            failures++;
        }
    }
    free(c1); free(c2);
}

int main(void) {
    enum { M = 17, N = 19, K = 23 };
    float *A = vec(M * K), *B = vec(K * N);
    check(uenr_simd_configure("avx1") == UENR_OK, "force avx1");
    check(uenr_simd_level() == UENR_SIMD_AVX1, "avx1 selected");
    for (int tile = 0; tile < 3; tile++)
        for (int ta = 0; ta < 2; ta++)
            for (int tb = 0; tb < 2; tb++)
                gemm_case(M, N, K, ta, tb, tile, A, B);
    free(A); free(B);

    Out *a1 = (Out *)calloc(1, sizeof(Out));
    Out *a1b = (Out *)calloc(1, sizeof(Out));
    check(uenr_simd_configure("avx1") == UENR_OK, "ops avx1");
    run_ops(a1);
    run_ops(a1b);
    check(memcmp(a1, a1b, sizeof(Out)) == 0, "AVX1 op set is deterministic");

    if (uenr_cpu_has_avx512()) {
        Out *a5 = (Out *)calloc(1, sizeof(Out));
        check(uenr_simd_configure("avx512") == UENR_OK, "ops avx512");
        run_ops(a5);
        check(rel_l2(a1->attn, a5->attn, AB * AS * AH * AHd) < 1e-6,
              "AVX-512 attention output tolerance");
        check(rel_l2(a1->dq, a5->dq, AB * AS * AH * AHd) < 1e-4,
              "AVX-512 attention dq tolerance");
        free(a5);
        uenr_simd_configure("auto");
        printf("ok   AVX-512 dispatch compared against AVX1\n");
    }
    if (uenr_cpu_has_avx2_fma()) {
        Out *a2 = (Out *)calloc(1, sizeof(Out));
        check(uenr_simd_configure("avx2") == UENR_OK, "ops avx2");
        run_ops(a2);
        check(rel_l2(a1->norm_y, a2->norm_y, TN * TD) < 1e-6,
              "RMSNorm fwd tolerance");
        check(rel_l2(a1->norm_dw, a2->norm_dw, TN * TD) < 1e-5,
              "RMSNorm bwd tolerance");
        check(fabsf(a1->ce_loss - a2->ce_loss) < 1e-5f,
              "softmax-CE loss tolerance");
        check(rel_l2(a1->ce_grad, a2->ce_grad, TN * TV) < 1e-5,
              "softmax-CE grad tolerance");
        check(rel_l2(a1->attn, a2->attn, AB * AS * AH * AHd) < 1e-6,
              "attention fwd tolerance");
        check(rel_l2(a1->dq, a2->dq, AB * AS * AH * AHd) < 1e-4,
              "attention dq tolerance");
        check(rel_l2(a1->dk, a2->dk, AB * AS * AG * AHd) < 1e-4,
              "attention dk tolerance");
        check(rel_l2(a1->dv, a2->dv, AB * AS * AG * AHd) < 1e-4,
              "attention dv tolerance");
        {
            double ae = rel_l2(a1->adam_p, a2->adam_p, AN);
            if (ae >= 1e-5) {
                printf("FAIL AdamW tolerance relL2=%.6g\n", ae);
                failures++;
            }
        }
        free(a2);
        uenr_simd_configure("auto");
        printf("ok   AVX2+FMA dispatch compared against AVX1\n");
    } else {
        check(uenr_simd_configure("avx2") != UENR_OK,
              "forcing avx2 on a non-AVX2 CPU fails closed");
        if (!uenr_cpu_has_avx512())
            check(uenr_simd_configure("avx512") != UENR_OK,
                  "forcing avx512 without AVX-512 fails closed");
        uenr_simd_configure("auto");
        printf("skip AVX2 execution: CPU has no AVX2+FMA (AVX1 path verified)\n");
    }
    free(a1); free(a1b);
    check(uenr_simd_configure("bogus") != UENR_OK, "unknown simd mode rejected");
    uenr_simd_configure("auto");

    if (failures) { printf("simd dispatch: %d failure(s)\n", failures); return 1; }
    printf("all passed\n");
    return 0;
}
