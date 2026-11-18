#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "uenr/uenr.h"

#define MENU_N 26

static void chdir_to_exe(void) {
    char path[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) return;
    path[n] = '\0';
    char *slash = strrchr(path, '/');
    if (!slash) return;
    *slash = '\0';
    if (chdir(path) != 0) { /* keep cwd */ }
}

static void pause_enter(void) {
    printf("\nPress Enter to return to the menu...");
    fflush(stdout);
    int c;
    while ((c = getchar()) != '\n' && c != EOF) { }
}

static int run(const char *cmd) {
    printf("\n$ %s\n", cmd);
    fflush(stdout);
    int rc = system(cmd);
    if (rc != 0) printf("\n[exit status %d]\n", rc);
    return rc;
}

static int ask_steps(void) {
    char buf[64];
    printf("steps [Enter = config default]: ");
    fflush(stdout);
    if (!fgets(buf, sizeof(buf), stdin)) return 0;
    buf[strcspn(buf, "\r\n")] = '\0';
    if (!buf[0]) return 0;
    int n = atoi(buf);
    return n > 0 ? n : 0;
}

static int ask_yn(const char *what) {
    char buf[16];
    printf("%s [y/N]: ", what);
    fflush(stdout);
    if (!fgets(buf, sizeof(buf), stdin)) return 0;
    return buf[0] == 'y' || buf[0] == 'Y';
}

static int ask_line(const char *what, char *out, size_t cap) {
    printf("%s: ", what);
    fflush(stdout);
    if (!fgets(out, (int)cap, stdin)) return 0;
    out[strcspn(out, "\r\n")] = '\0';
    return out[0] != '\0';
}

static void kv_report(void) {
    UenrConfig cfg;
    if (uenr_config_load(NULL, &cfg) != UENR_OK) { printf("cannot load config\n"); return; }
    int nslots = uenr_active_kv_layers(&cfg);
    int KV = cfg.model.kv_heads * cfg.model.head_dim;
    int local = 0, global = 0;
    for (int l = 0; l < nslots; l++)
        if (uenr_layer_is_sliding(&cfg, l)) local++; else global++;
    printf("\n-- KV cache --\n");
    printf("active slots %d (local %d, global %d), W=%d, kv=%d dims\n",
           nslots, local, global, cfg.model.sliding_window, KV);
    long long seqs[2] = { cfg.model.max_seq, 1000000 };
    for (int i = 0; i < 2; i++) {
        long long S = seqs[i];
        long long rows = 0;
        for (int l = 0; l < nslots; l++) {
            if (uenr_layer_is_sliding(&cfg, l))
                rows += (cfg.model.sliding_window > 0 &&
                         cfg.model.sliding_window < S) ? cfg.model.sliding_window : S;
            else if (cfg.infer.kv_evict &&
                     cfg.infer.kv_sinks + cfg.infer.kv_evict_window < S)
                rows += cfg.infer.kv_sinks + cfg.infer.kv_evict_window;
            else rows += S;
        }
        double f32 = (double)rows * KV * 8.0;   /* K+V, 4 bytes each */
        double f16 = (double)rows * KV * 4.0;   /* K+V, 2 bytes each */
        double i8  = (double)rows * KV * 2.0;   /* K+V, 1 byte each */
        double q4  = (double)rows * KV * 1.0 +
                     (double)rows * cfg.model.kv_heads * 8.0; /* 4-bit + fp32 scales */
        printf("S=%lld: rows %lld | FP32 %.2f MB | FP16 %.2f MB | INT8 %.2f MB | Q4 %.2f MB\n",
               S, rows, f32 / 1e6, f16 / 1e6, i8 / 1e6, q4 / 1e6);
        printf("       per-token FP32 %.0f B | FP16 %.0f B | INT8 %.0f B | Q4 %.0f B\n",
               (double)KV * 8.0, (double)KV * 4.0, (double)KV * 2.0,
               (double)KV * 1.0 + (double)cfg.model.kv_heads * 8.0);
    }
    printf("Set inference.kv_fp16/kv_int8/kv_q4 to switch decode caches "
           "(priority int8 > q4 > fp16).\n");
}

static void print_menu(void) {
    printf("\n");
    printf("==============================================================\n");
    printf("  UENR — tiny sparse-MoE LLM, dependency-free C, Ivy Bridge\n");
    printf("--------------------------------------------------------------\n");
    printf("   1. Check environment (compilers, CPU, files)\n");
    printf("   2. Build everything\n");
    printf("   3. Run the test suite\n");
    printf("   4. Full check (build + tests + benches + ISA)\n");
    printf("   5. Benchmarks (GEMM vs measured roof)\n");
    printf("   6. Train the BPE tokenizer\n");
    printf("   7. Prepare dataset (train/val/SFT/pairs)\n");
    printf("   8. Pretrain\n");
    printf("   9. Supervised finetune (SFT)\n");
    printf("  10. LoRA finetune\n");
    printf("  11. Train pairwise reward model\n");
    printf("  12. DPO preference tuning\n");
    printf("  13. PPO policy-gradient tuning\n");
    printf("  14. GRPO group-relative tuning\n");
    printf("  15. Inference / chat\n");
    printf("  16. Batched inference from a prompt file\n");
    printf("  17. Evaluate (val loss/ppl + generation metrics)\n");
    printf("  18. KV-cache size calculator (FP32/FP16/INT8/Q4 + eviction)\n");
    printf("  19. View configuration\n");
    printf("  20. View documentation\n");
    printf("  21. Project status / checklist\n");
    printf("  22. Clean build artifacts\n");
    printf("  23. End-to-end smoke test (temporary directory)\n");
    printf("  24. Cloud bench dispatch (gh workflow run mode=bench)\n");
    printf("  25. Cloud smoke dispatch (gh workflow run mode=smoke)\n");
    printf("  26. Exit\n");
    printf("==============================================================\n");
    printf("choice> ");
    fflush(stdout);
}

static void env_check(void) {
    printf("\n-- toolchain --\n");
    system("command -v gcc && gcc --version | head -1");
    system("command -v make && make --version | head -1");
    system("command -v git && git --version");
    system("command -v python3 && python3 --version");
    printf("\n-- cpu --\n");
    system("grep -m1 'model name' /proc/cpuinfo");
    system("nproc");
    printf("\n-- project files --\n");
    system("ls -la config/uenr.toml config/scale.toml README.md docs/DETAILED.md docs/STATUS.md docs/TODO.md 2>/dev/null");
    system("ls -la artifacts/ data/*.bin checkpoints/*.bin 2>/dev/null | tail -12");

    printf("\nbuild-essential / gcc present? ");
    if (system("command -v gcc >/dev/null 2>&1 && command -v make >/dev/null 2>&1") == 0) {
        printf("yes\n");
    } else {
        printf("no\n");
        printf("Install with:  sudo apt-get update && sudo apt-get install -y build-essential\n");
        printf("Run that now? [y/N]: ");
        fflush(stdout);
        char b[16];
        if (fgets(b, sizeof(b), stdin) && (b[0] == 'y' || b[0] == 'Y'))
            run("sudo apt-get update && sudo apt-get install -y build-essential");
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    chdir_to_exe();
    for (;;) {
        print_menu();
        char line[64];
        if (!fgets(line, sizeof(line), stdin)) break;
        int choice = atoi(line);
        if (choice < 1 || choice > MENU_N) {
            printf("invalid choice\n");
            continue;
        }
        char cmd[1024];
        switch (choice) {
        case 1:
            env_check();
            break;
        case 2:
            run("make -j2");
            break;
        case 3:
            run("make test");
            break;
        case 4:
            run("make check");
            break;
        case 5:
            run("./uenr bench");
            run("make bench");
            break;
        case 6:
            run("./uenr tok-train");
            break;
        case 7:
            run("./uenr prepare");
            break;
        case 8: {
            int n = ask_steps();
            int rs = ask_yn("Resume newest checkpoint");
            if (n) snprintf(cmd, sizeof(cmd), "./uenr train --steps %d%s", n, rs ? " --resume" : "");
            else snprintf(cmd, sizeof(cmd), "./uenr train%s", rs ? " --resume" : "");
            run(cmd);
            break;
        }
        case 9: {
            /* SFT always starts from the pretrained checkpoint; it does not
             * implement --resume, so do not offer it here. */
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr sft --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr sft");
            run(cmd);
            break;
        }
        case 10: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr lora --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr lora");
            run(cmd);
            break;
        }
        case 11:
            run("./uenr train-rm");
            break;
        case 12: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr dpo --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr dpo");
            run(cmd);
            break;
        }
        case 13: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr ppo --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr ppo");
            run(cmd);
            break;
        }
        case 14: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr grpo --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr grpo");
            run(cmd);
            break;
        }
        case 15: {
            /* Call directly so the menu and the chat share one stdin buffer. */
            UenrConfig cfg;
            if (uenr_config_load(NULL, &cfg) == UENR_OK)
                uenr_infer_chat(&cfg, NULL, 0, 0);
            else
                printf("cannot load config\n");
            break;
        }
        case 16: {
            char file[400], nbuf[32];
            if (!ask_line("prompt file", file, sizeof(file))) {
                printf("no file given\n");
                break;
            }
            int n = 0;
            if (ask_line("tokens per prompt [Enter = default]", nbuf, sizeof(nbuf)))
                n = atoi(nbuf);
            if (n > 0) snprintf(cmd, sizeof(cmd), "./uenr infer --batch %s --tokens %d --greedy", file, n);
            else snprintf(cmd, sizeof(cmd), "./uenr infer --batch %s --greedy", file);
            run(cmd);
            break;
        }
        case 17:
            run("./uenr eval");
            break;
        case 18:
            kv_report();
            break;
        case 19:
            run("./uenr config");
            run("cat config/uenr.toml");
            break;
        case 20:
            run("ls -1 docs 2>/dev/null || echo 'no docs directory yet'");
            break;
        case 21:
            run("sed -n '1,40p' docs/CHECK_LIST.md");
            run("sed -n '1,30p' docs/STATUS.md");
            break;
        case 22:
            printf("Remove build artifacts and binaries? [y/N]: ");
            fflush(stdout);
            if (fgets(line, sizeof(line), stdin) && (line[0] == 'y' || line[0] == 'Y'))
                run("make clean");
            break;
        case 23:
            run("make smoke");
            break;
        case 24:
            run("gh workflow run train.yml --ref master -f mode=bench -f bootstrap_datasets=true");
            break;
        case 25:
            run("gh workflow run train.yml --ref master -f mode=smoke -f bootstrap_datasets=true");
            break;
        case 26:
            printf("bye.\n");
            return 0;
        }
        pause_enter();
    }
    return 0;
}
