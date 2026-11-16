#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "uenr/uenr.h"

#define MENU_N 17

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

static void print_menu(void) {
    printf("\n");
    printf("==============================================================\n");
    printf("  UENR — tiny sparse-MoE LLM, dependency-free C, Ivy Bridge\n");
    printf("--------------------------------------------------------------\n");
    printf("   1. Check environment (compilers, CPU, files)\n");
    printf("   2. Build everything\n");
    printf("   3. Run the test suite\n");
    printf("   4. Benchmarks (GEMM vs measured roof)\n");
    printf("   5. Train the BPE tokenizer\n");
    printf("   6. Prepare dataset (train/val/SFT/pairs)\n");
    printf("   7. Pretrain\n");
    printf("   8. Supervised finetune (SFT)\n");
    printf("   9. LoRA finetune\n");
    printf("  10. DPO preference tuning\n");
    printf("  11. PPO policy-gradient tuning\n");
    printf("  12. GRPO group-relative tuning\n");
    printf("  13. Inference / chat\n");
    printf("  14. View configuration\n");
    printf("  15. View documentation\n");
    printf("  16. Clean build artifacts\n");
    printf("  17. Exit\n");
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
    system("ls -la config/uenr.toml README.md AUDIT.md TODO.md 2>/dev/null");
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
        char cmd[512];
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
            run("./uenr bench");
            run("make bench");
            break;
        case 5:
            run("./uenr tok-train");
            break;
        case 6:
            run("./uenr prepare");
            break;
        case 7: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr train --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr train");
            run(cmd);
            break;
        }
        case 8: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr sft --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr sft");
            run(cmd);
            break;
        }
        case 9: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr lora --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr lora");
            run(cmd);
            break;
        }
        case 10: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr dpo --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr dpo");
            run(cmd);
            break;
        }
        case 11: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr ppo --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr ppo");
            run(cmd);
            break;
        }
        case 12: {
            int n = ask_steps();
            if (n) snprintf(cmd, sizeof(cmd), "./uenr grpo --steps %d", n);
            else snprintf(cmd, sizeof(cmd), "./uenr grpo");
            run(cmd);
            break;
        }
        case 13: {
            /* Call directly so the menu and the chat share one stdin buffer. */
            UenrConfig cfg;
            if (uenr_config_load(NULL, &cfg) == UENR_OK)
                uenr_infer_chat(&cfg, NULL, 0, 0);
            else
                printf("cannot load config\n");
            break;
        }
        case 14:
            run("./uenr config");
            run("cat config/uenr.toml");
            break;
        case 15:
            run("ls -1 docs 2>/dev/null || echo 'no docs directory yet'");
            break;
        case 16:
            printf("Remove build artifacts and binaries? [y/N]: ");
            fflush(stdout);
            if (fgets(line, sizeof(line), stdin) && (line[0] == 'y' || line[0] == 'Y'))
                run("make clean");
            break;
        case 17:
            printf("bye.\n");
            return 0;
        }
        pause_enter();
    }
    return 0;
}
