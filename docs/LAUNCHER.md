# Interactive Launcher

`./uenr-menu` is a single numbered C program (no Python, no shell wrapper).
It changes to its own directory, prints the menu, and returns to it after every
action. Press Enter after each action to continue.

```
 1. Check environment (compilers, CPU, files)
 2. Build everything
 3. Run the test suite
 4. Full check (build + tests + benches + ISA)
 5. Benchmarks (GEMM vs measured roof)
 6. Train the BPE tokenizer
 7. Prepare dataset (train/val/SFT/pairs)
 8. Run training pipeline (pretrain -> SFT -> DPO -> GRPO -> eval)
 9. Pretrain
10. Supervised finetune (SFT)
11. DPO preference tuning (loads SFT)
12. GRPO group-relative tuning (loads DPO)
13. Evaluate (val loss/ppl + generation metrics)
14. Inference / chat
15. Batched inference from a prompt file
16. KV-cache size calculator (FP32/FP16/INT8/Q4 + eviction)
17. View configuration
18. View documentation
19. Project status / checklist
20. Clean build artifacts
21. End-to-end smoke test (temporary directory)
22. Cloud / GitHub
23. Exit
```

## Behaviour

- **1** prints gcc/make/git/python versions, CPU model and core count, key
  project files, and offers `sudo apt-get install build-essential` if the
  toolchain is missing (confirmation required).
- **2–4** run `make -j2`, `make test`, and `make check` (the latter includes
  the tests, benchmark build and the AVX2/FMA ISA assertion).
- **5** runs `./uenr bench` and builds the benchmark binaries.
- **6–7** retrain the tokenizer and regenerate the prepared data. Both are
  idempotent and overwrite their artifacts.
- **8** is the straight-line pipeline. Enter runs the full configured budget
  (1600/1000/600/400); type `short` for 4/4/2/2 or a number `N` for N steps
  per phase. It calls `scripts/train_pipeline.sh`, stops at the first failure,
  and each phase loads the previous phase's checkpoint.
- **9–12** run one phase: pretrain, SFT (loads pretrain), DPO (loads SFT,
  pretrain reference), GRPO (loads DPO, pretrain reference). Each fails with
  an actionable message if its input checkpoint is missing.
- **13** runs eval and picks the newest available final checkpoint in the
  order GRPO -> DPO -> SFT -> pretrain.
- **14** calls the inference chat **in-process** so the menu and chat share one
  stdin buffer (a `system()` child would lose piped input to the parent's
  stdio buffer). Type lines; an empty line exits the chat and returns to the
  menu.
- **15** asks for a prompt file and token count, then runs
  `uenr infer --batch FILE --greedy` (continuous batching is controlled by
  `inference.continuous`).
- **16** prints active slots, local/global split, ring size and cache bytes at
  `max_seq` and 1M tokens for FP32/FP16/INT8/Q4.
- **17** prints the resolved config (including `swiglu_clamp` and the active
  KV precision) and the raw TOML. Configuration is strict: an unknown key or
  malformed value is reported with file/line/section and the command refuses
  to run.
- **18** lists `docs/` (including `ADAPT.md`, `CHECK_LIST03.md` and
  `BENCHMARKS.md`).
- **19** prints the top of `docs/CHECK_LIST03.md` and `docs/STATUS.md`.
- **20** asks for confirmation, then runs `make clean`.
- **21** runs `scripts/smoke.sh`: tokenizer, prepare, pretrain, resume with no
  final checkpoint, SFT, eval, single/batched/continuous inference, and the
  DPO/GRPO alignment checks in a temporary directory. Real data and
  checkpoints are never touched.
- **22** opens the Cloud / GitHub submenu (auth, token, run status,
  pipeline/smoke/bench dispatch with `--watch`, checkpoint-release cleanup).
- **23** exits.

Every action is resilient: a missing model, missing data, failed build, or
non-zero child status is reported and the menu continues.

## Scripted self-test

The menu can be driven from a pipe. Fast, read-only sequence (no training):

```bash
printf '16\n\n18\n\n19\n\n23\n' | ./uenr-menu
```

For chat testing, include a prompt line after `14` and an empty line to exit;
the menu calls the chat in-process so this works with piped input.
