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
 8. Pretrain
 9. Supervised finetune (SFT)
10. LoRA finetune
11. Train pairwise reward model
12. DPO preference tuning
13. PPO policy-gradient tuning
14. GRPO group-relative tuning
15. Inference / chat
16. Batched inference from a prompt file
17. Evaluate (val loss/ppl + generation metrics)
18. KV-cache size calculator (FP32/FP16/INT8/Q4 + eviction)
19. View configuration
20. View documentation
21. Project status / checklist
22. Clean build artifacts
23. End-to-end smoke test (temporary directory)
24. Exit
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
- **8** asks for a step count (Enter uses the config default) and optionally
  for `--resume`. Resume discovers periodic `step_*.bin` checkpoints by header
  step even when the final model file is absent; optimizer state is restored.
- **9–14** ask for a step count and run SFT/LoRA/reward/DPO/PPO/GRPO. SFT
  loads the pretrained checkpoint and does not implement `--resume`, so no
  resume prompt is shown. DPO/PPO/GRPO now fail with an actionable message if
  the policy or reference checkpoint is missing instead of silently using
  random weights.
- **15** calls the inference chat **in-process** so the menu and chat share one
  stdin buffer (a `system()` child would lose piped input to the parent's
  stdio buffer). Type lines; an empty line exits the chat and returns to the
  menu.
- **16** asks for a prompt file and token count, then runs
  `uenr infer --batch FILE --greedy` (continuous batching is controlled by
  `inference.continuous`).
- **17** runs the full eval: held-out loss/ppl, tokenizer bytes/token,
  generation end rate, repeated token-3grams and tool-JSON validity.
- **18** prints active slots, local/global split, ring size and cache bytes at
  `max_seq` and 1M tokens for FP32/FP16/INT8/Q4.
- **19** prints the resolved config and the raw TOML. Configuration is now
  strict: an unknown key or malformed value is reported with file/line/section
  and the command refuses to run.
- **20** lists `docs/` (including the new `BENCHMARKS.md`).
- **21** prints the top of `docs/CHECK_LIST.md` and `docs/STATUS.md`; the
  latter includes the correctness-pass commit table.
- **22** asks for confirmation, then runs `make clean`.
- **23** runs `scripts/smoke.sh`: tokenizer, prepare, pretrain, resume with no
  final checkpoint, SFT, eval, single/batched/continuous inference,
  DPO/PPO/GRPO and reward-model training in a temporary directory. Real data
  and checkpoints are never touched.
- **24** exits.

Every action is resilient: a missing model, missing data, failed build, or
non-zero child status is reported and the menu continues.

## Scripted self-test

The menu can be driven from a pipe. Fast, read-only sequence (no training):

```bash
printf '18\n\n20\n\n1\n\n24\n' | ./uenr-menu
```

For chat testing, include a prompt line after `15` and an empty line to exit;
the menu calls the chat in-process so this works with piped input.
