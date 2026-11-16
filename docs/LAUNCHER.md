# Interactive Launcher

`./uenr-menu` is a single numbered C program (no Python, no shell wrapper).
It changes to its own directory, prints the menu, and returns to it after every
action. Press Enter after each action to continue.

```
 1. Check environment (compilers, CPU, files)
 2. Build everything
 3. Run the test suite
 4. Benchmarks (GEMM vs measured roof)
 5. Train the BPE tokenizer
 6. Prepare dataset (train/val/SFT/pairs)
 7. Pretrain
 8. Supervised finetune (SFT)
 9. LoRA finetune
10. DPO preference tuning
11. PPO policy-gradient tuning
12. GRPO group-relative tuning
13. Inference / chat
14. View configuration
15. View documentation
16. Clean build artifacts
17. Exit
```

## Behaviour

- **1** prints gcc/make/git/python versions, CPU model and core count, key
  project files, and offers `sudo apt-get install build-essential` if the
  toolchain is missing (confirmation required).
- **2–3** run `make -j2` and `make test`.
- **4** runs `./uenr bench` and builds the benchmark binaries.
- **5–6** retrain the tokenizer and regenerate the prepared data. Both are
  idempotent and overwrite their artifacts.
- **8–14** ask for a step count (Enter uses the config default) and run the
  matching trainer with `--steps N`. Pretrain and SFT also ask “Resume newest
  checkpoint?”; answering `y` appends `--resume`.
- **13** calls the inference chat **in-process** so the menu and chat share one
  stdin buffer (a `system()` child would lose piped input to the parent's
  stdio buffer). Type lines; an empty line exits the chat and returns to the
  menu.
- **14** prints the resolved config and the raw TOML.
- **15** lists `docs/`.
- **16** asks for confirmation, then runs `make clean`.
- **17** exits.

Every action is resilient: a missing model, missing data, failed build, or
non-zero child status is reported and the menu continues.

## Scripted self-test

The menu can be driven from a pipe. One tested sequence (also useful as a
regression):

```bash
printf '1\n\n2\n\n3\n\n4\n\n5\n\n6\n\n7\n1\n\n8\n1\n\n9\n1\n\n10\n1\n\n11\n1\n\n12\n1\n\n19\n\n20\n\n23\n' \
  | ./uenr-menu
```

For chat testing, include a prompt line after `13` and an empty line to exit;
the menu calls the chat in-process so this works with piped input.

## Updated menu (23 options)

The launcher now orchestrates the full program, including the post-optimization
features:

```
 1 env check       2 build          3 make test      4 make check
 5 benchmarks      6 tok-train      7 prepare        8 pretrain
 9 SFT            10 LoRA          11 reward model  12 DPO
13 PPO            14 GRPO          15 chat          16 batch inference
17 eval           18 KV calculator (incl. Q4+evict) 19 config        20 docs
21 status         22 clean         23 exit
```

- **16** asks for a prompt file and token count, then runs
  `uenr infer --batch FILE --greedy`.
- **17** runs the full eval: held-out loss/ppl, tokenizer bytes/token,
  generation end rate, repeated token-3grams and tool-JSON validity.
- **18** prints active slots, local/global split, ring size and cache bytes at
  `max_seq` and 1M tokens for FP32/FP16/INT8/Q4.
- **21** prints the top of `CHECK_LIST.md` and `docs/STATUS.md`.
- After any interrupted training run, choose 8 (or 9), enter the target step
  count, answer `y` to resume, and training continues from the newest periodic
  checkpoint or `checkpoints/model.bin` with optimizer state restored.

Piped self-test (no training, fast):

```bash
printf '18\n\n20\n\n22\nn\n\n23\n' | ./uenr-menu
```
