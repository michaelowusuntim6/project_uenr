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
- **7–12** ask for a step count (Enter uses the config default) and run the
  matching trainer with `--steps N`.
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
printf '1\n\n2\n\n3\n\n4\n\n5\n\n6\n\n7\n1\n\n8\n1\n\n9\n1\n\n10\n1\n\n11\n1\n\n12\n1\n\n14\n\n15\n\n17\n' \
  | ./uenr-menu
```

For chat testing, include a prompt line after `13` and an empty line to exit;
the menu calls the chat in-process so this works with piped input.
