# UENR Documentation Index

**Start with [DETAILED.md](DETAILED.md)** — one long, ordered, deduplicated
guide covering the project state, architecture, KV/long-context engineering,
training, QLoRA/block checkpointing, inference/serving, configuration,
measurements, build/operation, limitations, and future work. The files below
are focused references for deeper dives.

- [RELEASE.md](RELEASE.md) — v0.1.0 release documentation, verification and limitations.
- [../CHANGELOG.md](../CHANGELOG.md) — release history (v0.1.0 is the first public release).
- [DETAILED.md](DETAILED.md) — the complete guide; read this first.
- [AUDIT.md](AUDIT.md) — the two reference repos, the measured CPU audit,
  ceilings, and the original project plan.
- [ARCHITECTURE.md](ARCHITECTURE.md) — modules, parameter/activation layouts,
  forward/backward, GEMM, threading, checkpoints.
- [FORMULAS.md](FORMULAS.md) — derivations for every custom formula.
- [CONFIG.md](CONFIG.md) — every TOML field.
- [TOKENIZER.md](TOKENIZER.md) — byte-level BPE, sanitizer, chat/tool template.
- [TRAINING.md](TRAINING.md) — pretrain, SFT, LoRA/QLoRA, DPO, PPO, GRPO,
  memory modes, logging, resume.
- [MOE.md](MOE.md) — routing, dispatch, auxiliary loss, gradients.
- [INFERENCE.md](INFERENCE.md) — generation, sampling, chat, KV cache, serving,
  draft speculation.
- [BUILD.md](BUILD.md) — build/test/benchmark instructions and measured numbers.
- [BENCHMARKS.md](BENCHMARKS.md) — methodology, current measurements, and
  rejected optimization experiments.
- [LAUNCHER.md](LAUNCHER.md) — the numbered interactive menu.
- [ROADMAP.md](ROADMAP.md) — completed work and next steps.
- [STATUS.md](STATUS.md) — verified state, exact commands, and handoff notes.
- [CHECK_LIST.md](CHECK_LIST.md) — items 1–58 with measured xN and commits.
- [CHECK_LIST02.md](CHECK_LIST02.md) — the second optimization plan and the
  productionization addendum with measured xN.
- [TODO.md](TODO.md) — the living implementation checklist.
- [DOCUMENT.md](DOCUMENT.md) — the original architecture/optimization brief.

The only Markdown file in the repository root is `README.md` (quickstart).
Everything else lives in `docs/`.
