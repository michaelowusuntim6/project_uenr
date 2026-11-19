# CHECK_LIST03 — pipeline simplification, CPU adaptation and inference defaults

This is the active work checklist for the post-ADAPT pass. Status keys:

- `[x]` applied and verified
- `[~]` experiment completed / decision recorded, not enabled
- `[ ]` pending
- `[-]` rejected with a documented reason

Every item names the files it touches and the short test used. Long training is
not part of this checklist; verification uses unit tests, `make smoke`,
short 20–100-step runs, and `mode=bench`/`mode=smoke` dispatches only.

## Phase A — repository and pipeline shape

- [x] **A1. Move the adaptation study.** `ADAPT.md` → `docs/ADAPT.md`;
  update `docs/README.md`. Test: file exists, links resolve.
- [x] **A2. Freeze a baseline.** Record current `make test` count, `make
  smoke`, KV defaults and cloud throughput in this checklist before edits.
- [x] **A3. Audit training entry points.** List every `uenr` training command,
  its caller, checkpoint dependency and test coverage.
- [x] **A4. Retire LoRA from the default pipeline.** Tiny full fine-tuning is
  cheaper than maintaining a second parameter path; keep low-level code/tests
  only if still referenced. Remove menu/workflow/config defaults.
- [x] **A5. Retire PPO from the default pipeline.** Sequence-level PPO has no
  value head/GAE and duplicates GRPO's policy-gradient role. Remove the
  workflow job/mode/menu entry; retain the math test as a legacy unit test.
- [x] **A6. Retire the learned reward model from the default pipeline.**
  `train-rm` on synthetic pairs is weak and R1 recommends rule/verifier
  rewards for reasoning. Remove job/mode/menu; keep heuristic reward.
- [x] **A7. Define the surviving sequence:**
  `pretrain → SFT → DPO → GRPO → eval`. DPO and GRPO consume the previous
  checkpoint; GRPO's reference remains the frozen pretrain checkpoint.
- [x] **A8. Make GRPO load the DPO checkpoint.** `scripts/run_phase.sh`:
  download `dpo-checkpoint` when available and pass `--load checkpoints/dpo.bin`;
  fall back to SFT only when DPO is absent.
- [x] **A9. Make eval load the most advanced checkpoint.** Prefer GRPO, then
  DPO, then SFT, then pretrain; update the workflow eval job and log which
  checkpoint was evaluated.
- [x] **A10. Add a linear `pipeline` workflow mode.** One mode runs
  pretrain→SFT→DPO→GRPO→eval; remove `full-core`/`full-rl` as separate modes.
- [x] **A11. Flatten workflow `needs`:** `build`/`diagnostic`/`restore` are
  roots; `prepare-all←restore`; `pretrain←build,prepare-all`;
  `sft←pretrain`; `dpo←sft`; `grpo←dpo`; `eval←grpo`; `bench←build,prepare-all`.
- [x] **A12. Remove retired workflow jobs and inputs** (`lora`, `ppo`,
  `train-rm`, `steps_lora`, `steps_ppo`, `steps_train_rm`, `reward=rm`).
- [x] **A13. Add `scripts/train_pipeline.sh`** for the local/orchestrator
  sequence with per-phase steps, checkpoint handoff, stop-on-failure and a
  final eval.
- [x] **A14. Flatten the local menu.** Replace the branched per-phase options
  with: full pipeline, pretrain, SFT, DPO, GRPO, eval, inference, batch
  inference, KV calculator, config, docs, status, clean, smoke, cloud, exit.
  Keep an expert-only note that retired methods are not in the pipeline.
- [x] **A15. Flatten the cloud submenu.** Dispatch pipeline/smoke/bench and
  single phases that still exist; remove LoRA/PPO/train-rm/custom retired
  modes.
- [x] **A16. Update `scripts/smoke.sh`** to the surviving sequence and remove
  LoRA/PPO/train-rm from the end-to-end test.
- [x] **A17. Update `docs/TRAINING.md`, `CONFIG.md`, `DETAILED.md`,
  `CLOUD_TRAINING.md`, `README.md`, `LAUNCHER.md`** so no document advertises a
  retired default pipeline.
- [x] **A18. Add a short pipeline test.** 2–4 steps per surviving phase in a
  temporary directory; assert checkpoint handoffs and monotonic phase tags.

## Applied notes

- A1–A18: see commit chain `225315d..` (move ADAPT, linear pipeline, workflow
  flattening, menu/CLI retirement, short pipeline test).
- B1–B4: INT8 KV is the default and works in single, batched and continuous
  inference; SwiGLU clamp is shared by scalar/AVX2/quantized paths.
- B13/B15: rejected items and measured decisions are recorded in `ADAPT.md`
  and `ARCH_OPTIMIZATION.md`.
- B14: `make test` 19/19, `make check-isa` clean, `make smoke` pass, short
  pipeline 4/4/2/2 pass, menu self-test (options 16/18/19) pass.

## Phase B — inference defaults and CPU architecture

- [x] **B1. Choose the default KV precision.** Requirement: below FP32,
  minimal throughput loss. Candidate ranking from existing measurements:
  INT8 (64 B/token, ~1.04× decode, max logit error 4.5e-3) > FP16 (128 B,
  ~1.0×) > Q4 (40 B, 0.60–0.76×, too slow). **Decision: INT8.**
- [x] **B2. Set `inference.kv_int8 = true`, FP16/Q4/Q2 false.** Document the
  exact bytes/token and quality trade-off. Test: `uenr config`, `kvbench` at
  512/2048, `make smoke`.
- [~] **B3. Decide on auxiliary-loss-free MoE.** Current auxiliary loss is
  already batch-wise; V3 reports batch-wise balancing ≈ aux-free. Adding a
  persistent bias would change checkpoint semantics (sidecar needed). Decision:
  keep batch-wise aux loss for now; record as a controlled experiment, not a
  default edit.
- [x] **B4. Add optional SwiGLU clamp.** Gate ≤10, linear ∈[-10,10], config
  `training.swiglu_clamp`, default 10 for training/inference; derivative zero
  when clamped. Test: finite difference / `test_ops` and smoke.
- [ ] **B5. SFT best-fit packing.** Bucket conversations, pack to `seq_len`,
  block-diagonal attention mask, assistant-only labels. Test: packed vs
  unpacked loss equivalence on a fixed batch.
- [ ] **B6. R1 verifier rewards.** Add exact-match/format verifier for a small
  arithmetic/tool-call curriculum; zero-variance group skip; reference refresh
  every 400 steps. Keep sequence-level GRPO clip 0.2. Test: reward unit tests,
  one short GRPO run.
- [ ] **B7. HCA-lite experiment.** Compressed global KV m=4/8 plus exact SWA
  ring, optional rank-2 residual. Evaluate cached ppl and bytes before enabling.
- [ ] **B8. SWA bounded replay + persistent KV experiment.** Last `n_win=128`
  replay on prefix miss; storage format and eviction policy; measure TTFT.
- [ ] **B9. MLA-lite experiment.** d_c=16 + 8 RoPE dims; must beat existing
  rank-8 low-rank KV at equal bytes before adoption.
- [ ] **B10. CED-lite experiment.** Inference-only global KV projection from
  layer L/2; measure prefill at 8K–64K; lower priority than HCA/replay.
- [ ] **B11. MTP-lite experiment.** One extra depth, shared head, λ=0.3;
  report matched-token val loss and wall time.
- [ ] **B12. True Muon/Newton–Schulz ablation.** 5 iterations, non-expert
  matrices first, hard 10% step-time budget; compare matched tokens and wall.
- [x] **B13. Reject list, documented:** mHC, full Engram, DSpark/draft at
  current acceptance, native FP4, FP8/BF16/EP, grouped output projection,
  anticipatory routing.
- [x] **B14. CPU test matrix.** `make test`, `make check-isa`, `make smoke`,
  `test_simd_dispatch`, `kvbench` INT8 vs FP32, short 20-step pipeline.
- [x] **B15. Update ADAPT/ARCH docs** with measured decisions and negative
  results; do not leave projections presented as measurements.

## Phase C — finalisation

- [ ] **C1.** Run the short local pipeline (4 phases + eval) and a cloud
  `mode=smoke` dispatch; verify tokenizer release and checkpoint handoff.
- [ ] **C2.** Run one `mode=bench` dispatch with the INT8 default; record
  decode/throughput.
- [x] **C3.** Update handoff, changelog, README, launcher and deployment docs.
- [x] **C4.** Commit in logical groups, push `master`, confirm laptop tree
  `/home/mike/UENR/uenr` remains clean at `e4729f4` (verified after push).

## Explicit non-goals for this pass

- No full-length training run during development.
- No checkpoint format change.
- No removal of the AVX1 path or `make check-isa`.
- No mHC, full Engram, DSpark/draft, FP4/FP8, or multimodal work.
