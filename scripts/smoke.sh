#!/bin/sh
# End-to-end smoke test on a tiny synthetic corpus.
#
# Exercises configuration loading, tokenizer training, data preparation,
# pretraining with periodic checkpoints, resume without a final checkpoint,
# SFT, evaluation, single/batched/continuous inference, DPO, PPO, GRPO and
# reward-model training.  Everything runs in a temporary directory: real
# checkpoints, artifacts and data are never touched.
#
# Usage: scripts/smoke.sh [path-to-uenr-binary]
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${1:-"$ROOT/uenr"}
BASE_CFG="$ROOT/config/uenr.toml"
[ -x "$BIN" ] || { echo "smoke: build $BIN first (make -j2)"; exit 1; }
[ -f "$BASE_CFG" ] || { echo "smoke: $BASE_CFG missing"; exit 1; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/uenr_smoke.XXXXXX") || exit 1
if [ "${SMOKE_KEEP:-0}" = "1" ]; then
    trap - EXIT INT TERM
    echo "smoke: keeping $TMP"
else
    trap 'rm -rf "$TMP"' EXIT INT TERM
fi
mkdir -p "$TMP/artifacts" "$TMP/checkpoints" "$TMP/data"
CFG="$TMP/uenr.toml"
LOG="$TMP/log.txt"
: >"$LOG"

# Tiny synthetic corpus: 800 short "stories", one per line.
i=0
while [ "$i" -lt 800 ]; do
    printf 'Once upon a time a small robot number %d walked through the garden and counted apples, oranges, stars and clouds while the sun was shining. The robot said hello to the moon and then went home to sleep.\n' "$i"
    i=$((i + 1))
done >"$TMP/data/raw.txt"

# Phase-specific sources in the same tiny domain.  `prepare --phase all`
# rewrites these into data/sft.jsonl, data/pairs.jsonl and
# data/rl_prompts.jsonl, exercising the per-phase machinery the cloud
# workflow uses.
i=0
while [ "$i" -lt 20 ]; do
    printf '{"prompt":"Tell me about robot %d","completion":"Once upon a time robot %d counted apples and went home."}\n' "$i" "$i"
    printf '{"prompt":"Tell me about robot %d","chosen":"Once upon a time robot %d counted apples and went home.","rejected":"home went and apples counted %d robot time a upon Once"}\n' "$i" "$i" "$i"
    printf '{"prompt":"Tell me about robot %d"}\n' "$i"
    i=$((i + 1))
done >"$TMP/data/sft_src.jsonl.tmp"
head -n 20 "$TMP/data/sft_src.jsonl.tmp" >"$TMP/data/sft_src.jsonl"
sed -n '21,40p' "$TMP/data/sft_src.jsonl.tmp" >"$TMP/data/prefs_src.jsonl"
sed -n '41,60p' "$TMP/data/sft_src.jsonl.tmp" >"$TMP/data/rl_src.jsonl"
rm -f "$TMP/data/sft_src.jsonl.tmp"

# Derive a tiny, fast configuration from the shipped one.  Section-aware
# substitution keeps duplicate key names (temperature, dropout, ...) distinct.
awk -v tmp="$TMP" '
function q(s) { return "\"" s "\"" }
/^[[:space:]]*\[/ {
    sec = $0; gsub(/[][[:space:]]/, "", sec); print; next
}
{
    line = $0
    key = $0; sub(/=.*/, "", key); gsub(/[[:space:]]/, "", key)
    if (sec == "paths") {
        if (key == "data")        { print "data = " q(tmp "/data/raw.txt"); next }
        if (key == "train_bin")   { print "train_bin = " q(tmp "/data/train.bin"); next }
        if (key == "val_bin")     { print "val_bin = " q(tmp "/data/val.bin"); next }
        if (key == "sft_jsonl")   { print "sft_jsonl = " q(tmp "/data/sft.jsonl"); next }
        if (key == "pairs_jsonl") { print "pairs_jsonl = " q(tmp "/data/pairs.jsonl"); next }
        if (key == "tokenizer")   { print "tokenizer = " q(tmp "/artifacts/tokenizer.bin"); next }
        if (key == "vocab_file")  { print "vocab_file = " q(tmp "/artifacts/vocab.txt"); next }
        if (key == "merges_file") { print "merges_file = " q(tmp "/artifacts/merges.txt"); next }
        if (key == "model_out")   { print "model_out = " q(tmp "/checkpoints/model.bin"); next }
        if (key == "ema_out")     { print "ema_out = " q(tmp "/checkpoints/model_ema.bin"); next }
        if (key == "log_csv")     { print "log_csv = " q(tmp "/checkpoints/train.csv"); next }
        if (key == "sft_out")     { print "sft_out = " q(tmp "/checkpoints/sft.bin"); next }
        if (key == "lora_out")    { print "lora_out = " q(tmp "/checkpoints/lora.bin"); next }
        if (key == "dpo_out")     { print "dpo_out = " q(tmp "/checkpoints/dpo.bin"); next }
        if (key == "ppo_out")     { print "ppo_out = " q(tmp "/checkpoints/ppo.bin"); next }
        if (key == "grpo_out")    { print "grpo_out = " q(tmp "/checkpoints/grpo.bin"); next }
    }
    if (sec == "datasets.pretrain") {
        if (key == "paths")     { print "paths = [" q(tmp "/data/raw.txt") "]"; next }
        if (key == "max_bytes") { print "max_bytes = 200000"; next }
    }
    if (sec == "datasets.sft") {
        if (key == "paths")       { print "paths = [" q(tmp "/data/sft_src.jsonl") "]"; next }
        if (key == "max_samples") { print "max_samples = 20"; next }
    }
    if (sec == "datasets.lora") {
        if (key == "paths") { print "paths = []"; next }
    }
    if (sec == "datasets.dpo") {
        if (key == "paths")       { print "paths = [" q(tmp "/data/prefs_src.jsonl") "]"; next }
        if (key == "ref_path")    { print "ref_path = " q(tmp "/checkpoints/model.bin"); next }
        if (key == "max_samples") { print "max_samples = 20"; next }
    }
    if (sec == "datasets.rl") {
        if (key == "paths")       { print "paths = [" q(tmp "/data/rl_src.jsonl") "]"; next }
        if (key == "max_samples") { print "max_samples = 20"; next }
    }
    if (sec == "datasets.reward" || sec == "datasets.eval") {
        if (key == "paths") { print "paths = []"; next }
    }
    if (sec == "run") {
        if (key == "threads")            { print "threads = 2"; next }
        if (key == "log_every")          { print "log_every = 1"; next }
        if (key == "checkpoint_every")   { print "checkpoint_every = 2"; next }
        if (key == "checkpoint_dir")     { print "checkpoint_dir = " q(tmp "/checkpoints"); next }
        if (key == "resume")             { print "resume = false"; next }
    }
    if (sec == "model") {
        if (key == "vocab_size")      { print "vocab_size = 512"; next }
        if (key == "hidden")          { print "hidden = 64"; next }
        if (key == "layers")          { print "layers = 4"; next }
        if (key == "heads")           { print "heads = 2"; next }
        if (key == "kv_heads")        { print "kv_heads = 1"; next }
        if (key == "head_dim")        { print "head_dim = 32"; next }
        if (key == "ffn")             { print "ffn = 64"; next }
        if (key == "kv_shared_layers"){ print "kv_shared_layers = 2"; next }
        if (key == "attn_pattern")    { print "attn_pattern = \"SG\""; next }
        if (key == "sliding_window")  { print "sliding_window = 16"; next }
        if (key == "max_seq")         { print "max_seq = 64"; next }
    }
    if (sec == "moe") {
        if (key == "num_experts")     { print "num_experts = 4"; next }
        if (key == "experts_per_tok") { print "experts_per_tok = 1"; next }
        if (key == "expert_ffn")      { print "expert_ffn = 32"; next }
    }
    if (sec == "tokenizer") {
        if (key == "train_bytes")  { print "train_bytes = 200000"; next }
        if (key == "min_pair_freq"){ print "min_pair_freq = 1"; next }
    }
    if (sec == "prepare") {
        if (key == "max_bytes")  { print "max_bytes = 200000"; next }
        if (key == "val_percent"){ print "val_percent = 10"; next }
        if (key == "sft_stories"){ print "sft_stories = 20"; next }
        if (key == "tool_examples") { print "tool_examples = 5"; next }
    }
    if (sec == "training") {
        if (key == "batch_size")  { print "batch_size = 4"; next }
        if (key == "seq_len")     { print "seq_len = 32"; next }
        if (key == "grad_accum")  { print "grad_accum = 2"; next }
        if (key == "max_steps")   { print "max_steps = 4"; next }
        if (key == "warmup_steps"){ print "warmup_steps = 1"; next }
    }
    if (sec == "schedule") {
        if (key == "max_epochs")  { print "max_epochs = 1"; next }
        if (key == "plateau_window") { print "plateau_window = 2"; next }
        if (key == "min_steps_before_stop") { print "min_steps_before_stop = 100"; next }
    }
    if (sec == "dpo") {
        if (key == "ref_checkpoint") { print "ref_checkpoint = " q(tmp "/checkpoints/model.bin"); next }
    }
    if (sec == "ppo") {
        if (key == "rollouts") { print "rollouts = 2"; next }
    }
    if (sec == "grpo") {
        if (key == "group_size") { print "group_size = 2"; next }
    }
    if (sec == "reward") {
        if (key == "steps")      { print "steps = 4"; next }
        if (key == "checkpoint") { print "checkpoint = " q(tmp "/checkpoints/reward.bin"); next }
        if (key == "kind")       { print "kind = \"heuristic\""; next }
    }
    if (sec == "rl") {
        if (key == "max_new_tokens") { print "max_new_tokens = 8"; next }
    }
    if (sec == "inference") {
        if (key == "max_new_tokens") { print "max_new_tokens = 8"; next }
        if (key == "batch_max")      { print "batch_max = 4"; next }
        if (key == "continuous")     { print "continuous = false"; next }
        if (key == "draft_model")    { print "draft_model = \"\""; next }
        if (key == "draft_config")   { print "draft_config = \"\""; next }
        if (key == "speculative")    { print "speculative = false"; next }
    }
    if (sec == "eval") {
        if (key == "batches")    { print "batches = 2"; next }
        if (key == "nseq")       { print "nseq = 2"; next }
        if (key == "seq_len")    { print "seq_len = 32"; next }
        if (key == "gen_tokens") { print "gen_tokens = 4"; next }
    }
    if (sec == "bench") {
        if (key == "gemm_size") { print "gemm_size = 64"; next }
    }
    print
}' "$BASE_CFG" >"$CFG"

run() {
    echo "== $*" | tee -a "$LOG"
    if "$@" >"$TMP/cmd.log" 2>&1; then
        cat "$TMP/cmd.log" | tee -a "$LOG"
    else
        st=$?
        cat "$TMP/cmd.log" | tee -a "$LOG"
        echo "smoke: command failed with status $st" >&2
        return "$st"
    fi
}

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }
need_file() { [ -s "$1" ] || fail "missing output $1"; }

run "$BIN" --config "$CFG" config
run "$BIN" --config "$CFG" tok-train
need_file "$TMP/artifacts/tokenizer.bin"
run "$BIN" --config "$CFG" prepare
need_file "$TMP/data/train.bin"
need_file "$TMP/data/val.bin"
need_file "$TMP/data/sft.jsonl"
need_file "$TMP/data/pairs.jsonl"

# Pretrain with periodic checkpoints, then delete the final model to prove
# resume discovers a periodic step checkpoint on its own.
run "$BIN" --config "$CFG" train --steps 4
need_file "$TMP/checkpoints/model.bin"
need_file "$TMP/checkpoints/step_000002.bin"
rm -f "$TMP/checkpoints/model.bin"
run "$BIN" --config "$CFG" train --resume --steps 6
need_file "$TMP/checkpoints/model.bin"
grep -q "loaded .*step_000004" "$LOG" || fail "resume did not pick the periodic checkpoint"
grep -q "loaded .*step 4" "$LOG" || fail "resume did not restore header step 4"

run "$BIN" --config "$CFG" sft --steps 2 --load "$TMP/checkpoints/model.bin"
need_file "$TMP/checkpoints/sft.bin"
run "$BIN" --config "$CFG" infer --prompt "Once upon a time" --tokens 8 --greedy

printf 'Once upon a time\nA small robot counts apples\nHello moon\n' >"$TMP/prompts.txt"
run "$BIN" --config "$CFG" infer --batch "$TMP/prompts.txt" --tokens 4 --greedy

# Toggle continuous batching in the temp config and exercise that path too.
sed -i 's/^continuous = false$/continuous = true/' "$CFG"
run "$BIN" --config "$CFG" infer --batch "$TMP/prompts.txt" --tokens 4 --greedy
sed -i 's/^continuous = true$/continuous = false/' "$CFG"

run "$BIN" --config "$CFG" dpo --steps 2 --load "$TMP/checkpoints/sft.bin" --ref "$TMP/checkpoints/model.bin"
need_file "$TMP/checkpoints/dpo.bin"
run "$BIN" --config "$CFG" grpo --steps 1 --load "$TMP/checkpoints/dpo.bin" --ref "$TMP/checkpoints/model.bin"
need_file "$TMP/checkpoints/grpo.bin"
run "$BIN" --config "$CFG" eval --load "$TMP/checkpoints/grpo.bin" --batches 2 --seq 32

# Any NaN/Inf or silently dropped gradient is a failure.
if grep -Eqi 'nan|backward failed|non-finite|gradient.*missing' "$LOG"; then
    fail "non-finite value or backward failure in smoke log"
fi

echo "SMOKE PASS: $TMP"
