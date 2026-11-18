#!/usr/bin/env bash
# Offline-only conversion instructions for HuggingFace datasets that ship as
# parquet.  The UENR workflow never runs Python and never reads parquet.
#
# Run these commands ONCE on a machine with Python/internet, then upload the
# resulting JSONL as the stable release asset.  The workflow will download the
# asset on subsequent runs and will not touch HuggingFace.
#
# Requires: pip install datasets huggingface_hub
set -euo pipefail
REPO=${GITHUB_REPOSITORY:-michaelowusuntim6/project_uenr}

cat <<'EOF'
This helper does not convert anything automatically; it documents the one-time
commands for parquet-only datasets. Run them manually, then publish the JSONL.

1) UltraFeedback binarized -> DPO pairs
   pip install datasets huggingface_hub
   python3 - <<'PY'
   import json
   from datasets import load_dataset
   ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
   with open("data/prefs_ultrafeedback.jsonl", "w") as f:
       for r in ds:
           chosen = r["chosen"][-1]["content"]
           rejected = r["rejected"][-1]["content"]
           prompt = r["prompt"]
           f.write(json.dumps({"prompt": prompt, "chosen": chosen,
                               "rejected": rejected}) + "\n")
   PY

2) smoltldr -> RL prompts
   python3 - <<'PY'
   import json
   from datasets import load_dataset
   ds = load_dataset("mlabonne/smoltldr", split="train")
   with open("data/rl_prompts_smoltldr.jsonl", "w") as f:
       for r in ds:
           f.write(json.dumps({"prompt": r["prompt"]}) + "\n")
   PY

3) Publish as the stable release assets (replace tags only if you want to
   override the current defaults; the workflow uses --clobber only on upload):
   gh release create dataset-dpo -R "$REPO" --prerelease \
       --title "dataset-dpo" --notes "DPO preference pairs"
   gh release upload dataset-dpo data/prefs_ultrafeedback.jsonl -R "$REPO" --clobber
   # Rename to the configured asset name if needed:
   #   cp data/prefs_ultrafeedback.jsonl data/prefs_tinystories.jsonl
   gh release create dataset-rl -R "$REPO" --prerelease \
       --title "dataset-rl" --notes "RL prompt set"
   gh release upload dataset-rl data/rl_prompts_smoltldr.jsonl -R "$REPO" --clobber

Important: if you upload under different filenames, update
[datasets.dpo].paths / [datasets.rl].paths in config/uenr.toml, or rename the
asset to prefs_tinystories.jsonl / rl_prompts.jsonl.  Do not commit the JSONL
to this repository; keep it as a release asset.
EOF
