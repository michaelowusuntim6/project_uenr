#!/usr/bin/env bash
# One-time dataset bootstrap and cache.  Pure bash + the C `uenr` converter;
# no Python.  Assets are stored on stable release tags and reused thereafter.
#
# Tags/assets:
#   dataset-pretrain  tinystories_train.txt
#   dataset-sft       sft_tinystories.jsonl   (prompt/completion JSONL)
#   dataset-dpo       prefs_tinystories.jsonl (prompt/chosen/rejected JSONL)
#   dataset-rl        rl_prompts.jsonl        (prompt-only JSONL)
#
# Environment: GH_TOKEN, GITHUB_REPOSITORY,
#              BOOTSTRAP_DATASETS=true|false (default true)
set -euo pipefail

REPO=${GITHUB_REPOSITORY:-michaelowusuntim6/project_uenr}
BOOT=${BOOTSTRAP_DATASETS:-true}
BIN=./uenr
mkdir -p data

need() { [ -s "$1" ] || { echo "ERROR: missing $1" >&2; exit 1; }; }
tag_exists() { gh release view "$1" -R "$REPO" >/dev/null 2>&1; }
ensure_tag() { # tag title
    if ! tag_exists "$1"; then
        gh release create "$1" -R "$REPO" --prerelease --title "$2" \
            --notes "Stable dataset asset for UENR CI (do not delete)." >/dev/null
    fi
}
have_asset() { # tag asset
    gh release view "$1" -R "$REPO" --json assets \
        --jq ".assets[].name" 2>/dev/null | grep -qx "$2"
}
download_asset() { # tag asset dest
    local tag=$1 asset=$2 dest=$3
    if have_asset "$tag" "$asset"; then
        if [ ! -s "$dest" ]; then
            gh release download "$tag" -R "$REPO" -p "$asset" -D data --clobber >/dev/null
        fi
        return 0
    fi
    return 1
}
upload_asset() { # tag asset file
    ensure_tag "$1" "$1 dataset"
    gh release upload "$1" "$3" -R "$REPO" --clobber >/dev/null
}

# ------------------------------------------------------------------ pretrain
PRETRAIN_URL=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
if ! download_asset dataset-pretrain tinystories_train.txt data/tinystories_train.txt; then
    if [ "$BOOT" != true ]; then
        echo "ERROR: dataset-pretrain asset missing; set bootstrap_datasets=true" >&2
        echo "  source: $PRETRAIN_URL" >&2
        exit 1
    fi
    echo "bootstrap: downloading TinyStories (~1.9 GB)"
    curl -fL --retry 3 --retry-delay 5 -o data/tinystories_train.txt "$PRETRAIN_URL"
    SZ=$(stat -c %s data/tinystories_train.txt)
    [ "$SZ" -gt 1700000000 ] || { echo "ERROR: TinyStories download too small ($SZ)" >&2; exit 1; }
    upload_asset dataset-pretrain tinystories_train.txt data/tinystories_train.txt
fi
need data/tinystories_train.txt

# ----------------------------------------------------------------------- SFT
SFT_URL=https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt
if ! download_asset dataset-sft sft_tinystories.jsonl data/sft_tinystories.jsonl; then
    if [ "$BOOT" != true ]; then
        echo "ERROR: dataset-sft asset missing; set bootstrap_datasets=true" >&2
        echo "  source: $SFT_URL" >&2
        exit 1
    fi
    echo "bootstrap: downloading TinyStoriesInstruct validation split (~27 MB)"
    curl -fL --retry 3 --retry-delay 5 -o data/tinystories_instruct.txt "$SFT_URL"
    SZ=$(stat -c %s data/tinystories_instruct.txt)
    [ "$SZ" -gt 20000000 ] || { echo "ERROR: TinyStoriesInstruct download too small ($SZ)" >&2; exit 1; }
    $BIN convert --from tinystories_instruct --to prompt_completion \
        --in data/tinystories_instruct.txt --out data/sft_tinystories.jsonl --limit 200000
    upload_asset dataset-sft sft_tinystories.jsonl data/sft_tinystories.jsonl
fi
need data/sft_tinystories.jsonl

# ----------------------------------------------------------------------- DPO
# Domain-specific pairs synthesized from the SFT records (chosen story vs a
# deterministic word-reversed rejected story).  This keeps the vocabulary
# aligned with TinyStories while avoiding parquet readers in the workflow.
if ! download_asset dataset-dpo prefs_tinystories.jsonl data/prefs_tinystories.jsonl; then
    if [ "$BOOT" != true ]; then
        echo "ERROR: dataset-dpo asset missing; set bootstrap_datasets=true" >&2
        echo "  alternative: offline-convert HuggingFaceH4/ultrafeedback_binarized (scripts/fetch_and_convert.sh)" >&2
        exit 1
    fi
    $BIN convert --from prompt_completion --to prefs \
        --in data/sft_tinystories.jsonl --out data/prefs_tinystories.jsonl --limit 50000
    upload_asset dataset-dpo prefs_tinystories.jsonl data/prefs_tinystories.jsonl
fi
need data/prefs_tinystories.jsonl

# ------------------------------------------------------------------------ RL
if ! download_asset dataset-rl rl_prompts.jsonl data/rl_prompts.jsonl; then
    if [ "$BOOT" != true ]; then
        echo "ERROR: dataset-rl asset missing; set bootstrap_datasets=true" >&2
        echo "  alternative: offline-convert mlabonne/smoltldr (scripts/fetch_and_convert.sh)" >&2
        exit 1
    fi
    $BIN convert --from prompt_completion --to prompts \
        --in data/sft_tinystories.jsonl --out data/rl_prompts.jsonl --limit 5000
    upload_asset dataset-rl rl_prompts.jsonl data/rl_prompts.jsonl
fi
need data/rl_prompts.jsonl

echo "bootstrap: dataset assets ready"
