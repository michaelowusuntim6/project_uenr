#!/usr/bin/env bash
# Dataset bootstrap and cache, driven entirely by config/uenr.toml.
#
# `uenr datasets` prints one tab-separated recipe per configured path:
#   phase path release_tag release_asset url derive_from derive_path
#   convert_from convert_to limit bootstrap format
#
# For each path the script:
#   1. uses the local file if it is already present;
#   2. otherwise downloads `release_asset` from `release_tag`;
#   3. otherwise, when bootstrap=true, materialises the file from `url`
#      (with an optional `uenr convert` step) or from `derive_path`;
#   4. uploads the result back to the release tag when it was missing.
#
# Changing datasets therefore means editing [datasets.*] in the TOML only.
# The workflow calls this script, including for browser-triggered dispatches.
#
# Environment: GH_TOKEN, GITHUB_REPOSITORY,
#              BOOTSTRAP_DATASETS=true|false (default true)
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
REPO=${GITHUB_REPOSITORY:-michaelowusuntim6/project_uenr}
BOOT=${BOOTSTRAP_DATASETS:-true}
BIN=${UENR_BIN:-./uenr}
CFG=${UENR_CONFIG:-config/uenr.toml}

[ -x "$BIN" ] || { echo "ERROR: $BIN missing; run make first" >&2; exit 1; }
mkdir -p data artifacts checkpoints

tag_exists() { gh release view "$1" -R "$REPO" >/dev/null 2>&1; }
ensure_tag() {
    if ! tag_exists "$1"; then
        gh release create "$1" -R "$REPO" --prerelease --title "$2" \
            --notes "Stable dataset asset for UENR CI (do not delete)." >/dev/null
    fi
}
have_asset() {
    command -v gh >/dev/null 2>&1 || return 1
    gh release view "$1" -R "$REPO" --json assets \
        --jq ".assets[].name" 2>/dev/null | grep -qx "$2"
}
download_asset() { # tag asset dest
    [ -n "$1" ] && [ -n "$2" ] || return 1
    have_asset "$1" "$2" || return 1
    local tmp rc=1
    tmp=$(mktemp -d) || return 1
    if gh release download "$1" -R "$REPO" -p "$2" -D "$tmp" --clobber >/dev/null 2>&1 &&
       [ -s "$tmp/$2" ]; then
        mkdir -p "$(dirname "$3")"
        mv "$tmp/$2" "$3"
        rc=0
    fi
    rm -rf "$tmp"
    [ "$rc" -eq 0 ] && [ -s "$3" ]
}
upload_asset() { # tag asset file
    command -v gh >/dev/null 2>&1 || {
        echo "bootstrap: gh not installed; keeping $3 local" >&2
        return 0
    }
    ensure_tag "$1" "$1 dataset"
    gh release upload "$1" "$3" -R "$REPO" --clobber >/dev/null
}

convert() { # from to in out limit
    if [ "${5:-0}" -gt 0 ]; then
        "$BIN" convert --from "$1" --to "$2" --in "$3" --out "$4" --limit "$5"
    else
        "$BIN" convert --from "$1" --to "$2" --in "$3" --out "$4"
    fi
}

fail=0
while IFS=$'\x1f' read -r phase idx path tag asset url derive dpath cfrom cto limit boot fmt; do
    [ -n "$path" ] || continue
    mkdir -p "$(dirname "$path")"

    if [ -s "$path" ]; then
        if [ "$BOOT" = true ] && [ -n "$tag" ] && [ -n "$asset" ] && ! have_asset "$tag" "$asset"; then
            upload_asset "$tag" "$asset" "$path"
            echo "bootstrap[$phase]: published existing $path -> $tag/$asset"
        fi
        continue
    fi

    # A published asset is authoritative and already in the final format.
    if download_asset "$tag" "$asset" "$path"; then
        echo "bootstrap[$phase]: downloaded $tag/$asset -> $path"
        continue
    fi

    if [ "$BOOT" != true ] || [ "$boot" != 1 ]; then
        echo "ERROR: $path is missing and bootstrap is disabled for [$phase]" >&2
        fail=1
        continue
    fi
    if [ "$idx" != 1 ]; then
        echo "ERROR: additional path $path (index $idx) for [$phase] is missing" >&2
        echo "       fetch recipes apply to the first path; provide extra paths locally" >&2
        fail=1
        continue
    fi

    created=0
    if [ -n "$derive" ] && [ -n "$cfrom" ] && [ -n "$cto" ] && [ -s "$dpath" ]; then
        echo "bootstrap[$phase]: deriving $path from $dpath"
        convert "$cfrom" "$cto" "$dpath" "$path" "$limit"
        [ -s "$path" ] && created=1
    fi

    if [ "$created" -eq 0 ] && [ -n "$url" ]; then
        echo "bootstrap[$phase]: downloading $url"
        raw="$path.download.$$"
        rm -f "$raw"
        curl -fL --retry 3 --retry-delay 5 -o "$raw" "$url"
        if [ -n "$cfrom" ] && [ -n "$cto" ]; then
            echo "bootstrap[$phase]: converting $url -> $fmt"
            convert "$cfrom" "$cto" "$raw" "$path" "$limit"
            rm -f "$raw"
        else
            mv "$raw" "$path"
        fi
        [ -s "$path" ] && created=1
    fi

    if [ "$created" -eq 0 ]; then
        echo "ERROR: cannot materialise $phase path '$path'" >&2
        echo "       set release_tag/release_asset, url or derive_from in [datasets.$phase]" >&2
        fail=1
        continue
    fi

    if [ -n "$tag" ] && [ -n "$asset" ]; then
        upload_asset "$tag" "$asset" "$path"
        echo "bootstrap[$phase]: published $tag/$asset"
    fi
done < <("$BIN" --config "$CFG" datasets)

if [ "$fail" -ne 0 ]; then
    echo "bootstrap: one or more datasets are unavailable" >&2
    exit 1
fi
echo "bootstrap: dataset assets ready"
