#!/usr/bin/env bash
# Shared GitHub environment loader for UENR orchestration scripts.
# Source this file; it exports UENR_GH_REPO and (when available) GH_TOKEN.
#
# Token lookup order:
#   1. $GH_TOKEN or $GITHUB_TOKEN already in the environment
#   2. <repo>/github_token.txt          (recommended; gitignored)
#   3. <repo>/.github_token             (gitignored)
#   4. <repo>/config/github_token.txt   (gitignored)
#   5. an existing `gh auth login` session
#
# Never commit a real token. `scripts/github_auth.sh setup` creates the file
# with mode 0600 and validates it against api.github.com.

UENR_GH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UENR_GH_ROOT

_uenr_repo_from_git() {
    local url
    url=$(git -C "$UENR_GH_ROOT" config --get remote.origin.url 2>/dev/null || true)
    case "$url" in
        https://github.com/*) url=${url#https://github.com/} ;;
        git@github.com:*)     url=${url#git@github.com:} ;;
        ssh://git@github.com/*) url=${url#ssh://git@github.com/} ;;
        *) echo ""; return ;;
    esac
    url=${url%.git}
    case "$url" in */*) echo "$url" ;; *) echo "" ;; esac
}

UENR_GH_REPO=${UENR_GH_REPO:-${GITHUB_REPOSITORY:-}}
if [ -z "$UENR_GH_REPO" ]; then
    UENR_GH_REPO=$(_uenr_repo_from_git)
fi
if [ -z "$UENR_GH_REPO" ]; then
    UENR_GH_REPO="michaelowusuntim6/project_uenr"
fi
export UENR_GH_REPO

_uenr_token_from_file() {
    local f tok
    for f in "$UENR_GH_ROOT/github_token.txt" \
             "$UENR_GH_ROOT/.github_token" \
             "$UENR_GH_ROOT/config/github_token.txt"; do
        [ -r "$f" ] || continue
        tok=$(tr -d '\r\n' < "$f" 2>/dev/null || true)
        case "$tok" in
            ""|*PASTE*|*your_token*|*YOUR_TOKEN*) continue ;;
        esac
        # Warn (once) if the secret is readable by other users.
        if [ -n "${UENR_GH_PERM_WARNED:-}" ]; then :; else
            local mode
            mode=$(stat -c '%a' "$f" 2>/dev/null || echo "")
            case "$mode" in
                6*|600) : ;;
                "") : ;;
                *) echo "warning: $f has mode $mode; run: chmod 600 $f" >&2
                   export UENR_GH_PERM_WARNED=1 ;;
            esac
        fi
        export GH_TOKEN="$tok"
        export UENR_GH_TOKEN_SOURCE="$f"
        return 0
    done
    return 1
}

uenr_load_gh_token() {
    if [ -n "${GH_TOKEN:-}" ]; then
        export UENR_GH_TOKEN_SOURCE="environment"
        return 0
    fi
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        export GH_TOKEN="$GITHUB_TOKEN"
        export UENR_GH_TOKEN_SOURCE="environment"
        return 0
    fi
    if _uenr_token_from_file; then
        return 0
    fi
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        export UENR_GH_TOKEN_SOURCE="gh auth"
        return 0
    fi
    cat >&2 <<'MSG'
error: GitHub authentication is not configured.
  Create a token at https://github.com/settings/tokens (repo + workflow scope)
  then run one of:
    scripts/github_auth.sh setup      # hidden prompt, writes github_token.txt
    gh auth login                     # use an existing gh session
  or export GH_TOKEN=... in your shell.
MSG
    return 1
}

uenr_gh() {
    uenr_load_gh_token || return 1
    command gh -R "$UENR_GH_REPO" "$@"
}
