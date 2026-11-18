#!/usr/bin/env bash
# Configure or inspect GitHub authentication for UENR orchestration.
# Usage: scripts/github_auth.sh [status|setup|logout]
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
# shellcheck source=gh_env.sh
. "$HERE/gh_env.sh"

need_gh() {
    command -v gh >/dev/null 2>&1 && return 0
    echo "error: GitHub CLI (gh) is not installed." >&2
    echo "  run: python3 install.py --global" >&2
    echo "  or:  sudo apt-get install -y gh" >&2
    return 1
}

show_status() {
    echo "repository: $UENR_GH_REPO"
    if ! need_gh; then return 1; fi
    if uenr_load_gh_token; then
        echo "auth source: ${UENR_GH_TOKEN_SOURCE:-unknown}"
        local login
        login=$(gh api user --jq .login 2>/dev/null || true)
        if [ -n "$login" ]; then
            echo "gh user:     $login"
            gh auth status 2>&1 | sed 's/^/  /' || true
            return 0
        fi
        echo "error: token was found but api.github.com rejected it." >&2
        return 1
    fi
    return 1
}

setup_token() {
    need_gh || return 1
    local f="$ROOT/github_token.txt" tok login
    echo "Create a token with the 'repo' and 'workflow' scopes:"
    echo "  https://github.com/settings/tokens"
    printf "Paste GitHub token (input hidden): "
    IFS= read -r -s tok || true
    echo
    tok=$(printf '%s' "$tok" | tr -d '\r\n')
    if [ -z "$tok" ]; then
        echo "no token entered; nothing changed." >&2
        return 1
    fi
    if ! login=$(GH_TOKEN="$tok" gh api user --jq .login 2>/dev/null); then
        echo "error: token validation failed (check scopes/expiry)." >&2
        return 1
    fi
    umask 077
    printf '%s\n' "$tok" > "$f"
    chmod 600 "$f"
    echo "saved $f (mode 600); authenticated as $login"
    echo "use 'scripts/github_auth.sh status' to re-check."
}

logout_token() {
    local f removed=0
    for f in "$ROOT/github_token.txt" "$ROOT/.github_token" "$ROOT/config/github_token.txt"; do
        if [ -f "$f" ]; then rm -f "$f"; echo "removed $f"; removed=1; fi
    done
    [ "$removed" = 1 ] || echo "no token file to remove (gh auth session untouched)"
}

case "${1:-status}" in
    status|check) show_status ;;
    setup|login)  setup_token ;;
    logout)       logout_token ;;
    *) echo "usage: $0 [status|setup|logout]" >&2; exit 2 ;;
esac
