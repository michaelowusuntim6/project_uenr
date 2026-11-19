#!/usr/bin/env bash
# Live GitHub Actions monitor without a full-screen TUI.
#
#   scripts/watch_run.sh [RUN_ID] [--interval SECONDS]
#
# - Plain line output, one update per interval (default 10 s; not 3 s).
# - Ctrl+C stops watching immediately; the run continues on GitHub.
# - Echo and mouse reporting are disabled while watching, so wheel/arrow
#   escape sequences cannot leak into the terminal as "[[A" garbage.
#
# Environment: UENR_WATCH_INTERVAL (default 10), GITHUB_REPOSITORY.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=gh_env.sh
. "$HERE/gh_env.sh"

RUN_ID=""
INTERVAL=${UENR_WATCH_INTERVAL:-10}
while [ $# -gt 0 ]; do
    case "$1" in
        --interval) INTERVAL=${2:?--interval needs a value}; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "watch: unknown option $1" >&2; exit 2 ;;
        *) [ -z "$RUN_ID" ] && RUN_ID=$1 || { echo "watch: extra argument $1" >&2; exit 2; }; shift ;;
    esac
done
case "$INTERVAL" in ''|*[!0-9]*) echo "watch: interval must be an integer" >&2; exit 2 ;; esac
[ "$INTERVAL" -ge 2 ] || INTERVAL=2

uenr_load_gh_token || exit 1

if [ -z "$RUN_ID" ]; then
    RUN_ID=$(gh run list -R "$UENR_GH_REPO" --workflow=train.yml --limit 1 \
             --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null || true)
    [ -n "$RUN_ID" ] || { echo "watch: no workflow run found" >&2; exit 2; }
fi

if ! gh run view "$RUN_ID" -R "$UENR_GH_REPO" --json databaseId >/dev/null 2>&1; then
    echo "watch: run $RUN_ID not found or not accessible" >&2
    exit 2
fi

TTY_SAVE=""
STOPPED=0
term_reset() {
    if [ -t 1 ]; then
        printf '\033[?1000l\033[?1002l\033[?1003l\033[?1006l\033[?25h'
    fi
    return 0
}
cleanup() {
    term_reset
    if [ -n "$TTY_SAVE" ]; then stty "$TTY_SAVE" 2>/dev/null || true; fi
}
on_signal() {
    STOPPED=1
    cleanup
    printf '\nwatch: stopped; the workflow keeps running on GitHub.\n'
    exit 130
}
trap cleanup EXIT
trap on_signal INT TERM HUP

# A previous TUI watcher may have left mouse/cursor modes enabled.
term_reset
if [ -t 0 ]; then
    TTY_SAVE=$(stty -g 2>/dev/null || true)
    stty -echo 2>/dev/null || true
fi

echo "watch: run $RUN_ID (every ${INTERVAL}s; Ctrl+C stops watching)"
last=""
last_hb=0
fail=0
while :; do
    now=$(date +%s)
    info=$(gh run view "$RUN_ID" -R "$UENR_GH_REPO" \
        --json status,conclusion,url,jobs \
        --jq '"\(.status)|\(.conclusion // "-")|\(.url)|\([.jobs[]|select(.status=="completed")]|length)/\(.jobs|length)|\([.jobs[]|select(.status=="in_progress")|.name]|join(","))|\([.jobs[]|select(.status=="queued")|.name]|join(","))|\([.jobs[]|select(.conclusion=="failure")|.name]|join(","))"' \
        2>/dev/null) || info=""
    if [ -z "$info" ]; then
        if [ $((now - last_hb)) -ge 60 ]; then
            echo "[$(date +%H:%M:%S)] network/API hiccup; retrying..."
            last_hb=$now
        fi
    else
        IFS='|' read -r status conclusion url done running queued failed <<<"$info"
        line="$status $done running=[$running] queued=[$queued] failed=[$failed]"
        if [ "$line" != "$last" ] || [ $((now - last_hb)) -ge 60 ]; then
            printf '[%s] %s\n' "$(date +%H:%M:%S)" "$line"
            last=$line
            last_hb=$now
        fi
        if [ "$status" = "completed" ]; then
            cleanup
            echo "watch: completed: ${conclusion:-unknown}"
            echo "watch: $url"
            case "$conclusion" in
                success) exit 0 ;;
                cancelled) exit 2 ;;
                *) exit 1 ;;
            esac
        fi
    fi
    sleep "$INTERVAL"
done
