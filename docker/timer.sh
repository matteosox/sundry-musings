#! /usr/bin/env bash
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

TIMER_FILE="setup.cfg"
INTERVAL=300

start_timer() {
    touch "$TIMER_FILE"
}

get_timer() {
    PIDS=$(ps -A --format pid --no-headers)
    NUM_PIDS=$(echo "$PIDS" | wc -l)
    if [[ "$NUM_PIDS" -gt 3 ]]; then
        # Other processes detected, so restarting timer
        start_timer
        ELAPSED=0
    elif ! THEN=$(date +%s -r "$TIMER_FILE"); then
        # No timer found, so restarting it
        start_timer
        ELAPSED=0
    else
        NOW=$(date +%s)
        ELAPSED=$(("$NOW" - "$THEN"))
    fi
    echo "$ELAPSED"
}

start_timer

while sleep "${SLEEP_SECS-$INTERVAL}"; do
    ELAPSED=$(get_timer)
    if [[ "$ELAPSED" -ge "$INTERVAL" ]]; then
        exit 0
    fi
    SLEEP_SECS=$(("$INTERVAL" - "$ELAPSED"))
done
