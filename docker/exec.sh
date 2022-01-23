#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR"/strict_mode.sh

# Execute a command inside docker

OPTS=()
NAME="sundry-musings-$GIT_SHA"

usage() {
    echo "usage: exec.sh [--env -e ENV] [CMD ...]"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -e | --env)
            OPTS+=("--env" "$2")
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            IFS=" " read -r -a CMD <<< "$@"
            break
            ;;
    esac
done

STATE=$(docker ps --all --filter "name=$NAME" --format "{{.State}}")

if [[ -z "$STATE" ]]; then
    echo "No $NAME Docker container found, so creating it for you"
    docker/create_container.sh
fi

if [[ "$STATE" == "paused" ]]; then
    echo "Unpausing Docker container"
    docker unpause "$NAME"
elif [[ "$STATE" == "exited" ]]; then
    echo "Starting Docker container"
    docker start "$NAME"
fi

if [[ -z "${CMD[0]-}" ]]; then
    OPTS+=("--interactive" "--tty")
    CMD=("bash")
fi

OPTS+=("$NAME")
EXEC_CMD=("${OPTS[@]}" "${CMD[@]}")

restart_timer() {
    touch README.md
}
trap restart_timer EXIT

docker exec "${EXEC_CMD[@]}"
