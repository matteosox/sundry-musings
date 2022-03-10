#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR"/strict_mode.sh

# Execute a command inside docker

EXEC_CMD=(--privileged)
NAME="sundry-musings-$GIT_SHA"

usage() {
    echo "usage: exec.sh [--env -e ENV] [CMD ...]"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -e | --env)
            EXEC_CMD+=("--env" "$2")
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
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

if [[ "$#" = 0 ]]; then
    EXEC_CMD+=(--interactive --tty "$NAME" bash)
else
    cp docker/inner_exec.sh .
    printf "%q " "$@" | sed 's/.$/\n/' >> inner_exec.sh
    EXEC_CMD+=("$NAME" ./inner_exec.sh)
fi

restart_timer() {
    touch setup.cfg
    rm -f inner_exec.sh
}
trap restart_timer EXIT

docker exec "${EXEC_CMD[@]}"
