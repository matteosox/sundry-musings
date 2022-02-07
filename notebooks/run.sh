#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR"/../docker/strict_mode.sh

# Opens up a Jupyter notebook in a Docker container

GIT_SHA=$(git rev-parse --short HEAD)
PORT=8888
IP=0.0.0.0

usage() {
    echo "usage: run.sh [--git-sha -g sha=$GIT_SHA]"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -g | --git-sha)
            GIT_SHA="$2"
            shift 2
            ;;
        -h | --help)
            usage
            exit
            ;;
        *)
            echo "Invalid inputs, see below"
            usage
            exit 1
            ;;
    esac
done

echo "Opening up a Jupyter notebook in your browser at http://$IP:$PORT for git sha $GIT_SHA"

open_browser() {
    sleep 2
    python -m webbrowser http://"$IP":"$PORT"
}

open_browser &
docker/exec.sh jupyter notebook \
    --allow-root \
    --ip="$IP" \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.notebook_dir=/root/sundry_musings/notebooks \
    --port="$PORT"
