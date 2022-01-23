#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Running isort"

if ! docker/exec.sh isort . "$@"; then
    echo "isort check failed. Run \`test/isort.sh\` to resolve."
    exit 1
fi

echo "$0 completed successfully!"
