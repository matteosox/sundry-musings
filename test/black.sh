#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Running Black"

if ! docker/exec.sh black "$@" .; then
    echo "black check failed. Run \`test/black.sh\` to resolve."
    exit 1
fi

echo "$0 completed successfully!"
