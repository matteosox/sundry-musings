#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Running shfmt"

if ! docker/exec.sh shfmt "$@" .; then
    echo "shfmt check failed. Run \`test/shfmt.sh -w\` to resolve."
    exit 1
fi

echo "$0 completed successfully!"
