#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Running unit tests"
docker/exec.sh python -m pytest

echo "$0 completed successfully!"
