#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Running Pylint"

docker/exec.sh pylint src/* test/unit_tests

echo "$0 completed successfully!"
