#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Running ShellCheck"

docker/exec.sh test/inner_shellcheck.sh

echo "$0 completed successfully!"
