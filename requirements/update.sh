#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Updating requirements"

docker/exec.sh requirements/inner_update.sh

echo "$0 completed successfully!"
