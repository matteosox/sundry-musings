#! /usr/bin/env bash
# Shared settings for shell scripts in this repo.
# - Changes directory to the root of the repo
# - Exposes a `GIT_SHA` environment variable
# - Enables "unofficial strict mode", read more at
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
# and https://ss64.com/bash/set.html

set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"/..
cd "$REPO_DIR"

# shellcheck disable=SC2034
GIT_SHA=$(git rev-parse --short HEAD)
