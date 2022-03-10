#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

# Inner shell script for updating requirements

export CUSTOM_COMPILE_COMMAND="requirements/update.sh"

cd requirements || exit 1

pip-compile --allow-unsafe --verbose requirements.in > requirements.txt

conda-lock --file environment.yaml --platform linux-64 --kind explicit
