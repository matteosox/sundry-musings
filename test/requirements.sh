#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

echo "Checking requirements.txt to make sure it is updated appropriately"

REQS_IN_FILES='^requirements/update\.sh$
^requirements/inner_update\.sh$
^requirements/requirements\.in$'
REQS_OUT_FILE='^requirements/requirements\.txt$'
DIFF_FILES=$(git diff --staged --name-only)

if echo "$DIFF_FILES" | grep --quiet --regexp="$REQS_IN_FILES"; then
    if echo "$DIFF_FILES" | grep --quiet --regexp="$REQS_OUT_FILE"; then
        exit 0
    else
        echo "
Requirements input files updated, but $REQS_OUT_FILE unchanged.
You must update the requirements file by running requirements/update.sh
before committing.
"
        exit 1
    fi
fi

echo "$0 completed successfully!"
