#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/../docker/strict_mode.sh"

# Inner shell script for linting shell scripts

EXIT_CODE=0
SHELL_SCRIPTS=$(git ls-files | grep ".*\.sh$")
declare -A STATUSES

report_status() {
    echo "Shellcheck Summary"
    for script in $SHELL_SCRIPTS; do
        status=${STATUSES[$script]:-"NOT STARTED"}
        echo "  - $script: $status"
    done
}
trap report_status EXIT

for script in $SHELL_SCRIPTS; do
    STATUSES[$script]=RUNNING
    if shellcheck -x "$script"; then
        STATUSES[$script]=SUCCESS
    else
        EXIT_CODE=1
        STATUSES[$script]=FAILED
    fi
done

exit $EXIT_CODE
