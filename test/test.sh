#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR"/../docker/strict_mode.sh

echo "Running tests"

NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
LIGHT_CYAN='\033[0;36m'

NOT_STARTED="${RED}NOT STARTED${LIGHT_CYAN}"
RUNNING="${RED}RUNNING${LIGHT_CYAN}"
FAILED="${RED}FAILED${LIGHT_CYAN}"
SUCCESS="${GREEN}SUCCESS${LIGHT_CYAN}"

REQUIREMENTS_STATUS="$NOT_STARTED"
BLACK_STATUS="$NOT_STARTED"
ISORT_STATUS="$NOT_STARTED"
SHFMT_STATUS="$NOT_STARTED"
PYLINT_STATUS="$NOT_STARTED"
SHELLCHECK_STATUS="$NOT_STARTED"
MYPY_STATUS="$NOT_STARTED"
UNIT_TESTS_STATUS="$NOT_STARTED"
EXIT_CODE=0

report_status() {
    if [[ "$EXIT_CODE" == 0 ]]; then
        TEST_STATUS="$SUCCESS"
    else
        TEST_STATUS=" $FAILED"
    fi

    echo -e "${LIGHT_CYAN}
Test Summary: $TEST_STATUS
=====================
  - Requirements: $REQUIREMENTS_STATUS
  - Black: $BLACK_STATUS
  - isort: $ISORT_STATUS
  - shfmt: $SHFMT_STATUS
  - Pylint: $PYLINT_STATUS
  - ShellCheck: $SHELLCHECK_STATUS
  - Mypy: $MYPY_STATUS
  - Unit Tests: $UNIT_TESTS_STATUS
${NC}"
}
trap report_status EXIT

echo -e "${LIGHT_CYAN}
Running requirements check
==========================
${NC}"
REQUIREMENTS_STATUS="$RUNNING"
if test/requirements.sh; then
    REQUIREMENTS_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    REQUIREMENTS_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
Requirements check completed w/ status: $REQUIREMENTS_STATUS


Running Black
=============
${NC}"
BLACK_STATUS="$RUNNING"
if test/black.sh --diff --check; then
    BLACK_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    BLACK_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
Black completed w/ status: $BLACK_STATUS


Running isort
=============
${NC}"
ISORT_STATUS="$RUNNING"
if test/isort.sh --check-only; then
    ISORT_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    ISORT_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
isort completed w/ status: $ISORT_STATUS


Running shfmt
=============
${NC}"
SHFMT_STATUS="$RUNNING"
if test/shfmt.sh -d; then
    SHFMT_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    SHFMT_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
shfmt completed w/ status: $SHFMT_STATUS


Running Pylint
==============
${NC}"
PYLINT_STATUS="$RUNNING"
if test/pylint.sh; then
    PYLINT_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    PYLINT_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
Pylint completed w/ status: $PYLINT_STATUS


Running ShellCheck
==================
${NC}"
SHELLCHECK_STATUS="$RUNNING"
if test/shellcheck.sh; then
    SHELLCHECK_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    SHELLCHECK_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
ShellCheck completed w/ status: $SHELLCHECK_STATUS


Running Mypy
============
${NC}"
MYPY_STATUS="$RUNNING"
if test/mypy.sh; then
    MYPY_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    MYPY_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
Mypy completed w/ status: $MYPY_STATUS


Running unit tests
==================
${NC}"
UNIT_TESTS_STATUS="$RUNNING"
if test/unit_tests.sh; then
    UNIT_TESTS_STATUS="$SUCCESS"
else
    EXIT_CODE=1
    UNIT_TESTS_STATUS="$FAILED"
fi
echo -e "${LIGHT_CYAN}
Unit tests completed w/ status: $UNIT_TESTS_STATUS

${NC}"
exit $EXIT_CODE
