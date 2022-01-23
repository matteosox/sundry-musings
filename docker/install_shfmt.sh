#! /usr/bin/env bash
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

echo "Installing shfmt"

# shfmt doesn't have an apt package, so we install it using
# webinstaller, but unfortunately that requires quite a bit of
# cleanup after to remove unnecessary stuff, and also put
# shfmt in a place on the PATH for non-login shells

cleanup() {
    rm -rf .config/envman
    rm -rf .local/bin/pathman .local/bin/shfmt .local/bin/webi
    rm -rf .local/opt/pathman-* .local/opt/shfmt-*
    rm -rf .local/share/virtualenv
    rm -rf Downloads
}
trap cleanup EXIT

curl -sS https://webinstall.dev/shfmt@v3.4.1 | bash
mv ~/.local/opt/shfmt-v3.4.1/bin/shfmt /usr/bin/
