#! /usr/bin/env bash
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

VIRTUAL_ENV=/root/.venv

# Create env
mamba create --copy --name myenv --file conda-linux-64.lock

# Install conda-pack
mamba install conda-pack

# Pack env into /venv
conda-pack --name myenv --output /tmp/myenv.tar
mkdir "$VIRTUAL_ENV"
cd "$VIRTUAL_ENV"
tar xf /tmp/myenv.tar
rm /tmp/myenv.tar
"$VIRTUAL_ENV"/bin/conda-unpack
