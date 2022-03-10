#! /usr/bin/env bash
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

# Create env
mamba create --name myenv --file conda-linux-64.lock

# Install conda-pack
mamba install conda-pack

# Pack env into /venv
conda-pack --name myenv --output /tmp/myenv.tar
mkdir /venv
cd /venv
tar xf /tmp/myenv.tar
rm /tmp/myenv.tar
/venv/bin/conda-unpack
