#! /usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR"/strict_mode.sh

# Build Docker image

export DOCKER_BUILDKIT=1
echo "Building Docker image"
docker build \
    --progress=plain \
    --tag sundry-musings:"$GIT_SHA" \
    --file docker/Dockerfile \
    .

docker rm -f "sundry-musings-$GIT_SHA" &> /dev/null || true

echo "$0 completed successfully!"
