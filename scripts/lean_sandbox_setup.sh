#!/bin/bash
# Start a NeMo-Skills Lean 4 sandbox container for proof verification.
# Requires Docker-in-Docker (BEAKER_ALLOW_SUBCONTAINERS=1 + mount_docker_socket).
#
# The sandbox exposes /execute on port 6000 and compiles Lean 4 code with Mathlib.
# Build from NeMo-Skills Dockerfile.sandbox or use a pre-built image.

set -e

LEAN_SANDBOX_PORT=${LEAN_SANDBOX_PORT:-6000}
LEAN_SANDBOX_IMAGE=${LEAN_SANDBOX_IMAGE:-""}
LEAN_SANDBOX_MEM_LIMIT=${LEAN_SANDBOX_MEM_LIMIT:-"16g"}

echo "Setting up Lean 4 sandbox on port $LEAN_SANDBOX_PORT..."

# Check if Docker is available
if ! command -v docker &>/dev/null; then
    echo "WARNING: Docker not available. Lean verification will fail."
    echo "Ensure mount_docker_socket=True and BEAKER_ALLOW_SUBCONTAINERS=1"
    export LEAN_SANDBOX_URL="http://127.0.0.1:${LEAN_SANDBOX_PORT}"
    exit 0
fi

# Build from NeMo-Skills if no pre-built image specified
if [ -z "$LEAN_SANDBOX_IMAGE" ]; then
    echo "Building Lean sandbox from NeMo-Skills..."
    LEAN_SANDBOX_IMAGE="lean4-sandbox:local"

    if [ ! -d "/tmp/nemo-skills" ]; then
        git clone --depth 1 https://github.com/NVIDIA-NeMo/Skills.git /tmp/nemo-skills
    fi

    cd /tmp/nemo-skills
    docker build \
        --tag="${LEAN_SANDBOX_IMAGE}" \
        --build-arg="NUM_WORKERS=$(nproc --all)" \
        -f dockerfiles/Dockerfile.sandbox .
    cd -
fi

# Start the sandbox container
echo "Starting Lean sandbox container: $LEAN_SANDBOX_IMAGE"
docker run -d \
    --network=host \
    --memory="${LEAN_SANDBOX_MEM_LIMIT}" \
    --name=lean4-sandbox \
    "${LEAN_SANDBOX_IMAGE}" || {
    echo "Container may already be running, checking..."
    docker start lean4-sandbox 2>/dev/null || true
}

# Wait for it to be ready
echo "Waiting for Lean sandbox to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${LEAN_SANDBOX_PORT}/health" >/dev/null 2>&1; then
        echo "Lean sandbox is ready on port $LEAN_SANDBOX_PORT"
        break
    fi
    sleep 5
done

export LEAN_SANDBOX_URL="http://127.0.0.1:${LEAN_SANDBOX_PORT}"
echo "LEAN_SANDBOX_URL=$LEAN_SANDBOX_URL"
