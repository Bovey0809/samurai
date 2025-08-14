#!/bin/bash
# Source this script to set up SAM2 environment variables

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.6"
export APP_ROOT=~/sam2

# Add CUDA to PATH if it exists
if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="${PATH}:/usr/local/cuda/bin"
fi

# Add user's local bin to PATH for pip-installed packages
export PATH="${PATH}:${HOME}/.local/bin"

echo "SAM2 environment variables set"
echo "  - CUDA_HOME: $CUDA_HOME"
echo "  - TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  - APP_ROOT: $APP_ROOT"
