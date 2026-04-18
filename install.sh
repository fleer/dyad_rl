#!/bin/bash

#exit on error
set -e

# create and activate the virtual environment
if [ -d ".venv" ]; then
  echo "Virtual environment already exists. Skipping creation."
else
  echo "Creating virtual environment..."
  uv venv --python=3.11 ~/dyad_rl
fi
source .venv/bin/activate

# build the C libraries
mkdir -p puzzle_env/rlp/lib
cd puzzle_env/rlp/lib
cmake ../../puzzles
TMP_MAKEFLAGS=$MAKEFLAGS
export MAKEFLAGS='-j 1'
if grep -q "^icons:" Makefile; then
  make icons
else
  echo "Skipping 'make icons' (icons target not available on this platform)."
fi
export MAKEFLAGS=$TMP_MAKEFLAGS
make
cd ../../..

# install rlp and its dependencies
# uv pip install -e ./puzzle_env
# uv pip install torch torchvision torchaudio
# # Uncomment for AMD GPUs (ROCm 7.1)
# # uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1
# uv pip install -r requirements.txt
