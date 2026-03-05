#!/bin/bash

#exit on error
set -e

# create and activate the virtual environment
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    uv venv --python=3.11
fi
source .venv/bin/activate

# build the C libraries
mkdir -p rlp/lib
cd puzzle_env/rlp/lib
cmake ../../puzzles
TMP_MAKEFLAGS=$MAKEFLAGS
export MAKEFLAGS='-j 1'
make icons
export MAKEFLAGS=$TMP_MAKEFLAGS
make
cd ../../..

# install rlp and its dependencies
uv pip install -e ./puzzle_env
# uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
