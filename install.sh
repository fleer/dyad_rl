#!/bin/bash

#exit on error
set -e

# create and activate the virtual environment
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
fi
source .venv/bin/activate

# build the C libraries
mkdir -p rlp/lib
cd rlp/lib
cmake ../../puzzles
TMP_MAKEFLAGS=$MAKEFLAGS
export MAKEFLAGS='-j 1'
make icons
export MAKEFLAGS=$TMP_MAKEFLAGS
make
cd ../..

# install rlp and its dependencies
pip install -e .
pip install -r requirements.txt
